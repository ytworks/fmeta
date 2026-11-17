"""Place a funnel automatically from a bound pose.

Given a protein with a ligand in its binding site, find the direction in which
the ligand can leave towards bulk solvent, pick protein atoms that pin the
funnel axis to that direction, and size the funnel from the pocket geometry.

The idea of anchoring the axis on alpha carbons behind and around the site
follows BioSimSpace's ``makeFunnel``; the exit search differs. Instead of
averaging free points on a small grid around the ligand, straight paths from
the ligand are scored by how much protein (or membrane) they cross and how soon
they reach bulk. That keeps working for buried sites and for membrane proteins,
where "free of protein" is not the same as "in water".
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import openmm.app as app
import openmm.unit as unit
from scipy import ndimage
from scipy.optimize import nnls

from ..system.selection import LigandSpec, Selection, classify
from .geometry import Funnel, centroid, funnel_frame
from .obstacles import MembraneSlab, Obstacles
from .validate import ERROR, OK, UNBOUND_WINDOW, WARNING, FunnelReport, funnel_points, validate_funnel


@dataclass
class AutoFunnelSettings:
    method: str = "clearance"        # "clearance" (default) or "bss" (grid average, fixed lengths)
    n_directions: int = 1000
    max_path: float = 4.5            # nm scanned from the binding site
    step: float = 0.05
    probe: float = 0.3               # nm; a path closer than this to an atom is squeezing through
    contact_clearance: float = 0.6   # nm; beyond this the ligand centroid has left the surface
    bulk_margin: float = 1.0         # nm between ligand atoms and solute that counts as bulk
    charged_bulk_margin: float = 1.5 # ... for a ligand with a net charge (long-range attraction)
    anchor_min_atoms: int = 10
    anchor_radius: float = 1.0       # nm around each anchor target searched for alpha carbons
    anchor_separation: float = 1.2   # nm between the p0 and p1 centroids
    pose_margin: float = 0.2         # nm kept between the bound pose and the wall
    min_width: float = 0.6
    max_width: float = 1.5
    exit_direction: Optional[Sequence[float]] = None   # manual override of the exit search
    check_channel: bool = True       # flood-fill test that the site connects to bulk


def _as_nm(positions) -> np.ndarray:
    if unit.is_quantity(positions):
        positions = positions.value_in_unit(unit.nanometer)
    return np.asarray(positions, dtype=float)


def fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` nearly uniform unit vectors."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    phi = k * np.pi * (3.0 - np.sqrt(5.0))
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _first_persistent(values: np.ndarray, threshold: float) -> np.ndarray:
    """For each row, the first index from which ``values`` stays >= threshold to
    the end of the row, or -1."""
    tail_min = np.minimum.accumulate(values[:, ::-1], axis=1)[:, ::-1]
    reached = tail_min >= threshold
    index = np.argmax(reached, axis=1)
    index[~reached.any(axis=1)] = -1
    return index


def find_exit_direction(center: np.ndarray, obstacles: Obstacles, bulk_clearance: float,
                        settings: AutoFunnelSettings) -> tuple[np.ndarray, dict]:
    """Direction from ``center`` to bulk that crosses the least protein.

    Each straight path is charged for the length over which it passes closer
    than ``probe`` to an obstacle, plus a small term for how far it has to go
    to reach bulk (no obstacle within ``bulk_clearance`` from there on).
    """
    directions = fibonacci_sphere(settings.n_directions)
    t = np.arange(settings.step, settings.max_path + 0.5 * settings.step, settings.step)
    points = center + t[None, :, None] * directions[:, None, :]
    clearance = obstacles.clearance(points)
    exit_index = _first_persistent(clearance, bulk_clearance)
    valid = exit_index >= 0
    if not valid.any():
        raise ValueError(f"no straight path reaches bulk within {settings.max_path} nm of the "
                         "binding site; increase max_path or give exit_direction")
    squeeze = np.maximum(settings.probe - clearance, 0.0) * settings.step
    before_exit = np.arange(t.size)[None, :] < exit_index[:, None]
    blocking = np.where(valid, (squeeze * before_exit).sum(axis=1), np.inf)
    exit_distance = np.where(valid, t[np.clip(exit_index, 0, None)], np.inf)
    cost = blocking + 0.02 * exit_distance

    best = int(np.argmin(cost))
    near = (directions @ directions[best] > np.cos(np.radians(20.0))) & (cost <= cost[best] + 0.02)
    mean = directions[near].mean(axis=0)
    direction = mean / np.linalg.norm(mean)

    # distinct alternatives, best first, for the report
    alternatives, order = [], np.argsort(cost)
    for index in order:
        if not np.isfinite(cost[index]) or len(alternatives) == 3:
            break
        if all(directions[index] @ other["direction"] < np.cos(np.radians(45.0))
               for other in alternatives):
            alternatives.append(dict(direction=directions[index], blocking=float(blocking[index]),
                                     exit_distance=float(exit_distance[index])))
    info = dict(blocking=float(blocking[best]), exit_distance=float(exit_distance[best]),
                alternatives=[dict(direction=[round(float(x), 3) for x in a["direction"]],
                                   blocking=round(a["blocking"], 4),
                                   exit_distance=round(a["exit_distance"], 2))
                              for a in alternatives])
    return direction, info


def channel_is_open(center: np.ndarray, ligand_coords: np.ndarray, obstacles: Obstacles,
                    bulk_clearance: float, probe: float = 0.3, extent: float = 4.5,
                    spacing: float = 0.1) -> bool:
    """Whether a probe sphere can travel from the site to bulk (voxel flood fill).

    Unlike the straight-line search this follows curved channels, so it tells a
    genuinely closed site from one whose exit merely bends.
    """
    axis = np.arange(-extent, extent + 0.5 * spacing, spacing)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1) + center
    clearance = obstacles.clearance(grid)
    free = clearance >= probe
    # the ligand's own volume is free by construction
    ligand = Obstacles(ligand_coords)
    free |= ligand.clearance(grid) <= 0.25
    labels, _ = ndimage.label(free)
    middle = len(axis) // 2
    site_label = labels[middle, middle, middle]
    if site_label == 0:
        return False
    return bool(((labels == site_label) & (clearance >= bulk_clearance)).any())


def structured_ca_mask(topology: app.Topology, positions: np.ndarray,
                       ca_atoms: np.ndarray) -> np.ndarray:
    """Alpha carbons that sit in a helix or strand, judged from CA-CA distances
    alone (i to i+2, i+3, i+4), so no external secondary-structure tool is needed."""
    index_of = {int(atom): k for k, atom in enumerate(ca_atoms)}
    mask = np.zeros(len(ca_atoms), dtype=bool)
    for chain in topology.chains():
        chain_ca = [atom.index for residue in chain.residues() for atom in residue.atoms()
                    if atom.index in index_of]
        xyz = positions[chain_ca]
        for i in range(len(chain_ca) - 4):
            d2, d3, d4 = (np.linalg.norm(xyz[i + k] - xyz[i]) for k in (2, 3, 4))
            helix = 0.50 <= d2 <= 0.60 and 0.45 <= d3 <= 0.59 and 0.55 <= d4 <= 0.70
            strand = 0.60 <= d2 <= 0.72 and d3 >= 0.90 and d4 >= 1.15
            if helix or strand:
                for k in range(5):
                    mask[index_of[chain_ca[i + k]]] = True
    return mask


def _anchor_weights(coords: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Non-negative weights, as uniform as possible, whose centroid hits ``target``."""
    n = len(coords)
    uniformity = 1.0e-2
    for _ in range(4):
        a = np.vstack([coords.T, 10.0 * np.ones((1, n)), np.sqrt(uniformity) * np.eye(n)])
        b = np.concatenate([target, [10.0], np.full(n, np.sqrt(uniformity) / n)])
        weights, _ = nnls(a, b)
        weights /= weights.sum()
        if 1.0 / (weights**2).sum() >= min(4.0, n):   # enough atoms share the load
            break
        uniformity *= 10.0
    return weights


def _pick_anchor(target: np.ndarray, positions: np.ndarray, ca_atoms: np.ndarray,
                 preferred: np.ndarray, settings: AutoFunnelSettings):
    """Choose alpha carbons around ``target`` and weight them onto it."""
    for pool in (ca_atoms[preferred], ca_atoms):
        if len(pool) == 0:
            continue
        distance = np.linalg.norm(positions[pool] - target, axis=1)
        chosen = pool[distance <= settings.anchor_radius]
        if len(chosen) >= settings.anchor_min_atoms:
            break
    else:
        # sparse protein around the target: take the nearest ones instead
        distance = np.linalg.norm(positions[ca_atoms] - target, axis=1)
        nearest = np.argsort(distance)[:settings.anchor_min_atoms]
        if len(nearest) < 3 or distance[nearest[min(2, len(nearest) - 1)]] > 2.5:
            raise ValueError("too few alpha carbons near the binding site to anchor the funnel")
        chosen = ca_atoms[np.sort(nearest)]
    weights = _anchor_weights(positions[chosen], target)
    keep = weights > 1.0e-4
    return [int(i) for i in chosen[keep]], [float(w) for w in weights[keep] / weights[keep].sum()]


def _bss_direction(center: np.ndarray, protein: Obstacles, positions: np.ndarray,
                   ca_atoms: np.ndarray) -> np.ndarray:
    """BioSimSpace-style direction: from the pocket's alpha carbons to the mean
    of the grid points (5x5x5 over a 2 nm cube) that have no protein within 0.2 nm."""
    offsets = np.linspace(-1.0, 1.0, 5)
    grid = np.stack(np.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = grid + center
    free = grid[protein.clearance(grid) > 0.2]
    near = ca_atoms[np.linalg.norm(positions[ca_atoms] - center, axis=1) <= 1.0]
    if len(free) == 0 or len(near) == 0:
        raise ValueError("BSS-style funnel placement found no free grid points or alpha carbons")
    vector = free.mean(axis=0) - positions[near].mean(axis=0)
    return vector / np.linalg.norm(vector)


def auto_funnel(topology: app.Topology, positions, ligand: LigandSpec = None,
                membrane: Optional[MembraneSlab] = None,
                settings: Optional[AutoFunnelSettings] = None,
                selection: Optional[Selection] = None,
                ligand_charge: Optional[float] = None,
                **funnel_overrides) -> tuple[Funnel, FunnelReport]:
    """Define a funnel for the ligand pose in ``positions``.

    ``membrane`` describes a bilayer that is not part of the structure yet (the
    usual case: the funnel is placed on the dry complex and the box is sized
    from it). Lipids already present are obstacles automatically. Keyword
    arguments override individual funnel parameters (e.g. ``upper=4.0``).

    ``ligand_charge`` (in e), if known, lengthens the funnel for charged
    ligands: their attraction to the protein dies off slowly, so the unbound
    region has to sit further out before the free-energy profile is flat.

    Returns the funnel and a report; check ``report.ok`` or call
    ``report.raise_for_errors()``.
    """
    settings = settings or AutoFunnelSettings()
    positions = _as_nm(positions)
    selection = selection or classify(topology, ligand)
    report = FunnelReport()
    if ligand_charge is not None and abs(ligand_charge) > 0.5:
        margin = max(settings.bulk_margin, settings.charged_bulk_margin)
        settings = dataclasses.replace(settings, bulk_margin=margin)
        report.add("ligand_charge", WARNING,
                   f"the ligand carries a net charge of {ligand_charge:+.0f} e: the unbound region "
                   f"is placed {margin:.1f} nm from the solute instead of 1.0 nm. Check that W(s) "
                   "is flat there in the analysis, and raise the upper wall if it is not")

    ligand_heavy = selection.heavy_atoms("ligand")
    if len(ligand_heavy) == 0:
        raise ValueError("the ligand has no heavy atoms")
    atoms = list(topology.atoms())
    masses = [atoms[i].element.mass.value_in_unit(unit.dalton) for i in ligand_heavy]
    center = centroid(positions, ligand_heavy, masses)
    r_lig = float(np.linalg.norm(positions[ligand_heavy] - center, axis=1).max())
    bulk_clearance = settings.bulk_margin + r_lig

    solute_heavy = np.concatenate([selection.heavy_atoms("protein"), selection.heavy_atoms("other"),
                                   selection.heavy_atoms("lipid")])
    slab = None if selection.lipid.size else membrane
    obstacles = Obstacles(positions[solute_heavy], slab)
    ca_atoms = np.array([i for i in selection.protein if atoms[i].name == "CA"], dtype=int)
    if len(ca_atoms) < 6:
        raise ValueError("found fewer than six protein alpha carbons; cannot anchor a funnel")

    # ---- exit direction -----------------------------------------------------
    info: dict = {}
    if settings.exit_direction is not None:
        direction = np.asarray(settings.exit_direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        report.add("exit", OK, "exit direction given by the user")
    elif settings.method == "bss":
        direction = _bss_direction(center, Obstacles(positions[selection.heavy_atoms("protein")]),
                                   positions, ca_atoms)
        report.add("exit", OK, "exit direction from the BioSimSpace-style grid average")
    elif settings.method == "clearance":
        direction, info = find_exit_direction(center, obstacles, bulk_clearance, settings)
        if info["blocking"] > 0.02:
            report.add("exit", WARNING,
                       "no straight open path from the site to bulk; using the least obstructed "
                       f"direction (blocking {info['blocking']:.2f})")
        else:
            report.add("exit", OK, f"open path to bulk, {info['exit_distance']:.2f} nm long")
    else:
        raise ValueError(f"unknown method {settings.method!r}")

    if settings.check_channel and settings.exit_direction is None:
        open_channel = channel_is_open(center, positions[ligand_heavy], obstacles, bulk_clearance,
                                       extent=settings.max_path)
        info["channel_open"] = open_channel
        if not open_channel:
            report.add("channel", WARNING,
                       "the binding site is closed in this structure (a 0.3 nm probe cannot "
                       "reach bulk); unbinding will need the protein to open")

    # ---- anchors ------------------------------------------------------------
    preferred = structured_ca_mask(topology, positions, ca_atoms)
    if slab is not None or selection.lipid.size:
        # membrane proteins: keep the anchors in the transmembrane region
        centre_z = (membrane.center_z if membrane is not None
                    else float(positions[selection.heavy_atoms("lipid"), 2].mean()))
        half = membrane.half_thickness if membrane is not None else 2.2
        preferred &= np.abs(positions[ca_atoms, 2] - centre_z) <= half + 0.5
    if settings.method == "bss":
        near = ca_atoms[np.linalg.norm(positions[ca_atoms] - center, axis=1) <= 1.0]
        com = positions[near].mean(axis=0)
        behind = ca_atoms[np.linalg.norm(positions[ca_atoms] - (com - direction), axis=1) <= 0.7]
        if len(behind) == 0:
            raise ValueError("BSS-style funnel placement found no alpha carbons behind the site")
        p1_atoms, p1_weights = [int(i) for i in near], None
        p0_atoms, p0_weights = [int(i) for i in behind], None
    else:
        p1_atoms, p1_weights = _pick_anchor(center, positions, ca_atoms, preferred, settings)
        p0_atoms, p0_weights = _pick_anchor(center - settings.anchor_separation * direction,
                                            positions, ca_atoms, preferred, settings)

    funnel = Funnel(ligand_atoms=[int(i) for i in ligand_heavy], ligand_weights=masses,
                    p0_atoms=p0_atoms, p0_weights=p0_weights,
                    p1_atoms=p1_atoms, p1_weights=p1_weights)
    origin, axis, _ = funnel_frame(funnel, positions)
    deviation = float(np.degrees(np.arccos(np.clip(axis @ direction, -1.0, 1.0))))
    if deviation > 20.0:
        report.add("axis", ERROR, f"anchors give an axis {deviation:.0f} deg off the exit direction")
    elif deviation > 8.0:
        report.add("axis", WARNING, f"axis is {deviation:.0f} deg off the exit direction")
    else:
        report.add("axis", OK, f"axis within {deviation:.1f} deg of the exit direction")

    # ---- lengths from the pocket geometry -----------------------------------
    s_bound = float((center - origin) @ axis)
    if settings.method != "bss":
        s_scan = s_bound + np.arange(0.0, settings.max_path + 0.5 * settings.step, settings.step)
        profile = obstacles.clearance(origin + s_scan[:, None] * axis)[None, :]
        i_surface = int(_first_persistent(profile, settings.contact_clearance)[0])
        i_bulk = int(_first_persistent(profile, bulk_clearance)[0])
        if i_bulk < 0:
            raise ValueError("the funnel axis does not reach bulk solvent within "
                             f"{settings.max_path} nm; increase max_path or set exit_direction")
        s_surface, s_bulk = float(s_scan[max(i_surface, 0)]), float(s_scan[i_bulk])
        funnel.inflection = max(s_surface, s_bound + 0.5)
        funnel.upper = s_bulk + 0.6
        funnel.lower = max(s_bound - 0.5, 0.05)
        info.update(s_surface=s_surface, s_bulk=s_bulk)

        relative = positions[ligand_heavy] - origin
        s_atoms = relative @ axis
        rho_atoms = np.linalg.norm(relative - np.outer(s_atoms, axis), axis=1)
        needed = ((rho_atoms + settings.pose_margin - funnel.buffer)
                  * (1.0 + np.exp(funnel.steepness * (s_atoms - funnel.inflection))))
        funnel.width = float(np.clip(needed.max(), settings.min_width, settings.max_width))

        # The wall has a finite radius out there: raise the upper wall until the
        # whole unbound window, not just the axis, is clear of the solute.
        limit = s_bound + settings.max_path + 1.0
        while funnel.upper < limit:
            window = np.linspace(funnel.upper - UNBOUND_WINDOW, funnel.upper, 7)
            gap = obstacles.clearance(funnel_points(funnel, positions, window)).min() - r_lig
            if gap >= settings.bulk_margin:
                break
            funnel.upper += 0.1

    for name, value in funnel_overrides.items():
        if not hasattr(funnel, name):
            raise TypeError(f"unknown funnel parameter {name!r}")
        setattr(funnel, name, value)
    funnel.__post_init__()
    funnel.metadata = dict(method=settings.method,
                           exit_direction=[round(float(x), 4) for x in direction],
                           axis_deviation_deg=round(deviation, 2))

    info.update(exit_direction=[float(x) for x in direction], axis_deviation_deg=deviation,
                ligand_radius=r_lig, bulk_clearance=bulk_clearance)
    report.info.update(info)
    validate_funnel(funnel, positions, selection, slab=membrane, report=report)
    return funnel, report
