"""Checks that a funnel is usable for a given structure (and, once built, box)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..system.selection import Selection
from .geometry import Funnel, centroid, funnel_frame
from .obstacles import MembraneSlab, Obstacles

OK, WARNING, ERROR = "ok", "warning", "error"
UNBOUND_WINDOW = 0.6     # nm below the upper wall that the analysis treats as bulk


@dataclass
class Check:
    name: str
    status: str
    message: str


@dataclass
class FunnelReport:
    checks: list[Check] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def add(self, name: str, status: str, message: str) -> None:
        self.checks.append(Check(name, status, message))

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if c.status == ERROR]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("funnel validation failed:\n" + "\n".join(
                f"  - {c.name}: {c.message}" for c in self.errors))

    def __str__(self) -> str:
        marks = {OK: "ok  ", WARNING: "WARN", ERROR: "FAIL"}
        lines = [f"[{marks[c.status]}] {c.name}: {c.message}" for c in self.checks]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return dict(ok=self.ok, info=self.info,
                    checks=[dict(name=c.name, status=c.status, message=c.message)
                            for c in self.checks])


def ligand_radius(funnel: Funnel, positions: np.ndarray) -> float:
    """Largest distance of a ligand group atom from the ligand centroid (nm)."""
    centre = centroid(positions, funnel.ligand_atoms, funnel.ligand_weights)
    return float(np.linalg.norm(positions[funnel.ligand_atoms] - centre, axis=1).max())


def _perpendicular_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(axis, e1)


def funnel_points(funnel: Funnel, positions: np.ndarray, s_values, n_ring: int = 12,
                  include_axis: bool = True) -> np.ndarray:
    """Points on the funnel wall (and axis) at the given projections."""
    origin, axis, _ = funnel_frame(funnel, positions)
    e1, e2 = _perpendicular_frame(axis)
    angles = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
    points = []
    for s in np.atleast_1d(s_values):
        centre = origin + s * axis
        radius = float(funnel.wall_radius(s))
        if include_axis:
            points.append(centre)
        points.extend(centre + radius * (np.cos(a) * e1 + np.sin(a) * e2) for a in angles)
    return np.array(points)


def _shortest_image_distance(box: np.ndarray) -> float:
    box = np.asarray(box, dtype=float)
    if box.ndim == 1:
        return float(box.min())
    best = np.inf
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                if i == j == k == 0:
                    continue
                best = min(best, float(np.linalg.norm(i * box[0] + j * box[1] + k * box[2])))
    return best


def required_image_distance(funnel: Funnel) -> float:
    """Smallest periodic image distance for which the ligand stays nearer to its own protein
    than to any periodic copy of it, everywhere the walls allow the ligand to go.

    The forces do not need it: they take the coordinates as they are. A trajectory that is
    analysed or made whole by nearest images does."""
    reach = float(np.hypot(funnel.upper + 0.3, funnel.max_radius))
    return 2.0 * (reach + 0.3)


def validate_funnel(funnel: Funnel, positions: np.ndarray, selection: Selection,
                    box: Optional[np.ndarray] = None, slab: Optional[MembraneSlab] = None,
                    cutoff: float = 1.0, report: Optional[FunnelReport] = None,
                    tumbling: bool = True) -> FunnelReport:
    """Run the geometric checks. ``box`` (three lengths or lattice vectors, nm)
    enables the periodic checks; ``slab`` stands in for a membrane not built yet.
    ``tumbling=False`` (membrane systems) applies the rule of
    :func:`required_image_distance` per box axis instead of to the shortest image
    distance, since the solute cannot rotate out of the membrane plane."""
    positions = np.asarray(positions, dtype=float)
    report = report or FunnelReport()
    origin, axis, lig = funnel_frame(funnel, positions)
    r_lig = ligand_radius(funnel, positions)
    obstacle_atoms = np.concatenate([selection.heavy_atoms("protein"),
                                     selection.heavy_atoms("other"),
                                     selection.heavy_atoms("lipid")])
    has_lipids = selection.lipid.size > 0
    obstacles = Obstacles(positions[obstacle_atoms], None if has_lipids else slab)

    # -- bound pose inside the cone ------------------------------------------
    relative = positions[funnel.ligand_atoms] - origin
    s_atoms = relative @ axis
    rho_atoms = np.linalg.norm(relative - np.outer(s_atoms, axis), axis=1)
    margin = float((funnel.wall_radius(s_atoms) - rho_atoms).min())
    s_bound = float((lig - origin) @ axis)
    if margin < 0.0:
        report.add("pose", ERROR, f"ligand atoms lie {-margin:.2f} nm outside the funnel wall")
    elif margin < 0.2:
        report.add("pose", WARNING, f"ligand is only {margin:.2f} nm inside the funnel wall")
    else:
        report.add("pose", OK, f"ligand is {margin:.2f} nm inside the wall at s = {s_bound:.2f} nm")
    if s_bound < funnel.lower + 0.2:
        report.add("lower_wall", ERROR,
                   f"bound pose (s = {s_bound:.2f} nm) is within 0.2 nm of the lower wall")

    # -- anchors -------------------------------------------------------------
    separation = float(np.linalg.norm(
        centroid(positions, funnel.p1_atoms, funnel.p1_weights) - origin))
    for name in ("p0", "p1"):
        atoms = getattr(funnel, f"{name}_atoms")
        weights = np.asarray(getattr(funnel, f"{name}_weights"))
        centre = centroid(positions, atoms, weights)
        spread = float(np.sqrt((weights * ((positions[atoms] - centre) ** 2).sum(axis=1)).sum()
                               / weights.sum()))
        effective = float(weights.sum() ** 2 / (weights**2).sum())
        if spread > 2.0:
            report.add(f"{name}_anchor", ERROR,
                       f"anchor atoms are spread over {spread:.1f} nm; is the input wrapped "
                       "across the periodic box?")
        elif effective < 3.0:
            report.add(f"{name}_anchor", WARNING,
                       f"axis rests on about {effective:.1f} atoms; expect a noisy axis")
        else:
            report.add(f"{name}_anchor", OK, f"{len(atoms)} atoms, effective {effective:.1f}")
    if separation < 0.8:
        report.add("anchor_separation", WARNING,
                   f"p0 and p1 are only {separation:.2f} nm apart; the axis will be noisy")

    # -- unbound region is in bulk -------------------------------------------
    window = np.linspace(funnel.upper - UNBOUND_WINDOW, funnel.upper, 7)
    unbound = funnel_points(funnel, positions, window)
    gap = float(obstacles.clearance(unbound).min()) - r_lig
    if gap < 0.3:
        report.add("unbound_region", ERROR,
                   f"unbound ligand would be {max(gap, 0):.2f} nm from the solute; "
                   "the funnel does not reach bulk solvent")
    elif gap < cutoff:
        report.add("unbound_region", WARNING,
                   f"unbound ligand stays within {gap:.2f} nm of the solute "
                   f"(nonbonded cutoff {cutoff:.1f} nm); consider a higher upper wall")
    else:
        report.add("unbound_region", OK, f"unbound ligand is {gap:.2f} nm clear of the solute")

    # -- membrane ------------------------------------------------------------
    if has_lipids:
        # membrane surface: where 98% of the lipid heavy atoms lie closer to the midplane
        z_lipid = positions[selection.heavy_atoms("lipid"), 2]
        centre_z = float(z_lipid.mean())
        half = float(np.percentile(np.abs(z_lipid - centre_z), 98))
        distance = float((np.abs(unbound[:, 2] - centre_z) - half).min())
        if distance < 0.5:
            report.add("membrane", ERROR,
                       f"unbound region is {distance:.2f} nm from the membrane surface")
        elif distance < 1.0:
            report.add("membrane", WARNING,
                       f"unbound region is only {distance:.2f} nm above the membrane surface")
        else:
            report.add("membrane", OK, f"unbound region is {distance:.2f} nm above the membrane")
        # lipids that ended up in the ligand's way (e.g. a tail packed into the vestibule)
        lipid_xyz = positions[selection.heavy_atoms("lipid")] - origin
        s_lipid = lipid_xyz @ axis
        rho_lipid = np.linalg.norm(lipid_xyz - np.outer(s_lipid, axis), axis=1)
        on_path = (s_lipid >= s_bound) & (s_lipid <= funnel.upper)
        inside = on_path & (rho_lipid < funnel.wall_radius(s_lipid))
        if (on_path & (rho_lipid < 0.3)).any():
            report.add("lipids_in_funnel", ERROR,
                       "lipid atoms sit on the funnel axis between the site and bulk; the exit "
                       "is blocked by the membrane")
        elif inside.any():
            report.add("lipids_in_funnel", WARNING,
                       f"{int(inside.sum())} lipid atoms lie inside the funnel walls; the ligand "
                       "will meet them on its way out")
        else:
            report.add("lipids_in_funnel", OK, "no lipid atoms inside the funnel")

    # -- periodic box --------------------------------------------------------
    if box is not None:
        if tumbling:
            shortest = _shortest_image_distance(box)
            needed = required_image_distance(funnel)
            if shortest < needed:
                report.add("box_cv", ERROR,
                           f"shortest periodic image distance {shortest:.2f} nm < {needed:.2f} nm: "
                           "before the upper wall the ligand would be nearer to a periodic copy "
                           "of the protein than to its own")
            else:
                report.add("box_cv", OK, f"image distance {shortest:.2f} nm >= {needed:.2f} nm")
        else:
            lengths = np.asarray(box, dtype=float)
            lengths = lengths if lengths.ndim == 1 else np.diag(lengths)
            reach = funnel.upper + 0.3
            need_z = 2.0 * (abs(axis[2]) * reach + funnel.max_radius + 0.3)
            need_xy = 2.0 * (np.hypot(axis[0], axis[1]) * reach + funnel.max_radius + 0.3)
            if lengths[2] < need_z or min(lengths[0], lengths[1]) < need_xy:
                report.add("box_cv", ERROR,
                           f"box {lengths[0]:.2f} x {lengths[1]:.2f} x {lengths[2]:.2f} nm is too "
                           f"small: before the upper wall the ligand would be nearer to a periodic "
                           f"copy of the protein than to its own (need xy >= {need_xy:.2f}, "
                           f"z >= {need_z:.2f} nm)")
            else:
                report.add("box_cv", OK, f"box z {lengths[2]:.2f} nm >= {need_z:.2f} nm, "
                                         f"xy >= {need_xy:.2f} nm")
        image_gap = obstacles.nearest_image_distance(unbound, box) - r_lig
        if image_gap < 0.5:
            report.add("box_images", ERROR,
                       f"unbound ligand would be {max(image_gap, 0):.2f} nm from a periodic image")
        elif image_gap < cutoff:
            report.add("box_images", WARNING,
                       f"unbound ligand comes within {image_gap:.2f} nm of a periodic image "
                       f"(cutoff {cutoff:.1f} nm)")
        else:
            report.add("box_images", OK,
                       f"unbound ligand is {image_gap:.2f} nm from the nearest periodic image")

    report.info.update(s_bound=s_bound, ligand_radius=r_lig, anchor_separation=separation)
    return report
