"""Build the simulation system: complex, funnel, then solvent or membrane.

The funnel is placed on the dry complex first, because the periodic box has to
be sized from it: the unbound end of the funnel must sit in bulk, clear of the
periodic images, and the minimum-image CVs must stay valid up to the upper wall.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from scipy.spatial import cKDTree

from ..funnel.autodetect import AutoFunnelSettings, auto_funnel
from ..funnel.forces import add_funnel_restraint
from ..funnel.geometry import Funnel, funnel_frame
from ..funnel.obstacles import MembraneSlab
from ..funnel.validate import (UNBOUND_WINDOW, FunnelReport, funnel_points, ligand_radius,
                               required_image_distance, validate_funnel)
from .inputs import read_ligand, read_structure
from .ligand import register_ligand
from .selection import Selection, classify

log = logging.getLogger(__name__)

# lipids for which a default temperature well above the gel transition matters
GEL_TRANSITION_K = {"DPPC": 314.0, "DMPC": 297.0, "DLPE": 303.0}


@dataclass
class SystemSettings:
    forcefields: Sequence[str] = ("amber14-all.xml", "amber14/tip3p.xml")
    water_model: str = "tip3p"
    ligand_method: str = "openff"            # "openff", "gaff" or "xml"
    ligand_forcefield: Optional[str] = None  # e.g. "openff-2.2.0"; default of the generator if None
    ligand_xml: Sequence[str] = ()           # ffxml file(s) holding the ligand template
    ligand_resname: str = "LIG"
    keep_water: bool = False                 # keep waters present in the protein file
    add_hydrogens: Optional[bool] = None     # None: only if the protein has none
    ph: float = 7.0
    ionic_strength: float = 0.15             # mol/L
    padding: float = 1.2                     # nm of solvent around the solute (lipid in XY for membranes)
    box_shape: str = "dodecahedron"          # or "cube"
    cutoff: float = 1.0                      # nm
    hydrogen_mass: Optional[float] = 1.5     # amu; None disables mass repartitioning
    temperature: float = 300.0
    pressure: float = 1.0                    # bar


@dataclass
class MembraneSettings:
    lipid: str = "POPC"
    center_z: float = 0.0                    # nm, membrane midplane in the input frame
    half_thickness: float = 2.2              # nm, used for funnel placement before the build
    patch: Optional[str] = None              # custom pre-equilibrated patch (PDB) instead of ``lipid``


@dataclass
class Prepared:
    topology: app.Topology
    system: mm.System
    positions: np.ndarray                    # nm
    funnel: Funnel
    selection: Selection
    report: FunnelReport
    restraint_group: int
    membrane: bool
    settings: SystemSettings = field(repr=False, default=None)

    @property
    def box_vectors(self) -> np.ndarray:
        return np.array(self.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer))


# ---- geometry helpers ------------------------------------------------------

def soluble_image_distance(funnel: Funnel, positions: np.ndarray, solute: np.ndarray,
                           cutoff: float, padding: float) -> float:
    """Periodic image distance for a freely tumbling solute.

    Three requirements, each independent of the solute's orientation:
    solvent padding around its bounding sphere; an unbound ligand that stays a
    cutoff (plus 0.5 nm) away from the next image; and minimum-image CVs that
    remain valid up to the upper wall. 3% is added for NPT volume changes.
    """
    coords = positions[solute]
    centre = 0.5 * (coords.min(axis=0) + coords.max(axis=0))
    r_solute = float(np.linalg.norm(coords - centre, axis=1).max())
    window = np.linspace(funnel.upper - UNBOUND_WINDOW, funnel.upper + 0.2, 6)
    reach = float(np.linalg.norm(funnel_points(funnel, positions, window) - centre, axis=1).max())
    reach += ligand_radius(funnel, positions)
    distance = max(2.0 * (r_solute + padding), reach + r_solute + cutoff + 0.5,
                   required_image_distance(funnel))
    return 1.03 * distance


def box_vectors(distance: float, shape: str):
    """Lattice vectors (nm) with nearest-image distance ``distance``."""
    if shape == "cube":
        rows = [[distance, 0, 0], [0, distance, 0], [0, 0, distance]]
    elif shape == "dodecahedron":
        rows = [[distance, 0, 0], [0, distance, 0],
                [0.5 * distance, 0.5 * distance, np.sqrt(0.5) * distance]]
    else:
        raise ValueError(f"unknown box shape {shape!r}")
    return [mm.Vec3(*row) * unit.nanometer for row in rows]


def membrane_box_z(funnel: Funnel, positions: np.ndarray, solute: np.ndarray,
                   cutoff: float, padding: float) -> float:
    """Box height for a membrane system.

    The solute does not tumble, so only z matters: the unbound ligand must stay
    a cutoff (plus 0.5 nm) from the periodic image above or below, the
    z component of the ligand-anchor vector must stay under half the box, and
    the solute needs ordinary solvent padding. 3% is added for NPT.
    """
    coords = positions[solute]
    z_min, z_max = float(coords[:, 2].min()), float(coords[:, 2].max())
    window = np.linspace(funnel.upper - UNBOUND_WINDOW, funnel.upper + 0.2, 6)
    points = funnel_points(funnel, positions, window)
    r_lig = ligand_radius(funnel, positions)
    top, bottom = float(points[:, 2].max()) + r_lig, float(points[:, 2].min()) - r_lig
    span = max(z_max, top) - min(z_min, bottom)
    images = span + cutoff + 0.5
    _, axis, _ = funnel_frame(funnel, positions)
    reach_z = abs(axis[2]) * (funnel.upper + 0.3) + funnel.max_radius
    return 1.03 * max(images, 2.0 * (reach_z + 0.3), (z_max - z_min) + 2.0 * padding)


def remap_atoms(old_positions: np.ndarray, new_positions: np.ndarray,
                atoms: Sequence[int]) -> list[int]:
    """Find, in a rebuilt structure, the atoms that sit where ``atoms`` were.
    Solvation and membrane building keep the solute coordinates but may reorder atoms."""
    tree = cKDTree(new_positions)
    distance, index = tree.query(old_positions[list(atoms)])
    if distance.max() > 1.0e-3:
        raise RuntimeError("solute atoms moved during system building; cannot map the funnel")
    return [int(i) for i in index]


def remap_funnel(funnel: Funnel, old_positions: np.ndarray, new_positions: np.ndarray) -> Funnel:
    data = funnel.to_dict()
    for name in ("ligand_atoms", "p0_atoms", "p1_atoms"):
        data[name] = remap_atoms(old_positions, new_positions, data[name])
    return Funnel.from_dict(data)


def add_barostat(system: mm.System, temperature: float, pressure: float, membrane: bool,
                 frequency: int = 25) -> None:
    """Isotropic barostat, or a semi-isotropic tensionless one for membranes."""
    if membrane:
        barostat = mm.MonteCarloMembraneBarostat(
            pressure * unit.bar, 0.0 * unit.bar * unit.nanometer, temperature * unit.kelvin,
            mm.MonteCarloMembraneBarostat.XYIsotropic, mm.MonteCarloMembraneBarostat.ZFree,
            frequency)
    else:
        barostat = mm.MonteCarloBarostat(pressure * unit.bar, temperature * unit.kelvin, frequency)
    system.addForce(barostat)


def create_system(forcefield: app.ForceField, topology: app.Topology,
                  settings: SystemSettings) -> mm.System:
    kwargs = dict(nonbondedMethod=app.PME, nonbondedCutoff=settings.cutoff * unit.nanometer,
                  constraints=app.HBonds, rigidWater=True)
    if settings.hydrogen_mass:
        kwargs["hydrogenMass"] = settings.hydrogen_mass * unit.amu
    return forcefield.createSystem(topology, **kwargs)


def _built_report(placement: FunnelReport, built: FunnelReport) -> FunnelReport:
    """Report for the finished system: how the funnel was placed (from the dry
    complex) followed by the geometric checks on the built system."""
    kept = [c for c in placement.checks if c.name in ("exit", "channel", "axis")]
    return FunnelReport(kept + built.checks, {**placement.info, **built.info})


# ---- the build -------------------------------------------------------------

def build_complex(protein, ligand, forcefield: app.ForceField,
                  settings: SystemSettings) -> tuple[app.Modeller, list[int]]:
    """Dry protein-ligand complex. Returns the modeller and the ligand atom indices."""
    topology, positions = read_structure(protein)
    modeller = app.Modeller(topology, positions * unit.nanometer)
    if not settings.keep_water:
        modeller.deleteWater()
    has_hydrogens = any(a.element is not None and a.element.symbol == "H"
                        for a in modeller.topology.atoms())
    add = settings.add_hydrogens if settings.add_hydrogens is not None else not has_hydrogens
    if add:
        modeller.addHydrogens(forcefield, pH=settings.ph)
    ligand_topology, ligand_positions = read_ligand(ligand, settings.ligand_resname)
    first = modeller.topology.getNumAtoms()
    modeller.add(ligand_topology, ligand_positions * unit.nanometer)
    return modeller, list(range(first, modeller.topology.getNumAtoms()))


def prepare(protein, ligand, settings: Optional[SystemSettings] = None,
            membrane: Optional[MembraneSettings] = None,
            funnel_settings: Optional[AutoFunnelSettings] = None,
            funnel_overrides: Optional[dict] = None, funnel: Optional[Funnel] = None,
            platform: Optional[mm.Platform] = None) -> Prepared:
    """From a protein file and a posed ligand to a solvated (or membrane-embedded)
    system with its funnel.

    Pass ``membrane`` to embed the complex in a bilayer with OpenMM's
    ``Modeller.addMembrane``; the protein must already be oriented with the
    membrane normal along z. ``funnel`` skips the automatic placement (its atom
    indices must refer to the dry complex: protein atoms first, then the ligand).
    """
    settings = settings or SystemSettings()
    if membrane is not None and settings.temperature < GEL_TRANSITION_K.get(membrane.lipid.upper(), 0) + 10:
        log.warning("%s is close to or below its gel transition at %.0f K; raise the temperature",
                    membrane.lipid, settings.temperature)
    forcefield = app.ForceField(*settings.forcefields, *settings.ligand_xml)
    register_ligand(forcefield, ligand, settings.ligand_method, settings.ligand_forcefield)
    modeller, ligand_atoms = build_complex(protein, ligand, forcefield, settings)
    dry_topology = modeller.topology
    dry_positions = np.array(modeller.positions.value_in_unit(unit.nanometer))
    dry_selection = classify(dry_topology, ligand_atoms)

    slab = None if membrane is None else MembraneSlab(membrane.center_z, membrane.half_thickness)
    if funnel is None:
        funnel, report = auto_funnel(dry_topology, dry_positions, membrane=slab,
                                     settings=funnel_settings, selection=dry_selection,
                                     **(funnel_overrides or {}))
    else:
        report = validate_funnel(funnel, dry_positions, dry_selection, slab=slab,
                                 cutoff=settings.cutoff)
    report.raise_for_errors()

    ions = dict(ionicStrength=settings.ionic_strength * unit.molar, neutralize=True)
    if membrane is None:
        distance = soluble_image_distance(funnel, dry_positions, dry_selection.solute,
                                          settings.cutoff, settings.padding)
        modeller.addSolvent(forcefield, model=settings.water_model,
                            boxVectors=box_vectors(distance, settings.box_shape), **ions)
    else:
        box_z = membrane_box_z(funnel, dry_positions, dry_selection.solute, settings.cutoff,
                               settings.padding)
        # addMembrane sets the box height to (existing box z) + 2 * minimumPadding, so
        # give it a box whose z makes the total come out at box_z; x and y are ignored.
        preset = max(box_z - 2.0 * settings.padding, 0.1)
        modeller.topology.setPeriodicBoxVectors(
            [mm.Vec3(preset, 0, 0), mm.Vec3(0, preset, 0), mm.Vec3(0, 0, preset)] * unit.nanometer)
        lipid = app.PDBFile(membrane.patch) if membrane.patch else membrane.lipid
        modeller.addMembrane(forcefield, lipidType=lipid,
                             membraneCenterZ=membrane.center_z * unit.nanometer,
                             minimumPadding=settings.padding * unit.nanometer,
                             platform=platform, **ions)

    topology = modeller.topology
    positions = np.array(modeller.positions.value_in_unit(unit.nanometer))
    funnel = remap_funnel(funnel, dry_positions, positions)
    selection = classify(topology, remap_atoms(dry_positions, positions, ligand_atoms))
    system = create_system(forcefield, topology, settings)
    restraint = add_funnel_restraint(system, funnel, periodic=True)

    box = np.array(topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    report = _built_report(report, validate_funnel(funnel, positions, selection, box=box,
                                                   cutoff=settings.cutoff,
                                                   tumbling=membrane is None))
    report.raise_for_errors()
    return Prepared(topology, system, positions, funnel, selection, report,
                    restraint.getForceGroup(), membrane is not None, settings)


def prepare_from_system(topology: app.Topology, system: mm.System, positions,
                        ligand=None, settings: Optional[SystemSettings] = None,
                        funnel_settings: Optional[AutoFunnelSettings] = None,
                        funnel_overrides: Optional[dict] = None,
                        funnel: Optional[Funnel] = None) -> Prepared:
    """Attach a funnel to a system that is already parameterised and solvated
    (AMBER, GROMACS or CHARMM-GUI input). Lipids in the topology make it a membrane system."""
    settings = settings or SystemSettings()
    if unit.is_quantity(positions):
        positions = positions.value_in_unit(unit.nanometer)
    positions = np.asarray(positions, dtype=float)
    selection = classify(topology, ligand)
    is_membrane = selection.lipid.size > 0
    placement = FunnelReport()
    if funnel is None:
        funnel, placement = auto_funnel(topology, positions, settings=funnel_settings,
                                        selection=selection, **(funnel_overrides or {}))
    vectors = topology.getPeriodicBoxVectors()
    if vectors is None:
        raise ValueError("the input system has no periodic box")
    box = np.array(vectors.value_in_unit(unit.nanometer))
    report = _built_report(placement, validate_funnel(funnel, positions, selection, box=box,
                                                      cutoff=settings.cutoff,
                                                      tumbling=not is_membrane))
    report.raise_for_errors()
    for index in reversed(range(system.getNumForces())):   # the workflow adds its own barostat
        if isinstance(system.getForce(index), (mm.MonteCarloBarostat, mm.MonteCarloMembraneBarostat,
                                               mm.MonteCarloAnisotropicBarostat)):
            system.removeForce(index)
    restraint = add_funnel_restraint(system, funnel, periodic=True)
    return Prepared(topology, system, positions, funnel, selection, report,
                    restraint.getForceGroup(), is_membrane, settings)
