"""Membrane systems: placement next to a bilayer, and the real addMembrane build."""

import numpy as np
import openmm as mm
import pytest
from fixtures import BENZENE_XML, write_membrane_inputs
from synthetic import membrane_barrel

from fmeta.config import Config
from fmeta.funnel.geometry import Funnel, funnel_frame
from fmeta.funnel.validate import validate_funnel
from fmeta.system.builder import (
    GEL_TRANSITION_K,
    MembraneSettings,
    SystemSettings,
    add_barostat,
    membrane_box_z,
    prepare,
)
from fmeta.system.selection import classify
from fmeta.workflow import analyze_stage, equilibrate_stage, prepare_stage, run_replicas

SETTINGS = dict(ligand_method="xml", ligand_xml=[BENZENE_XML], ligand_resname="BNZ")


def test_membrane_box_height_follows_the_funnel():
    topology, positions, site = membrane_barrel(seed=1)
    selection = classify(topology)
    ca = [a.index for a in topology.atoms() if a.name == "CA"]
    below = [i for i in ca if np.linalg.norm(positions[i] - (site - [0, 0, 1.2])) < 0.8]
    around = [i for i in ca if np.linalg.norm(positions[i] - site) < 0.8]
    def funnel(upper):
        return Funnel(ligand_atoms=list(selection.ligand), p0_atoms=below, p1_atoms=around,
                      lower=0.6, upper=upper, inflection=3.0)
    short, tall = funnel(5.0), funnel(6.5)
    z_short = membrane_box_z(short, positions, selection.solute, cutoff=1.0, padding=1.2)
    z_tall = membrane_box_z(tall, positions, selection.solute, cutoff=1.0, padding=1.2)
    # a longer funnel needs a taller box, by about twice the extra length
    assert z_tall - z_short == pytest.approx(2 * 1.5 * 1.03, abs=0.2)
    _, axis, _ = funnel_frame(tall, positions)
    assert z_tall >= 2 * (abs(axis[2]) * (tall.upper + 0.3) + tall.max_radius + 0.3)

    # per-axis box rule: tall enough in z passes even though x and y are short ...
    box = np.array([6.0, 6.0, z_tall])
    ok = validate_funnel(tall, positions, selection, box=box, tumbling=False)
    assert {c.name: c.status for c in ok.checks}["box_cv"] == "ok"
    # ... while the same box fails the rule for a freely tumbling solute, and a short z fails both
    assert {c.name: c.status for c in validate_funnel(tall, positions, selection, box=box).checks
            }["box_cv"] == "error"
    low = validate_funnel(tall, positions, selection, box=np.array([6.0, 6.0, 9.0]), tumbling=False)
    assert {c.name: c.status for c in low.checks}["box_cv"] == "error"


def test_membrane_barostat_and_lipid_table():
    system = mm.System()
    system.addParticle(1.0)
    add_barostat(system, 310.0, 1.0, membrane=True)
    barostat = system.getForce(0)
    assert isinstance(barostat, mm.MonteCarloMembraneBarostat)
    assert barostat.getXYMode() == mm.MonteCarloMembraneBarostat.XYIsotropic
    assert barostat.getZMode() == mm.MonteCarloMembraneBarostat.ZFree
    assert barostat.getDefaultSurfaceTension()._value == 0.0
    plain = mm.System()
    plain.addParticle(1.0)
    add_barostat(plain, 300.0, 1.0, membrane=False)
    assert type(plain.getForce(0)) is mm.MonteCarloBarostat
    assert GEL_TRANSITION_K["DPPC"] > 300


@pytest.fixture(scope="module")
def membrane_inputs(tmp_path_factory):
    return write_membrane_inputs(tmp_path_factory.mktemp("membrane_inputs"))


@pytest.mark.slow
def test_addmembrane_build(membrane_inputs):
    """The real thing: OpenMM packs a POPC bilayer around the complex."""
    p = prepare(*membrane_inputs, SystemSettings(**SETTINGS), membrane=MembraneSettings("POPC"),
                platform=mm.Platform.getPlatformByName("CPU"))
    assert p.membrane and p.report.ok, str(p.report)
    names = {r.name for r in p.topology.residues()}
    assert "POP" in names and "HOH" in names
    assert len(p.selection.lipid) > 10000 and len(p.selection.ligand) == 12

    checks = {c.name: c.status for c in p.report.checks}
    assert checks["membrane"] == "ok" and checks["box_cv"] == "ok" and checks["box_images"] != "error"
    box = np.diag(p.box_vectors)
    # box height is what the funnel asked for (addMembrane adds 2 * padding to the preset)
    dry_solute = np.concatenate([p.selection.protein, p.selection.ligand])
    assert box[2] == pytest.approx(membrane_box_z(p.funnel, p.positions, dry_solute, 1.0, 1.2), abs=0.05)

    # funnel points up, out of the bilayer, and its atoms are still the right ones
    origin, axis, ligand = funnel_frame(p.funnel, p.positions)
    assert axis[2] > 0.9
    atoms = list(p.topology.atoms())
    assert all(atoms[i].residue.name == "BNZ" for i in p.funnel.ligand_atoms)
    assert all(atoms[i].name == "CA" for i in p.funnel.p0_atoms + p.funnel.p1_atoms)
    lipid_z = p.positions[p.selection.heavy_atoms("lipid"), 2]
    surface = np.percentile(np.abs(lipid_z - lipid_z.mean()), 98) + lipid_z.mean()
    assert origin[2] + (p.funnel.upper - 0.6) * axis[2] > surface + 1.0


@pytest.mark.slow
def test_membrane_pipeline(membrane_inputs, tmp_path):
    """Membrane system through equilibration (semi-isotropic barostat) and a short run."""
    config = Config.from_dict({
        "input": {"protein": membrane_inputs[0], "ligand": membrane_inputs[1]},
        "system": SETTINGS,
        "membrane": {"lipid": "POPC"},
        "equilibration": {"minimize_iterations": 500, "heating_ps": 0.2, "release_ps": 0.2,
                          "free_ps": 0.4, "seed": 3},
        "run": {"length_ns": 0.001, "replicas": 1, "pace_ps": 0.1, "checkpoint_ps": 0.5,
                "trajectory_ps": 0.5, "seed": 7, "platform": "CPU"},
    })
    work = tmp_path / "work"
    prepared = prepare_stage(config, work)
    assert prepared.membrane
    summary = equilibrate_stage(config, work)
    assert summary["funnel_checks"]["ok"]
    box = summary["box_lengths"]
    assert box[0] / box[1] == pytest.approx(prepared.box_vectors[0, 0] / prepared.box_vectors[1, 1],
                                            rel=1e-6)          # x and y scale together
    run_replicas(config, work)
    assert len((work / "rep0" / "HILLS").read_text().splitlines()) == 3 + 10
    assert analyze_stage(work)["n_replicas"] == 1


# ---- membrane orientation ----------------------------------------------------

def test_slab_fit_finds_the_membrane_plane_of_a_helix_bundle():
    from synthetic import helix_bundle, random_rotation

    from fmeta.system.orientation import fit_slab, orient
    for seed in range(3):
        rng = np.random.default_rng(100 + seed)
        xyz, names = helix_bundle(seed)
        rotation, shift = random_rotation(rng), rng.uniform(-3, 3, 3)
        moved = xyz @ rotation.T + shift
        fit = fit_slab(moved, names)
        true_normal = rotation @ np.array([0.0, 0.0, 1.0])
        assert fit.confident and fit.score > 0.6
        assert np.degrees(np.arccos(abs(fit.normal @ true_normal))) < 12.0
        assert abs((fit.midpoint - shift) @ true_normal) < 0.3

        new, rot, translation = orient(moved, fit, center_z=0.5)
        assert np.allclose(new, moved @ rot.T + translation)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9) and np.linalg.det(rot) > 0
        again = fit_slab(new, names)
        assert again.tilt() < 12.0 and abs(again.offset(0.5)) < 0.3


def test_slab_fit_is_not_fooled_by_soluble_proteins():
    import os

    import openmm.app as app
    import openmm.unit as unit

    from fmeta.system.orientation import HYDROPHOBICITY, fit_membrane, fit_slab
    pdb = app.PDBFile(os.path.join(os.path.dirname(app.__file__), "data", "test.pdb"))
    positions = np.array(pdb.positions.value_in_unit(unit.nanometer))
    assert not fit_membrane(pdb.topology, positions).confident          # a real soluble protein
    for seed in range(3):                                               # globules, apolar core
        rng = np.random.default_rng(50 + seed)
        points = rng.uniform(-2.2, 2.2, size=(4000, 3))
        points = points[np.linalg.norm(points, axis=1) < 2.2][:320]
        names = [rng.choice(("LEU", "ILE", "VAL", "PHE", "ALA"))
                 if np.linalg.norm(p) < 1.5 and rng.random() < 0.8
                 else rng.choice(("LYS", "GLU", "SER", "ASP", "ASN", "THR", "GLN", "ARG"))
                 for p in points]
        assert not fit_slab(points, names).confident
    with pytest.raises(ValueError, match="too few"):
        fit_slab(points[:10], names[:10])
    assert set(HYDROPHOBICITY) >= {"LEU", "ARG", "GLY"}


def _bundle_topology(positions, names):
    import openmm.app as app
    topology = app.Topology()
    chain = topology.addChain("A")
    for name in names:
        topology.addAtom("CA", app.Element.getBySymbol("C"), topology.addResidue(str(name), chain))
    return topology


def test_orientation_check_and_auto_modes():
    from synthetic import helix_bundle

    from fmeta.system.builder import _membrane_orientation
    from fmeta.system.orientation import fit_slab
    xyz, names = helix_bundle(1)
    topology = _bundle_topology(xyz, names)

    # oriented input: accepted quietly; "input" skips the fit altogether
    same, (status, message, info) = _membrane_orientation(topology, xyz, MembraneSettings())
    assert status == "ok" and "agrees" in message and same is xyz
    assert _membrane_orientation(topology, xyz, MembraneSettings(orient="input")) == (xyz, None)

    # the same bundle lying on its side and 1.5 nm too high
    tilted = xyz[:, [2, 1, 0]] * np.array([-1.0, 1.0, 1.0]) + np.array([0.0, 0.0, 1.5])
    _, (status, message, info) = _membrane_orientation(topology, tilted, MembraneSettings())
    assert status == "warning" and "does not look oriented" in message and info["tilt_deg"] > 70

    fixed, (status, message, info) = _membrane_orientation(
        topology, tilted, MembraneSettings(orient="auto", center_z=0.0))
    assert status == "ok" and "re-oriented" in message
    refit = fit_slab(fixed, names)
    assert refit.tilt() < 12.0 and abs(refit.offset()) < 0.3
    assert np.allclose(fixed, tilted @ np.array(info["rotation"]).T + np.array(info["translation"]))

    # a protein without a belt cannot be auto-oriented, and its check is a warning: it checked nothing
    rng = np.random.default_rng(3)
    blob = rng.uniform(-2, 2, size=(300, 3))
    blob_names = list(rng.choice(["LEU", "LYS", "SER", "GLU", "ALA", "ARG"], size=300))
    with pytest.raises(ValueError, match="hydrophobic belt"):
        _membrane_orientation(_bundle_topology(blob, blob_names), blob, MembraneSettings(orient="auto"))
    _, (status, message, _) = _membrane_orientation(_bundle_topology(blob, blob_names), blob,
                                                    MembraneSettings())
    assert status == "warning" and "could not be checked" in message
    with pytest.raises(ValueError, match="membrane.orient"):
        _membrane_orientation(topology, xyz, MembraneSettings(orient="sideways"))


def test_lipids_inside_the_funnel_are_reported():
    import openmm.app as app
    topology, positions, site = membrane_barrel(seed=1)
    selection = classify(topology)
    ca = [a.index for a in topology.atoms() if a.name == "CA"]
    funnel = Funnel(ligand_atoms=list(selection.ligand),
                    p0_atoms=[i for i in ca if np.linalg.norm(positions[i] - (site - [0, 0, 1.2])) < 0.8],
                    p1_atoms=[i for i in ca if np.linalg.norm(positions[i] - site) < 0.8],
                    lower=0.6, upper=5.5, inflection=3.2)
    origin, axis, _ = funnel_frame(funnel, positions)

    def with_lipid(point):
        """The barrel plus a bilayer-like sheet of lipid atoms and one stray lipid atom."""
        new = app.Topology()
        chain = new.addChain("M")
        carbon = app.Element.getBySymbol("C")
        rng = np.random.default_rng(0)
        sheet = np.column_stack([rng.uniform(-3, 3, 400), rng.uniform(-3, 3, 400), rng.uniform(-2, 2, 400)])
        sheet = sheet[np.hypot(sheet[:, 0], sheet[:, 1]) > 2.0]
        coords = np.vstack([positions, sheet, [point]])
        for residue in topology.residues():
            copy = new.addResidue(residue.name, chain)
            for atom in residue.atoms():
                new.addAtom(atom.name, atom.element, copy)
        for _ in range(len(sheet) + 1):
            new.addAtom("C1", carbon, new.addResidue("POPC", chain))
        return new, coords

    for point, expected in ((origin + 4.0 * axis, "error"),                       # on the axis
                            (origin + 3.0 * axis + np.array([0.35, 0, 0]), "warning"),   # inside the wall
                            (np.array([2.5, 2.5, 0.0]), "ok")):                   # in the bilayer
        new, coords = with_lipid(point)
        report = validate_funnel(funnel, coords, classify(new, list(selection.ligand)),
                                 box=np.array([8.0, 8.0, 14.0]), tumbling=False)
        assert {c.name: c.status for c in report.checks}["lipids_in_funnel"] == expected
