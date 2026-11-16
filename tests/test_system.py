"""Inputs, configuration and system building on a small real protein."""

import json
import shutil

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest
from fixtures import BENZENE_XML
from scipy.spatial import cKDTree

from fmeta.cli import main as cli
from fmeta.config import Config
from fmeta.engine.equilibration import EquilibrationResult, _add_position_restraints, block_means
from fmeta.engine.hills import read_colvar
from fmeta.funnel.geometry import compute_cvs, funnel_frame
from fmeta.funnel.validate import required_image_distance
from fmeta.system.builder import (
    SystemSettings,
    box_vectors,
    prepare,
    prepare_from_system,
    remap_atoms,
)
from fmeta.system.inputs import read_ligand
from fmeta.workflow import (
    analyze_stage,
    equilibrate_stage,
    format_summary,
    load_prepared,
    prepare_stage,
    run_replicas,
    save_prepared,
)

SETTINGS = dict(ligand_method="xml", ligand_xml=[BENZENE_XML], ligand_resname="BNZ")


def test_ligand_readers(inputs, tmp_path):
    topology, positions = read_ligand(inputs[1], "BNZ")
    assert topology.getNumAtoms() == 12 and topology.getNumBonds() == 12
    assert [a.element.symbol for a in topology.atoms()] == ["C"] * 6 + ["H"] * 6
    assert np.linalg.norm(positions[0] - positions[3]) == pytest.approx(0.2794, abs=1e-4)   # para C-C

    mol2 = tmp_path / "water.mol2"
    mol2.write_text("@<TRIPOS>MOLECULE\nHOH\n3 2\nSMALL\n\n@<TRIPOS>ATOM\n"
                    "1 O 0.0 0.0 0.0 O.3 1 HOH 0\n2 H1 0.96 0.0 0.0 H 1 HOH 0\n"
                    "3 H2 -0.24 0.93 0.0 H 1 HOH 0\n@<TRIPOS>BOND\n1 1 2 1\n2 1 3 1\n")
    topology, positions = read_ligand(mol2)
    assert [a.element.symbol for a in topology.atoms()] == ["O", "H", "H"]
    assert topology.getNumBonds() == 2 and positions[1, 0] == pytest.approx(0.096)

    bare = tmp_path / "bare.sdf"
    lines = open(inputs[1]).read().splitlines()
    bare.write_text("\n".join(lines[:3] + ["  6  0  0  0  0  0  0  0  0  0999 V2000"] + lines[4:10]
                              + ["M  END"]) + "\n")
    with pytest.raises(ValueError, match="no hydrogens"):
        read_ligand(bare)


def test_config_mapping(tmp_path):
    (tmp_path / "sub").mkdir()
    path = tmp_path / "sub" / "c.yaml"
    path.write_text("input: {protein: ../p.pdb, ligand: l.sdf}\n"
                    "system: {temperature: 310, ligand_xml: [lig.xml]}\n"
                    "membrane: true\n"
                    "metad: {bias_factor: 15}\n"
                    "run: {timestep_fs: 2.0, pace_ps: 0.5, replicas: 3}\n")
    config = Config.load(path)
    assert config.input.protein == str(tmp_path / "p.pdb")
    assert config.system.ligand_xml == [str(tmp_path / "sub" / "lig.xml")]
    assert config.membrane.lipid == "POPC"
    assert config.metad.temperature == 310 and config.metad.bias_factor == 15
    assert config.metad.pace == 250                      # 0.5 ps at 2 fs
    config.save(tmp_path / "copy.yaml")
    assert Config.load(tmp_path / "copy.yaml").to_dict() == config.to_dict()

    with pytest.raises(ValueError, match="unknown option"):
        Config.from_dict({"run": {"lenght_ns": 1}})
    with pytest.raises(ValueError, match="unknown configuration section"):
        Config.from_dict({"runs": {}})
    with pytest.raises(ValueError, match="hydrogen_mass"):
        Config.from_dict({"system": {"hydrogen_mass": None}})

    cli(["init", str(tmp_path / "template.yaml")])
    template = Config.load(tmp_path / "template.yaml")
    assert template.run.replicas == 5 and template.membrane is None


def test_box_helpers():
    cube = np.array([[v.x, v.y, v.z] for v in (b.value_in_unit(unit.nanometer) for b in box_vectors(5.0, "cube"))])
    assert np.allclose(cube, 5.0 * np.eye(3))
    rows = np.array([[v.x, v.y, v.z] for v in (b.value_in_unit(unit.nanometer)
                                              for b in box_vectors(5.0, "dodecahedron"))])
    # every lattice vector, and every sum/difference of two, is at least 5 nm long
    images = [i * rows[0] + j * rows[1] + k * rows[2]
              for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    assert min(np.linalg.norm(v) for v in images) == pytest.approx(5.0)
    assert abs(np.linalg.det(rows)) == pytest.approx(5.0**3 / np.sqrt(2))   # 71% of the cube

    old = np.random.default_rng(0).uniform(0, 3, size=(20, 3))
    order = np.random.default_rng(1).permutation(20)
    new = np.vstack([np.full((5, 3), 9.0), old[order]])
    mapped = remap_atoms(old, new, [0, 7, 19])
    assert np.allclose(new[mapped], old[[0, 7, 19]])
    with pytest.raises(RuntimeError):
        remap_atoms(old, new + 0.01, [0])


def test_prepare_soluble(prepared, inputs):
    p = prepared
    assert p.report.ok and not p.membrane
    counts = {n: len(getattr(p.selection, n)) for n in ("protein", "ligand", "water", "ion", "lipid")}
    assert counts["protein"] == 582 and counts["ligand"] == 12 and counts["lipid"] == 0
    assert counts["water"] > 20000 and counts["ion"] > 10

    # the box obeys the rules it was sized by
    box = p.box_vectors
    images = [i * box[0] + j * box[1] + k * box[2]
              for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    shortest = min(np.linalg.norm(v) for v in images)
    assert shortest >= required_image_distance(p.funnel)
    checks = {c.name: c.status for c in p.report.checks}
    assert checks["box_cv"] == "ok" and checks["box_images"] == "ok" and checks["pose"] == "ok"

    # funnel atoms still point at the same atoms after solvation
    atoms = list(p.topology.atoms())
    assert all(atoms[i].residue.name == "BNZ" and atoms[i].element.symbol == "C"
               for i in p.funnel.ligand_atoms)
    assert all(atoms[i].name == "CA" for i in p.funnel.p0_atoms + p.funnel.p1_atoms)
    s, rho = compute_cvs(p.funnel, p.positions)
    assert p.funnel.lower + 0.2 < s < p.funnel.inflection and rho < 0.2

    # the system: PME, mass-repartitioned hydrogens, funnel walls in their own group
    forces = {type(f).__name__: f for f in p.system.getForces()}
    assert forces["NonbondedForce"].getNonbondedMethod() == mm.NonbondedForce.PME
    hydrogen = next(a.index for a in atoms if a.element.symbol == "H" and a.residue.name == "LEU")
    assert p.system.getParticleMass(hydrogen).value_in_unit(unit.amu) == pytest.approx(1.5)
    assert p.system.getForce(p.system.getNumForces() - 1).getForceGroup() == p.restraint_group
    context = mm.Context(p.system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("CPU"))
    context.setPositions(p.positions)
    wall = context.getState(getEnergy=True, groups={p.restraint_group}).getPotentialEnergy()
    assert wall.value_in_unit(unit.kilojoule_per_mole) == pytest.approx(0.0, abs=1e-6)


def test_a_kept_ion_stays_part_of_the_receptor(inputs, tmp_path):
    """A sodium ion kept from the input (``system.keep_hetero``) is still told apart from the
    sodium ions of the solvent in the built system and after a reload, and equilibration
    holds it in place with the protein (FM-D15)."""
    source = app.PDBFile(inputs[0])
    protein = np.array(source.positions.value_in_unit(unit.nanometer))
    centre = protein.mean(axis=0)
    away = centre - read_ligand(inputs[1], "BNZ")[1].mean(axis=0)     # the side opposite the ligand
    away /= np.linalg.norm(away)
    tree = cKDTree(protein)
    point = next(centre + step * away for step in np.arange(0.0, 5.0, 0.05)
                 if tree.query(centre + step * away)[0] >= 0.4)
    extra = app.Topology()
    extra.addAtom("NA", app.Element.getBySymbol("Na"), extra.addResidue("NA", extra.addChain("I")))
    modeller = app.Modeller(source.topology, source.positions)
    modeller.add(extra, [mm.Vec3(*point)] * unit.nanometer)
    path = str(tmp_path / "protein_na.pdb")
    with open(path, "w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle)

    p = prepare(path, inputs[1], SystemSettings(**SETTINGS, keep_hetero=["NA"]))
    (ion,) = p.selection.other
    atoms = list(p.topology.atoms())
    assert atoms[ion].residue.name == "NA" and np.allclose(p.positions[ion], point, atol=1e-3)
    bulk = [int(i) for i in p.selection.ion if atoms[i].residue.name == "NA"]
    assert bulk and p.report.ok                              # the same name, from the solvent

    work = tmp_path / "work"
    (work / "prepared").mkdir(parents=True)
    save_prepared(p, work / "prepared")
    config = Config.from_dict({"input": {"protein": inputs[0], "ligand": inputs[1]}, "system": SETTINGS})
    assert list(load_prepared(work, config).selection.other) == [ion]

    # the restraint of the first equilibration stages: on the kept ion, not on a solvent ion
    carrier = mm.System()
    for _ in range(len(p.positions)):
        carrier.addParticle(1.0)
    carrier.setDefaultPeriodicBoxVectors(*[mm.Vec3(*vector) for vector in p.box_vectors])
    _add_position_restraints(carrier, p, 1000.0)
    context = mm.Context(carrier, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))

    def energy(moved):
        positions = p.positions.copy()
        positions[moved] += [0.1, 0.0, 0.0]
        context.setPositions(positions)
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    assert energy(ion) == pytest.approx(0.5 * 1000.0 * 0.1**2)
    assert energy(bulk[0]) == pytest.approx(0.0, abs=1e-9)


def test_prepare_from_presolvated_system(prepared):
    """A system that arrives already solvated: the ligand is found by itself and
    the funnel is placed with water present."""
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml", BENZENE_XML)
    system = forcefield.createSystem(prepared.topology, nonbondedMethod=app.PME,
                                     constraints=app.HBonds)
    system.addForce(mm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))
    again = prepare_from_system(prepared.topology, system, prepared.positions)
    assert sorted(again.selection.ligand) == sorted(prepared.selection.ligand)
    assert not again.membrane and again.report.ok
    assert not any(isinstance(f, mm.MonteCarloBarostat) for f in again.system.getForces())
    direction = np.array(again.funnel.metadata["exit_direction"])
    assert direction @ np.array(prepared.funnel.metadata["exit_direction"]) > np.cos(np.radians(15))

    # in a box too small for the funnel the build is refused
    small_box = 0.6 * prepared.box_vectors
    prepared.topology.setPeriodicBoxVectors(small_box * unit.nanometer)
    try:
        with pytest.raises(ValueError, match="box_cv"):
            prepare_from_system(prepared.topology, system, prepared.positions)
    finally:
        prepared.topology.setPeriodicBoxVectors(prepared.box_vectors / 0.6 * unit.nanometer)


def test_equilibrate_stage_judges_the_ligand_by_the_walls(prepared, inputs, tmp_path, monkeypatch):
    """The stage accepts an equilibrated structure whose ligand leans on the wall (atoms beyond
    it, the centre inside) and reports it. The dynamics are replaced by that structure."""
    funnel = prepared.funnel
    origin, axis, lig = funnel_frame(funnel, prepared.positions)
    s = float((lig - origin) @ axis)
    offsets = prepared.positions[funnel.ligand_atoms] - lig
    offsets -= np.outer(offsets @ axis, axis)
    outward = offsets[np.linalg.norm(offsets, axis=1).argmax()]      # towards a ring atom
    outward /= np.linalg.norm(outward)
    positions = prepared.positions.copy()
    positions[prepared.selection.ligand] += (
        origin + s * axis + (float(funnel.wall_radius(s)) - 0.02) * outward - lig)

    carrier = mm.System()                    # no forces: it only turns the coordinates into a State
    for _ in range(len(positions)):
        carrier.addParticle(1.0)
    context = mm.Context(carrier, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    context.setPeriodicBoxVectors(*[mm.Vec3(*vector) for vector in prepared.box_vectors])
    context.setPositions(positions)
    state = context.getState(getPositions=True, getVelocities=True)
    monkeypatch.setattr("fmeta.workflow.equilibrate", lambda *args: EquilibrationResult(
        state, 0.0, 0.0, np.diag(prepared.box_vectors)))

    work = tmp_path / "work"
    (work / "prepared").mkdir(parents=True)
    save_prepared(prepared, work / "prepared")
    config = Config.from_dict({"input": {"protein": inputs[0], "ligand": inputs[1]},
                               "system": SETTINGS, "run": {"platform": "CPU"}})
    summary = equilibrate_stage(config, work)
    checks = {c["name"]: c["status"] for c in summary["funnel_checks"]["checks"]}
    assert checks["pose"] == "warning" and summary["funnel_checks"]["ok"]


@pytest.mark.slow
def test_full_pipeline_with_restart(inputs, tmp_path):
    """prepare -> equilibrate -> run -> continue -> analyse, through the public stages."""
    config = Config.from_dict({
        "input": {"protein": inputs[0], "ligand": inputs[1]},
        "system": SETTINGS,
        "equilibration": {"minimize_iterations": 200, "heating_ps": 0.2, "release_ps": 0.2,
                          "free_ps": 0.4, "seed": 3},
        "run": {"length_ns": 0.001, "replicas": 1, "pace_ps": 0.1, "checkpoint_ps": 0.5,
                "trajectory_ps": 0.5, "seed": 7, "platform": "CPU"},
    })
    work = tmp_path / "work"
    prepare_stage(config, work)
    for name in ("system.xml", "topology.cif", "funnel.yaml", "funnel.pdb", "report.json"):
        assert (work / "prepared" / name).exists()
    reloaded = load_prepared(work)
    assert reloaded.topology.getNumAtoms() == reloaded.system.getNumParticles()

    summary = equilibrate_stage(config, work)
    assert summary["wall_ok"] and summary["funnel_checks"]["ok"]

    run_replicas(config, work)
    first = (work / "rep0" / "HILLS").read_text()
    assert len(first.splitlines()) == 3 + 10
    run_replicas(config, work, length_ns=0.002)          # continue the same replica
    second = (work / "rep0" / "HILLS").read_text()
    assert second.startswith(first) and len(second.splitlines()) == 3 + 20

    result = analyze_stage(work)
    assert result["n_replicas"] == 1 and not result["converged"]
    assert "NOT CONVERGED" in format_summary(result)
    assert json.loads((work / "summary.json").read_text())["dG"] == result["dG"]

    # the same equilibrated system with two walkers sharing one bias (two subprocesses)
    shared = tmp_path / "shared"
    shared.mkdir()
    for stage in ("prepared", "equil"):
        shutil.copytree(work / stage, shared / stage)
    config.run.walkers = 2
    config.save(shared / "config.yaml")
    run_replicas(config, shared)
    for walker in ("w0", "w1"):
        assert len((shared / "rep0" / walker / "HILLS").read_text().splitlines()) == 3 + 10
    late = read_colvar(shared / "rep0" / "w0" / "COLVAR")["bias"][-1]
    assert late > 0.0                                   # a bias has built up
    merged = analyze_stage(shared)
    assert merged["n_replicas"] == 1
    assert json.loads((shared / "rep0" / "result.json").read_text())["n_walkers"] == 2


def test_the_free_stage_is_reported_in_equal_blocks():
    area = np.arange(20.0)                              # one sample every 2 ps
    blocks = block_means(area, 2.0 * area, sample_ps=2.0, blocks=4)
    assert [block["end_ps"] for block in blocks] == [10.0, 20.0, 30.0, 40.0]
    assert [block["area"] for block in blocks] == [2.0, 7.0, 12.0, 17.0]
    assert blocks[-1]["height"] == 34.0
    assert len(block_means(area[:3], area[:3], sample_ps=1.0)) == 3     # fewer samples than blocks
