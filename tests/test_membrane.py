"""Membrane systems: placement next to a bilayer, and the real addMembrane build."""

import numpy as np
import openmm as mm
import pytest

from fmeta.config import Config
from fmeta.funnel.geometry import Funnel, funnel_frame
from fmeta.funnel.validate import validate_funnel
from fmeta.system.builder import (GEL_TRANSITION_K, MembraneSettings, SystemSettings, add_barostat,
                                  membrane_box_z, prepare)
from fmeta.system.selection import classify
from fmeta.workflow import analyze_stage, equilibrate_stage, prepare_stage, run_replicas

from fixtures import BENZENE_XML, write_membrane_inputs
from synthetic import membrane_barrel

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
    # a longer funnel needs a taller box, by about twice the extra length (minimum-image rule)
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
