"""Input routes: complexes, protein repair, wrapped coordinates, other programs' formats."""

import sys
import types
import warnings

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest

from fmeta.config import Config
from fmeta.funnel.geometry import compute_cvs
from fmeta.system.builder import SystemSettings, prepare
from fmeta.system.imaging import make_whole
from fmeta.system.inputs import fix_protein, read_ligand, read_structure, split_complex
from fmeta.system.ligand import register_ligand
from fmeta.workflow import prepare_stage

from conftest import LIGAND_SETTINGS
from fixtures import BENZENE_XML, write_preparameterised


def crystal_like_complex(inputs, path):
    """Protein plus the benzene heavy atoms only, as a PDB entry would have it."""
    topology, positions = read_structure(inputs[0])
    ligand_topology, ligand_positions = read_ligand(inputs[1], "BNZ")
    modeller = app.Modeller(topology, positions * unit.nanometer)
    modeller.add(ligand_topology, ligand_positions * unit.nanometer)
    modeller.delete([a for a in modeller.topology.atoms()
                     if a.residue.name == "BNZ" and a.element.symbol == "H"])
    with open(path, "w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle)
    return ligand_positions


def test_split_complex_adds_hydrogens_from_smiles(inputs, tmp_path):
    pytest.importorskip("rdkit")
    reference = crystal_like_complex(inputs, tmp_path / "complex.pdb")
    protein, ligand = split_complex(tmp_path / "complex.pdb", "BNZ", tmp_path / "out",
                                    smiles="c1ccccc1")
    topology, positions = read_ligand(ligand, "BNZ")
    assert [a.element.symbol for a in topology.atoms()] == ["C"] * 6 + ["H"] * 6
    assert topology.getNumBonds() == 12
    assert np.abs(positions[:6] - reference[:6]).max() < 1e-3        # the pose is untouched
    assert np.linalg.norm(positions[6:] - positions[:6], axis=1) == pytest.approx(0.109, abs=0.005)
    assert read_structure(protein)[0].getNumResidues() == 35

    for smiles, message in ((None, "no hydrogens"), ("c1ccncc1", "does not match"),
                            ("CCCCCC", "does not match"), ("Cc1ccccc1", "7 heavy atoms")):
        with pytest.raises(ValueError, match=message):
            split_complex(tmp_path / "complex.pdb", "BNZ", tmp_path / "bad", smiles=smiles)
    with pytest.raises(ValueError, match="exactly one"):
        split_complex(tmp_path / "complex.pdb", "XXX", tmp_path / "bad", smiles="C")


def test_complex_input_through_prepare_stage(inputs, tmp_path):
    """A PDB-style complex and a SMILES are enough to get a solvated system with a funnel."""
    pytest.importorskip("rdkit")
    crystal_like_complex(inputs, tmp_path / "complex.pdb")
    config = Config.from_dict({
        "input": {"complex": "complex.pdb", "ligand_resname": "BNZ", "ligand_smiles": "c1ccccc1"},
        "system": {"ligand_method": "xml", "ligand_xml": [BENZENE_XML]},
        "run": {"platform": "CPU"}}, base_dir=tmp_path)
    prepared = prepare_stage(config, tmp_path / "work")
    assert prepared.report.ok and len(prepared.selection.ligand) == 12
    assert (tmp_path / "work" / "inputs" / "ligand.sdf").exists()
    saved = Config.load(tmp_path / "work" / "config.yaml")
    assert saved.input.ligand.endswith("ligand.sdf") and saved.system.ligand_resname == "BNZ"


def test_fix_protein_rebuilds_missing_atoms(inputs, tmp_path):
    pytest.importorskip("pdbfixer")
    topology, positions = read_structure(inputs[0])
    n_heavy = sum(a.element.symbol != "H" for a in topology.atoms())
    modeller = app.Modeller(topology, positions * unit.nanometer)
    ring = ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")
    modeller.delete([a for a in modeller.topology.atoms()
                     if a.element.symbol == "H" or (a.residue.name == "PHE" and a.name in ring)])
    with open(tmp_path / "broken.pdb", "w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle)
    assert modeller.topology.getNumAtoms() == n_heavy - 24            # four phenylalanines

    fixed_topology, fixed_positions = fix_protein(tmp_path / "broken.pdb")
    assert sum(a.element.symbol != "H" for a in fixed_topology.atoms()) == n_heavy
    # ... and the repaired protein goes through the normal build (hydrogens re-added)
    settings = SystemSettings(fix_protein=True, **LIGAND_SETTINGS)
    rebuilt = prepare(tmp_path / "broken.pdb", inputs[1], settings)
    assert len(rebuilt.selection.protein) == 582 and rebuilt.report.ok


def test_make_whole_undoes_wrapping(prepared):
    positions, box = prepared.positions, prepared.box_vectors
    solute = np.concatenate([prepared.selection.protein, prepared.selection.ligand])
    # wrap every atom separately into the unit cell and push the ligand one image away
    fractional = np.linalg.solve(box.T, positions.T).T
    wrapped = (fractional - np.floor(fractional)) @ box
    wrapped[prepared.selection.ligand] += box[0] - box[2]
    bonds = np.array([(a.index, b.index) for a, b in prepared.topology.bonds()])
    lengths = np.linalg.norm(wrapped[bonds[:, 0]] - wrapped[bonds[:, 1]], axis=1)
    assert lengths.max() > 2.0                                        # molecules really are broken

    whole = make_whole(prepared.topology, wrapped, box, assemble=solute)
    lengths = np.linalg.norm(whole[bonds[:, 0]] - whole[bonds[:, 1]], axis=1)
    assert lengths.max() < 0.25
    assert compute_cvs(prepared.funnel, whole) == pytest.approx(
        compute_cvs(prepared.funnel, positions), abs=1e-6)
    # each atom moved by a lattice vector only
    shift = np.linalg.solve(box.T, (whole - positions).T).T
    assert np.abs(shift - np.round(shift)).max() < 1e-6


@pytest.mark.parametrize("kind", ["amber", "gromacs", "charmm"])
def test_preparameterised_input_through_prepare_stage(prepared, tmp_path, kind):
    """The same system written as AMBER, GROMACS and CHARMM files (via ParmEd), with
    the coordinates wrapped into the box as simulation output often is."""
    pytest.importorskip("parmed")
    box = prepared.box_vectors
    fractional = np.linalg.solve(box.T, prepared.positions.T).T
    wrapped = (fractional - np.floor(fractional)) @ box
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        block = write_preparameterised(prepared, tmp_path, kind, positions=wrapped)
        config = Config.from_dict({"input": block, "run": {"platform": "CPU"}}, base_dir=tmp_path)
        loaded = prepare_stage(config, tmp_path / "work")

    assert loaded.report.ok, str(loaded.report)
    assert loaded.topology.getNumAtoms() == prepared.topology.getNumAtoms()
    atoms = list(loaded.topology.atoms())
    assert {atoms[i].residue.name for i in loaded.selection.ligand} == {"BNZ"}   # found by itself
    assert len(loaded.selection.ligand) == 12 and len(loaded.selection.protein) == 582
    checks = {c.name: c.status for c in loaded.report.checks}
    assert checks["box_cv"] == "ok" and checks["pose"] == "ok" and checks["p0_anchor"] == "ok"
    direction = np.array(loaded.funnel.metadata["exit_direction"])
    assert direction @ np.array(prepared.funnel.metadata["exit_direction"]) > np.cos(np.radians(15))
    # hydrogen mass repartitioning is applied to these inputs as well
    hydrogen = next(a.index for a in atoms if a.element.symbol == "H" and a.residue.name == "LEU")
    assert loaded.system.getParticleMass(hydrogen).value_in_unit(unit.amu) == pytest.approx(1.5)
    assert (tmp_path / "work" / "prepared" / "funnel.pdb").exists()


def test_template_generator_route(inputs, monkeypatch):
    """The OpenFF/GAFF route with the two external packages replaced by stand-ins:
    checks what we ask of them and that the generator we register is what
    parameterises the ligand during solvation and system creation."""
    calls = {}

    class Molecule:
        @classmethod
        def from_file(cls, path, allow_undefined_stereo=False):
            calls["file"] = (path, allow_undefined_stereo)
            return cls()

        def assign_partial_charges(self, method):
            calls["charges"] = method

    class Generator:
        def __init__(self, molecules=None, forcefield=None):
            calls["generator"] = (type(self).__name__, molecules, forcefield)
            calls["residues"] = []

        def generator(self, forcefield, residue):
            calls["residues"].append(residue.name)
            if residue.name != "BNZ":
                return False
            forcefield.loadFile(BENZENE_XML)
            return True

    toolkit = types.ModuleType("openff.toolkit")
    toolkit.Molecule = Molecule
    generators = types.ModuleType("openmmforcefields.generators")
    generators.SMIRNOFFTemplateGenerator = type("SMIRNOFFTemplateGenerator", (Generator,), {})
    generators.GAFFTemplateGenerator = type("GAFFTemplateGenerator", (Generator,), {})
    for name, module in (("openff", types.ModuleType("openff")), ("openff.toolkit", toolkit),
                         ("openmmforcefields", types.ModuleType("openmmforcefields")),
                         ("openmmforcefields.generators", generators)):
        monkeypatch.setitem(sys.modules, name, module)

    settings = SystemSettings(ligand_method="openff", ligand_forcefield="openff-2.2.0",
                              ligand_charge_method="gasteiger", ligand_resname="BNZ")
    built = prepare(*inputs, settings)
    assert calls["file"] == (inputs[1], True)
    assert calls["charges"] == "gasteiger"
    assert calls["generator"][0] == "SMIRNOFFTemplateGenerator"
    assert isinstance(calls["generator"][1], Molecule) and calls["generator"][2] == "openff-2.2.0"
    assert "BNZ" in calls["residues"]
    assert len(built.selection.ligand) == 12 and built.report.ok

    forcefield = app.ForceField("amber14-all.xml")
    register_ligand(forcefield, inputs[1], "gaff")
    assert calls["generator"][0] == "GAFFTemplateGenerator" and calls["generator"][2] is None
    with pytest.raises(ValueError, match="unknown ligand parameter method"):
        register_ligand(forcefield, inputs[1], "magic")


def test_missing_ligand_packages_give_a_clear_error(inputs, monkeypatch):
    monkeypatch.setitem(sys.modules, "openmmforcefields", None)
    with pytest.raises(ImportError, match="openmmforcefields"):
        register_ligand(app.ForceField("amber14-all.xml"), inputs[1], "openff")
