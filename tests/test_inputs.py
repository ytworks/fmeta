"""Input routes: complexes, protein repair, wrapped coordinates, ligand parameters."""

import sys
import types

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest

from fmeta.config import Config
from fmeta.funnel.geometry import compute_cvs
from fmeta.system.builder import SystemSettings, prepare
from fmeta.system.imaging import make_whole
from fmeta.system.inputs import (ensure_ligand_hydrogens, pdbfixer_available, read_ligand,
                                 read_protein, read_structure, split_complex)
from fmeta.system.ligand import register_ligand
from fmeta.workflow import prepare_stage

from conftest import LIGAND_SETTINGS
from fixtures import BENZENE_XML


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


def raw_entry(inputs, path, hydrogens=True):
    """The protein as a raw PDB entry would have it: with waters, an ion and a
    second copy of the ligand as hetero atoms."""
    import os
    source = app.PDBFile(os.path.join(os.path.dirname(app.__file__), "data", "test.pdb"))
    modeller = app.Modeller(source.topology, source.positions)
    waters = [r for r in modeller.topology.residues() if r.name == "HOH"]
    modeller.delete(waters[40:])                                   # keep 40 waters and the Cl ions
    ligand_topology, ligand_positions = read_ligand(inputs[1], "BNZ")
    modeller.add(ligand_topology, (ligand_positions + [3.0, 0, 0]) * unit.nanometer)
    if not hydrogens:
        modeller.delete([a for a in modeller.topology.atoms() if a.element.symbol == "H"])
    modeller.topology.setPeriodicBoxVectors(None)
    with open(path, "w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle)


def test_protein_file_is_reduced_to_the_protein(inputs, tmp_path):
    raw_entry(inputs, tmp_path / "raw.pdb")
    topology, positions, notes = read_protein(tmp_path / "raw.pdb", fix=False)
    assert topology.getNumResidues() == 35 and topology.getNumAtoms() == 582
    assert len(notes) == 1 and "HOH x40" in notes[0] and "BNZ x1" in notes[0] and "CL x2" in notes[0]

    kept, _, notes = read_protein(tmp_path / "raw.pdb", fix=False, keep_water=True, keep_hetero=["BNZ"])
    names = [r.name for r in kept.residues()]
    assert names.count("HOH") == 40 and names.count("BNZ") == 1 and "Cl" not in names

    empty = tmp_path / "empty.pdb"
    empty.write_text("HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\nEND\n")
    with pytest.raises(ValueError, match="no amino-acid residues"):
        read_protein(empty, fix=False)


@pytest.mark.parametrize("hydrogens", [True, False])
def test_raw_pdb_and_sdf_are_enough(inputs, tmp_path, hydrogens):
    """A raw entry (waters, ions, another ligand; with or without hydrogens) and the
    ligand SDF build a system without any further preparation by the user."""
    raw_entry(inputs, tmp_path / "raw.pdb", hydrogens=hydrogens)
    built = prepare(tmp_path / "raw.pdb", inputs[1], SystemSettings(**LIGAND_SETTINGS))
    assert built.report.ok, str(built.report)
    assert len(built.selection.protein) == 582 and len(built.selection.ligand) == 12
    assert len(built.selection.other) == 0                    # the stray ligand copy is gone
    checks = {c.name: (c.status, c.message) for c in built.report.checks}
    assert "removed from the protein file" in checks["protein_input"][1]
    assert ("hydrogens added" in checks["protein_input"][1]) == (not hydrogens)
    assert checks["pose_contact"][0] == "ok"


def test_protein_hydrogens_with_foreign_names_are_rebuilt(inputs, tmp_path):
    text = open(inputs[0]).read().replace(" HB2 LEU", " XB2 LEU").replace(" HB3 LEU", " XB3 LEU")
    assert " XB2 LEU" in text
    (tmp_path / "odd.pdb").write_text(text)
    built = prepare(tmp_path / "odd.pdb", inputs[1], SystemSettings(fix_protein=False, **LIGAND_SETTINGS))
    assert len(built.selection.protein) == 582
    message = {c.name: c.message for c in built.report.checks}["protein_input"]
    assert "did not match the force field and were rebuilt" in message


def test_pdbfixer_repairs_missing_atoms_automatically(inputs, tmp_path):
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

    assert pdbfixer_available()
    fixed_topology, _, notes = read_protein(tmp_path / "broken.pdb")     # fix defaults to "if available"
    assert sum(a.element.symbol != "H" for a in fixed_topology.atoms()) == n_heavy
    assert any("24 missing heavy atoms rebuilt" in note for note in notes)
    # ... and the repaired protein goes through the normal build (hydrogens re-added)
    rebuilt = prepare(tmp_path / "broken.pdb", inputs[1], SystemSettings(**LIGAND_SETTINGS))
    assert len(rebuilt.selection.protein) == 582 and rebuilt.report.ok
    # without the repair the same file is refused with advice, not a traceback from OpenMM
    with pytest.raises(ValueError, match="does not fit the force field"):
        prepare(tmp_path / "broken.pdb", inputs[1], SystemSettings(fix_protein=False, **LIGAND_SETTINGS))


def test_ligand_sdf_without_hydrogens_gets_them(inputs, tmp_path):
    pytest.importorskip("rdkit")
    lines = open(inputs[1]).read().splitlines()
    bare = tmp_path / "bare.sdf"
    bare.write_text("\n".join(lines[:3] + ["  6  6  0  0  0  0  0  0  0  0999 V2000"] + lines[4:10]
                              + lines[16:22] + ["M  END", "$$$$"]) + "\n")
    path, notes = ensure_ligand_hydrogens(bare, tmp_path / "scratch")
    assert path.endswith("bare_with_hydrogens.sdf") and "6 added" in notes[0] and "+0" in notes[0]
    topology, positions = read_ligand(path, "BNZ")
    reference = read_ligand(inputs[1], "BNZ")[1]
    assert [a.element.symbol for a in topology.atoms()] == ["C"] * 6 + ["H"] * 6
    assert np.abs(positions[:6] - reference[:6]).max() < 1e-4            # the pose is untouched
    assert ensure_ligand_hydrogens(inputs[1], tmp_path / "scratch") == (inputs[1], [])

    built = prepare(inputs[0], bare, SystemSettings(**LIGAND_SETTINGS), scratch_dir=tmp_path / "s2")
    assert len(built.selection.ligand) == 12 and built.report.ok
    assert {c.name: c.status for c in built.report.checks}["ligand_input"] == "warning"


def test_ligand_away_from_the_protein_is_refused(inputs, tmp_path):
    lines = open(inputs[1]).read().splitlines()
    moved = []
    for line in lines[4:16]:
        x = float(line[0:10]) + 30.0
        moved.append(f"{x:10.4f}" + line[10:])
    (tmp_path / "far.sdf").write_text("\n".join(lines[:4] + moved + lines[16:]) + "\n")
    with pytest.raises(ValueError, match="not in contact with the protein"):
        prepare(inputs[0], tmp_path / "far.sdf", SystemSettings(**LIGAND_SETTINGS))


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
