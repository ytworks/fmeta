"""Input routes: complexes, protein repair, wrapped coordinates, ligand parameters."""

import json
import sys
import types

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from conftest import LIGAND_SETTINGS
from fixtures import BENZENE_XML

from fmeta.config import Config
from fmeta.funnel.geometry import compute_cvs
from fmeta.system.builder import SystemSettings, prepare
from fmeta.system.imaging import make_whole
from fmeta.system.inputs import (
    complete_hydrogens,
    ensure_ligand_hydrogens,
    pdbfixer_available,
    read_ligand,
    read_protein,
    read_structure,
    split_complex,
)
from fmeta.system.ligand import register_ligand
from fmeta.system.protein import model_states
from fmeta.workflow import prepare_stage


def crystal_like_complex(inputs, path, hydrogens=False):
    """Protein plus the benzene heavy atoms only, as a PDB entry would have it (or with the
    hydrogens of the benzene, as a prepared complex would)."""
    topology, positions = read_structure(inputs[0])
    ligand_topology, ligand_positions = read_ligand(inputs[1], "BNZ")
    modeller = app.Modeller(topology, positions * unit.nanometer)
    modeller.add(ligand_topology, ligand_positions * unit.nanometer)
    modeller.delete([a for a in modeller.topology.atoms()
                     if a.residue.name == "BNZ" and a.element.symbol == "H" and not hydrogens])
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

    for smiles, message in (("c1ccncc1", "does not match"), ("CCCCCC", "does not match"),
                            ("Cc1ccccc1", "7 heavy atoms")):
        with pytest.raises(ValueError, match=message):
            split_complex(tmp_path / "complex.pdb", "BNZ", tmp_path / "bad", smiles)
    with pytest.raises(ValueError, match="exactly one"):
        split_complex(tmp_path / "complex.pdb", "XXX", tmp_path / "bad", "C")
    # hydrogens and CONECT records state no bond orders: without the SMILES this benzene was
    # written with single bonds, which a parameter generator reads as a cyclohexane short of
    # six hydrogens (review of PR #1)
    crystal_like_complex(inputs, tmp_path / "hydrogenated.pdb", hydrogens=True)
    with pytest.raises(ValueError, match="needs the SMILES"):
        split_complex(tmp_path / "hydrogenated.pdb", "BNZ", tmp_path / "none", None)
    assert not (tmp_path / "none").exists()


def test_split_keeps_the_protonation_state_a_residue_name_states(inputs, tmp_path):
    """HIS 27 called HIP in a complex without hydrogens is HIP in the protein of the split,
    as it is when the protein file is read directly (review of PR #1: OpenMM reads the name as
    HIS, and the split wrote that)."""
    pytest.importorskip("rdkit")
    crystal_like_complex(inputs, tmp_path / "full.pdb")
    lines = [line.replace("HIS A  27", "HIP A  27") for line in (tmp_path / "full.pdb").read_text().splitlines()
             if line[76:78].strip() != "H"]
    assert sum("HIP A  27" in line for line in lines) == 10
    (tmp_path / "complex.pdb").write_text("\n".join(lines) + "\n")
    protein, _ = split_complex(tmp_path / "complex.pdb", "BNZ", tmp_path / "out", smiles="c1ccccc1")
    assert read_protein(protein, fix=False).info["protonation_from_input"] == {"HIP": ["A:HIS27"]}


def test_split_drops_chains_and_residue_ranges_before_the_ligand_search(tmp_path, capsys):
    """`fmeta split --drop` on a receptor with a fusion partner numbered 1001-1106 in the same
    chain and a second copy of the complex in chain B."""
    pytest.importorskip("rdkit")
    from fmeta.cli import main as cli
    topology = app.Topology()
    carbon, hydrogen = app.Element.getBySymbol("C"), app.Element.getBySymbol("H")
    labels = {"A": ["11", "208", "1001", "1050", "1050A", "1106", "1106A", "1107", "219"], "B": ["5"]}
    for chain_id, residues in labels.items():
        chain = topology.addChain(chain_id)
        for label in residues:
            residue = topology.addResidue("GLY", chain, label.rstrip("A"), label[4:])
            topology.addAtom("CA", carbon, residue)
        methane = topology.addResidue("LIG", chain, "2401")
        atoms = [topology.addAtom(name, element, methane) for name, element in
                 [("C1", carbon)] + [(f"H{k}", hydrogen) for k in range(1, 5)]]
        for atom in atoms[1:]:
            topology.addBond(atoms[0], atom)
    positions = np.random.default_rng(0).uniform(0.0, 3.0, size=(topology.getNumAtoms(), 3))
    with open(tmp_path / "fusion.pdb", "w") as handle:
        app.PDBFile.writeFile(topology, positions * unit.nanometer, handle, keepIds=True)

    def kept(*drop):
        out = tmp_path / f"out{len(list(tmp_path.iterdir()))}"
        cli(["split", str(tmp_path / "fusion.pdb"), "--resname", "LIG", "--smiles", "C", "-o", str(out)]
            + [word for selection in drop for word in ("--drop", selection)])
        return [f"{r.chain.id}{r.id}{r.insertionCode.strip()}"
                for r in read_structure(out / "protein.pdb")[0].residues()]

    # numbers, not strings ("11" and "208" sort between "1001" and "1106"); insertion codes go
    # with their number; chain B and its ligand copy are gone before the ligand is looked for
    assert kept("A:1001-1106", "B") == ["A11", "A208", "A1107", "A219"]
    assert "dropped A:1001-1106: 5 residues (GLY1001 to GLY1106A)" in capsys.readouterr().out
    assert kept("A:1050A", "B:2401") == ["A11", "A208", "A1001", "A1050", "A1106", "A1106A",
                                         "A1107", "A219", "B5"]
    for drop, message in (([], "exactly one"), (["A:3000-3010", "B"], "matches no residue"),
                          (["A:1106-1001", "B"], "ends before"), (["A:1001..1106", "B"], "cannot read")):
        with pytest.raises(ValueError, match=message):
            split_complex(tmp_path / "fusion.pdb", "LIG", tmp_path / "bad", "C", drop=drop)


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
    protein = read_protein(tmp_path / "raw.pdb", fix=False)
    assert protein.topology.getNumResidues() == 35 and protein.topology.getNumAtoms() == 582
    assert protein.notes[0] == "removed from the protein file: BNZ x1, CL x2, HOH x40"

    kept = read_protein(tmp_path / "raw.pdb", fix=False, keep_water=True, keep_hetero=["BNZ"])
    names = [r.name for r in kept.topology.residues()]
    assert names.count("HOH") == 40 and names.count("BNZ") == 1 and "Cl" not in names
    assert kept.info["kept_hetero"] == {"BNZ": 1, "HOH": 40} and kept.info["removed"] == {"CL": 2}

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
    fixed = read_protein(tmp_path / "broken.pdb")                       # fix defaults to "if available"
    assert sum(a.element.symbol != "H" for a in fixed.topology.atoms()) == n_heavy
    assert any("24 missing heavy atoms rebuilt" in note for note in fixed.notes)
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
    assert calls["generator"][0] == "GAFFTemplateGenerator"
    with pytest.raises(ValueError, match="unknown ligand parameter method"):
        register_ligand(forcefield, inputs[1], "magic")


@pytest.mark.parametrize("method", ["openff", "gaff"])
def test_default_settings_work_with_the_real_generators(inputs, method):
    """A ligand SDF and nothing else, with the real packages (they are in the official image):
    the generators have no default force field of their own, so ours must name one they know."""
    pytest.importorskip("openmmforcefields")
    pytest.importorskip("openff.toolkit")
    forcefield = app.ForceField("amber14-all.xml")
    register_ligand(forcefield, inputs[1], method)
    topology, _ = read_ligand(inputs[1], "BNZ")
    system = forcefield.createSystem(topology)
    nonbonded = next(f for f in system.getForces() if f.getName() == "NonbondedForce")
    charges = [nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
               for i in range(system.getNumParticles())]
    assert len(charges) == 12 and abs(sum(charges)) < 1e-6
    assert max(charges) > 0.05 and min(charges) < -0.05                # benzene C-H dipoles


def _hide_ligand_packages(monkeypatch):
    """As on a machine without them, also when an earlier test has imported the real ones."""
    for name in ("openmmforcefields", "openmmforcefields.generators"):
        monkeypatch.setitem(sys.modules, name, None)


def test_missing_ligand_packages_give_a_clear_error(inputs, monkeypatch):
    _hide_ligand_packages(monkeypatch)
    with pytest.raises(ImportError, match="openmmforcefields"):
        register_ligand(app.ForceField("amber14-all.xml"), inputs[1], "openff")


def test_command_line_reports_input_problems_plainly(inputs, tmp_path, monkeypatch):
    """Missing ligand parameters and a bad pose end with a one-line message, not a traceback."""
    from fmeta.cli import main as cli
    _hide_ligand_packages(monkeypatch)
    with pytest.raises(SystemExit) as stop:
        cli(["prepare", "--protein", inputs[0], "--ligand", inputs[1], "-o", str(tmp_path / "a")])
    assert "fmeta: error:" in str(stop.value) and "openmmforcefields" in str(stop.value)
    with pytest.raises(SystemExit) as stop:
        cli(["prepare", "--protein", str(tmp_path / "missing.pdb"), "--ligand", inputs[1],
             "--ligand-xml", BENZENE_XML, "-o", str(tmp_path / "b")])
    assert "fmeta: error:" in str(stop.value)


# ---- the protein model itself: chain gaps, caps, protonation states, disulfides and
#      alternate locations. Structures are cut from the 35-residue test protein, whose
#      net charge is +2 with HIE 27 (index 26), ASP 3 (2), GLU 4 (3) and LYS 7 (6). ------

@pytest.fixture(scope="module")
def amber():
    return app.ForceField("amber14-all.xml")


def edited(path, source, skip=lambda atom: False, residues=None, atoms=None, extra=()):
    """Write a copy of the structure ``source`` without the atoms ``skip`` selects, with
    residues renamed (index -> name), atoms renamed ((residue index, name) -> name) and
    atoms added (residue index, name, element, position in nm)."""
    topology, positions = read_structure(source)
    new, coordinates = app.Topology(), []
    chain = new.addChain("A")
    for residue in topology.residues():
        target = new.addResidue((residues or {}).get(residue.index, residue.name), chain, residue.id)
        for atom in residue.atoms():
            if not skip(atom):
                new.addAtom((atoms or {}).get((residue.index, atom.name), atom.name), atom.element, target)
                coordinates.append(positions[atom.index])
        for index, name, symbol, position in extra:
            if index == residue.index:
                new.addAtom(name, app.Element.getBySymbol(symbol), target)
                coordinates.append(position)
    with open(path, "w") as handle:
        app.PDBFile.writeFile(new, np.array(coordinates) * unit.nanometer, handle, keepIds=True)
    return path


def modelled(path, forcefield, fix=False):
    """The protein model ``build_complex`` makes of a file: reduced, capped, repaired, with
    hydrogens in the states the input states. Returns it with what ``read_protein`` found."""
    protein = read_protein(path, fix=fix)
    modeller = app.Modeller(protein.topology, protein.positions * unit.nanometer)
    protein.notes += complete_hydrogens(modeller, forcefield, 7.0, protein.variants)
    return modeller.topology, np.array(modeller.positions.value_in_unit(unit.nanometer)), protein


def atoms_of(source):
    """Positions of the atoms of a structure file by (residue index, atom name)."""
    topology, positions = read_structure(source)
    return {(a.residue.index, a.name): positions[a.index] for a in topology.atoms()}


def atom_position(topology, positions, index, name):
    return positions[next(a.index for a in list(topology.residues())[index].atoms() if a.name == name)]


def clearance(topology, positions, index, neighbour):
    """Smallest distance from the heavy atoms of residue ``index`` to the heavy atoms of all
    residues but itself and the one it is bonded to."""
    heavy = [a for a in topology.atoms() if a.element.symbol != "H"]
    own = np.array([positions[a.index] for a in heavy if a.residue.index == index])
    rest = np.array([positions[a.index] for a in heavy if a.residue.index not in (index, neighbour)])
    return np.linalg.norm(own[:, None] - rest[None], axis=2).min()


def net_charge(forcefield, topology):
    system = forcefield.createSystem(topology)
    nonbonded = next(f for f in system.getForces() if f.getName() == "NonbondedForce")
    return sum(nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
               for i in range(system.getNumParticles()))


def names_of(topology, index):
    return {atom.name for atom in list(topology.residues())[index].atoms()}


def capped_fragment(path, source, style="amber"):
    """Residues 6-12 of the test protein between an acetyl and an N-methyl group cut from
    their neighbours. ``style="amber"``: residues ACE and NME in place. ``"unk"``: as a
    preparation program writes them, residues UNK with atoms C1, O1, C2 / N1, C1 after the
    chain."""
    def skip(atom):
        index, name = atom.residue.index, atom.name
        return (index < 4 or index > 12 or (index == 4 and name not in ("CA", "C", "O"))
                or (index == 12 and name not in ("N", "CA")))
    if style == "amber":
        return edited(path, source, skip, residues={4: "ACE", 12: "NME"},
                      atoms={(4, "CA"): "CH3", (12, "CA"): "C"})
    edited(path, source, skip, residues={4: "UNK", 12: "UNK"},
           atoms={(4, "CA"): "C2", (4, "C"): "C1", (4, "O"): "O1", (12, "N"): "N1", (12, "CA"): "C1"})
    lines = [line for line in open(path).read().splitlines() if line.startswith(("ATOM", "HETATM"))]
    caps = ["HETATM" + line[6:21] + " " + line[22:] for line in lines if line[17:20] == "UNK"]
    chain = [line for line in lines if line[17:20] != "UNK"]
    path.write_text("\n".join(chain + ["TER"] + caps + ["END"]) + "\n")
    return path


def protonated(path, source, foreign=False):
    """The test protein with HIP 27, ASH 3, GLH 4 and a neutral LYS 7 written in its hydrogens
    (net charge +4); ``foreign`` gives one hydrogen a name the force field does not know."""
    at = atoms_of(source)

    def hydrogen(residue, parent, *others):
        away = at[residue, parent] - np.mean([at[residue, name] for name in others], axis=0)
        return at[residue, parent] + 0.1 * away / np.linalg.norm(away)

    extra = [(26, "HD1", "H", hydrogen(26, "ND1", "CG", "CE1")),
             (2, "HD2", "H", hydrogen(2, "OD2", "CG")), (3, "HE2", "H", hydrogen(3, "OE2", "CD"))]
    return edited(path, source, skip=lambda atom: (atom.residue.index, atom.name) == (6, "HZ1"),
                  atoms={(0, "HB2"): "XB2"} if foreign else None, extra=extra)


def with_cysteines(path, source, separation):
    """PRO 21 and LYS 24 of the test protein turned into cysteines (no HG) whose sulfur
    atoms are ``separation`` nm apart; the net charge without the lysine is +1."""
    at = atoms_of(source)
    axis = at[23, "CB"] - at[20, "CB"]
    half = (np.linalg.norm(axis) - separation) / 2.0
    axis /= np.linalg.norm(axis)
    out = 0.5 * (at[20, "CB"] + at[23, "CB"] - at[20, "CA"] - at[23, "CA"])
    out -= (out @ axis) * axis
    out *= np.sqrt(0.181 ** 2 - half ** 2) / np.linalg.norm(out)
    backbone = ("N", "H", "CA", "HA", "C", "O", "CB", "HB2", "HB3")
    return edited(path, source, residues={20: "CYS", 23: "CYS"},
                  skip=lambda atom: atom.residue.index in (20, 23) and atom.name not in backbone,
                  extra=[(20, "SG", "S", at[20, "CB"] + half * axis + out),
                         (23, "SG", "S", at[23, "CB"] - half * axis + out)])


def with_ter(path, number):
    """Put a TER record behind residue ``number`` of a PDB file: for OpenMM the rest is then
    a chain of its own."""
    lines = path.read_text().splitlines()
    last = max(i for i, line in enumerate(lines) if line.startswith("ATOM") and int(line[22:26]) == number)
    path.write_text("\n".join(lines[:last + 1] + ["TER"] + lines[last + 1:]) + "\n")
    return path


@pytest.mark.parametrize("case", ["hydrogens", "filled", "heavy", "heavy_ter"])
def test_gap_in_the_chain_is_split_and_capped(inputs, tmp_path, amber, case):
    """Three residues are missing: the chain ends there in an NME and starts again with an ACE
    (OpenMM alone bonds ARG 14 to ALA 18 over 0.447 nm and the force field accepts it). The
    caps stand where the missing neighbours were. ``filled``: a preparation program put a
    hydrogen on each open end (an aldehyde and an amine, which are no termini). Heavy atoms
    only: through PDBFixer, which makes charged termini of the two ends when a TER record
    marks the gap and nothing caps it."""
    hydrogens = case in ("hydrogens", "filled")
    if not hydrogens:
        pytest.importorskip("pdbfixer")
    missing = atoms_of(inputs[0])

    def filler(index, parent, was):
        bond = missing[was] - missing[index, parent]
        return missing[index, parent] + 0.1 * bond / np.linalg.norm(bond)

    path = edited(tmp_path / "gap.pdb", inputs[0], skip=lambda atom: atom.residue.index in (14, 15, 16)
                  or (not hydrogens and atom.element.symbol == "H"),
                  extra=[(13, "HXT", "H", filler(13, "C", (14, "N"))),
                         (17, "H2", "H", filler(17, "N", (16, "C")))] if case == "filled" else ())
    if case == "heavy_ter":
        with_ter(path, 14)
    topology, positions, protein = modelled(path, amber, fix=not hydrogens)

    names = [residue.name for residue in topology.residues()]
    assert topology.getNumChains() == 2 and len(names) == 34 and names[14:16] == ["NME", "ACE"]
    lengths = [np.linalg.norm(positions[a.index] - positions[b.index])
               for a, b in topology.bonds() if a.residue is not b.residue]
    assert len(lengths) == 32 and max(lengths) < 0.15
    assert net_charge(amber, topology) == pytest.approx(2.0, abs=1e-6)       # no charged ends added
    assert protein.info["gaps_capped"] == [dict(after="A:ARG14", before="A:ALA18", c_n_nm=0.447,
                                                caps=["NME", "ACE"])]
    assert any("residues are missing between A:ARG14 and A:ALA18" in note for note in protein.notes)
    for cap, name, was in ((14, "N", (14, "N")), (14, "C", (14, "CA")), (15, "C", (16, "C")),
                           (15, "O", (16, "O")), (15, "CH3", (16, "CA"))):
        assert np.linalg.norm(atom_position(topology, positions, cap, name) - missing[was]) < 0.1
    assert clearance(topology, positions, 14, 13) > 0.25 and clearance(topology, positions, 15, 16) > 0.25


def test_gap_of_one_residue_is_refused(inputs, tmp_path):
    """The two caps would stand on the one missing residue: there is no model without it."""
    path = edited(tmp_path / "gap.pdb", inputs[0], skip=lambda atom: atom.residue.index == 15)
    with pytest.raises(ValueError, match="A:SER15 and A:PHE17 are not bonded .* no room for caps"):
        read_protein(path, fix=False)


@pytest.mark.parametrize("index, name", [(0, "CA"), (15, "C")])
def test_residue_without_a_backbone_atom_is_rebuilt_not_dropped(inputs, tmp_path, amber, index, name):
    """An amino acid that lacks N, CA or C is not removed with the hetero residues (the chain
    then lost its first residue, or was cut at the residue): its atoms are rebuilt before the
    chains are split, and without PDBFixer the file is refused."""
    whole = read_structure(inputs[0])[0]
    residue = list(whole.residues())[index]
    path = edited(tmp_path / "broken.pdb", inputs[0],
                  skip=lambda atom: atom.residue.index == index and atom.name == name)
    with pytest.raises(ValueError, match=rf"A:{residue.name}{residue.id} \({name}\)"):
        read_protein(path, fix=False)

    pytest.importorskip("pdbfixer")
    topology, positions, protein = modelled(path, amber, fix=True)
    assert (topology.getNumChains(), topology.getNumResidues(), topology.getNumAtoms()) == (1, 35, 582)
    assert protein.info["removed"] == {} and protein.info["gaps_capped"] == []
    assert any(f"A:{residue.name}{residue.id} ({name})" in note for note in protein.notes)
    # measured: 0.004 to 0.022 nm from the atom of the complete file, over five residues
    assert np.linalg.norm(atom_position(topology, positions, index, name) - atoms_of(inputs[0])[index, name]) < 0.05


@pytest.mark.parametrize("stated", ["OXT", "H3"])
def test_chain_end_stated_by_the_input_is_not_capped(inputs, tmp_path, amber, stated):
    """Two chains under one identifier with the numbering going on (a file from an MD
    program): the OXT of the first states its end, or the three hydrogens on the nitrogen of
    the second its start, so the two ends stay termini."""
    at = atoms_of(inputs[0])
    if stated == "OXT":
        path = edited(tmp_path / "chains.pdb", inputs[0], extra=[(13, "OXT", "O", at[14, "N"])],
                      skip=lambda atom: atom.residue.index in (14, 15, 16) or atom.element.symbol == "H")
    else:
        pytest.importorskip("pdbfixer")                       # it completes the end without OXT
        to_h, to_ca = (at[17, name] - at[17, "N"] for name in ("H", "CA"))
        away = -(to_h / np.linalg.norm(to_h) + to_ca / np.linalg.norm(to_ca))
        normal = np.cross(to_h, to_ca)
        tilted = [away / np.linalg.norm(away) + sign * normal / np.linalg.norm(normal) for sign in (1.0, -1.0)]
        path = edited(tmp_path / "chains.pdb", inputs[0], skip=lambda atom: atom.residue.index in (14, 15, 16),
                      extra=[(17, name, "H", at[17, "N"] + 0.1 * bond / np.linalg.norm(bond))
                             for name, bond in zip(("H2", "H3"), tilted)])
    topology, _, protein = modelled(with_ter(path, 14), amber, fix=stated == "H3")
    names = [residue.name for residue in topology.residues()]
    assert topology.getNumChains() == 2 and len(names) == 32 and not {"ACE", "NME"} & set(names)
    assert "OXT" in names_of(topology, 13) and {"H2", "H3"} <= names_of(topology, 14)
    assert protein.info["gaps_capped"] == []


def test_kept_ion_listed_in_the_protein_chain(inputs, tmp_path):
    """A kept ion that follows the last residue without a TER record is read into the protein
    chain (OPM files). It gets a chain of its own, so PDBFixer sees the C terminus and
    completes it (before FM-D15: no OXT, and no template for the last residue)."""
    pytest.importorskip("pdbfixer")
    path = edited(tmp_path / "ion.pdb", inputs[0],
                  skip=lambda atom: atom.element.symbol == "H" or atom.name == "OXT")
    lines = [line for line in path.read_text().splitlines() if line.startswith("ATOM")]
    x, y, z = 10.0 * read_structure(path)[1].max(axis=0) + 5.0
    ion = f"HETATM{len(lines) + 1:5d} NA    NA A  36    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          NA"
    path.write_text("\n".join(lines + [ion, "END"]) + "\n")
    assert read_structure(path)[0].getNumChains() == 1

    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    protein = read_protein(path, fix=True, keep_hetero=["NA"])
    modeller = app.Modeller(protein.topology, protein.positions * unit.nanometer)
    complete_hydrogens(modeller, forcefield, 7.0, protein.variants)
    chains = [[residue.name for residue in chain.residues()] for chain in modeller.topology.chains()]
    assert len(chains[0]) == 35 and chains[1:] == [["NA"]]
    assert "OXT" in names_of(modeller.topology, 34) and protein.info["kept_hetero"] == {"NA": 1}
    assert net_charge(forcefield, modeller.topology) == pytest.approx(3.0, abs=1e-6)


@pytest.mark.parametrize("fix", [False, True])
@pytest.mark.parametrize("style", ["amber", "unk"])
def test_caps_in_the_input_are_kept(inputs, tmp_path, amber, style, fix):
    """ACE and NME of the input, under their own names or as the UNK residues a preparation
    program writes after the chain, stay where they are: the ends are not made charged
    termini (before FM-D15 the caps were removed with the hetero components)."""
    if fix:
        pytest.importorskip("pdbfixer")
    at = atoms_of(inputs[0])
    given = {("ACE", "C"): at[4, "C"], ("ACE", "O"): at[4, "O"], ("ACE", "CH3"): at[4, "CA"],
             ("NME", "N"): at[12, "N"], ("NME", "C"): at[12, "CA"]}
    topology, positions, protein = modelled(capped_fragment(tmp_path / "capped.pdb", inputs[0], style),
                                            amber, fix=fix)
    names = [residue.name for residue in topology.residues()]
    assert names == ["ACE", "PHE", "LYS", "ALA", "VAL", "PHE", "GLY", "MET", "NME"]
    assert topology.getNumChains() == 1
    assert not {"H2", "H3"} & names_of(topology, 1) and "OXT" not in names_of(topology, 7)
    assert net_charge(amber, topology) == pytest.approx(1.0, abs=1e-6)       # the lysine alone
    assert protein.info["caps_kept"] == ["ACE before A:PHE6", "NME after A:MET12"]
    for (residue, name), position in given.items():
        found = atom_position(topology, positions, names.index(residue), name)
        assert np.abs(found - position).max() < 1e-6


def test_protonation_states_of_the_input_are_kept(inputs, tmp_path, amber):
    """HIP, ASH, GLH and a neutral lysine written in the hydrogens of the input reach the
    model (+4), also when a hydrogen with a foreign name makes fmeta rebuild them all (before
    FM-D15: +5 with the lysine charged again, and +2 after a rebuild). A file without
    hydrogens states them in its residue names. The report of ``prepare`` holds what was taken."""
    states = {"ASH": ["A:ASP3"], "GLH": ["A:GLU4"], "LYN": ["A:LYS7"], "HIP": ["A:HIS27"]}
    named = edited(tmp_path / "named.pdb", inputs[0], skip=lambda atom: atom.element.symbol == "H",
                   residues={2: "ASH", 3: "GLH", 6: "LYN", 26: "HIP"})
    for path, foreign in ((named, False), (protonated(tmp_path / "written.pdb", inputs[0]), False),
                          (protonated(tmp_path / "foreign.pdb", inputs[0], foreign=True), True)):
        topology, _, protein = modelled(path, amber)
        assert any("were rebuilt" in note for note in protein.notes) == foreign
        assert {"HD1", "HE2"} <= names_of(topology, 26) and "HD2" in names_of(topology, 2)
        assert "HE2" in names_of(topology, 3) and "HZ3" not in names_of(topology, 6)
        assert net_charge(amber, topology) == pytest.approx(4.0, abs=1e-6)
        assert protein.info["protonation_from_input"] == states

    built = prepare(path, inputs[1], SystemSettings(fix_protein=False, **LIGAND_SETTINGS))
    found = json.loads(json.dumps(built.report.to_dict(), default=float))["info"]["protein_input"]
    assert found["protonation_from_input"] == states and found["histidines"] == {"HIP": ["A:HIS27"]}
    assert found["disulfides"] == [] and found["gaps_capped"] == [] and found["caps_kept"] == []
    message = {check.name: check.message for check in built.report.checks}["protein_input"]
    assert "protonation states taken from the input: ASH x1, GLH x1, HIP x1, LYN x1" in message


def test_cysteines_of_the_input(inputs, tmp_path, amber):
    """Sulfur atoms 0.204 nm apart are a disulfide (CYX) and are reported. Further apart, in
    a file with hydrogens and none on the sulfur, they are thiolates as the input states
    (before FM-D15 they got HG back); in a file without hydrogens, free cysteines."""
    topology, _, protein = modelled(with_cysteines(tmp_path / "bridge.pdb", inputs[0], 0.204), amber)
    assert "HG" not in names_of(topology, 20) | names_of(topology, 23)
    assert net_charge(amber, topology) == pytest.approx(1.0, abs=1e-6)
    assert model_states(topology)["disulfides"] == [["A:CYS21", "A:CYS24"]]
    assert "CYM" not in protein.info["protonation_from_input"]

    topology, _, protein = modelled(with_cysteines(tmp_path / "free.pdb", inputs[0], 0.35), amber)
    assert "HG" not in names_of(topology, 20) | names_of(topology, 23)
    assert net_charge(amber, topology) == pytest.approx(-1.0, abs=1e-6)
    assert model_states(topology)["disulfides"] == []
    assert protein.info["protonation_from_input"]["CYM"] == ["A:CYS21", "A:CYS24"]

    bare = edited(tmp_path / "bare.pdb", tmp_path / "free.pdb", skip=lambda atom: atom.element.symbol == "H")
    topology, _, protein = modelled(bare, amber)
    assert "HG" in names_of(topology, 20) and "HG" in names_of(topology, 23)
    assert net_charge(amber, topology) == pytest.approx(1.0, abs=1e-6)
    assert protein.info["protonation_from_input"] == {}


def test_first_listed_alternate_location_is_taken_and_reported(inputs, tmp_path):
    """Of the alternate locations of an atom OpenMM keeps the one listed first, whatever its
    letter or occupancy; the report says which."""
    lines = []
    for line in open(inputs[0]).read().splitlines():
        if line.startswith("ATOM") and line[22:26].strip() == "5" and line[12:16].strip() in ("CB", "CG"):
            moved = f"{float(line[30:38]) + 1.0:8.3f}"
            lines += [line[:16] + "B" + line[17:54] + "  0.40" + line[60:],
                      line[:16] + "A" + line[17:30] + moved + line[38:54] + "  0.60" + line[60:]]
        else:
            lines.append(line)
    (tmp_path / "altloc.pdb").write_text("\n".join(lines) + "\n")
    protein = read_protein(tmp_path / "altloc.pdb", fix=False)
    assert np.abs(protein.positions - read_structure(inputs[0])[1]).max() < 1e-6
    assert protein.info["alternate_locations"] == [dict(residue="A:ASP5", kept="B", dropped=["A"])]
    assert protein.notes == ["alternate locations: the first listed was taken (A:ASP5 B)",
                             "protonation states taken from the input: HIE x1"]
