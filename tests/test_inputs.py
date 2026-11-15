"""Input routes: complexes, protein repair, wrapped coordinates, ligand parameters."""

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
from fmeta.workflow import prepare_stage


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
    """The protein model ``build_complex`` makes of a file: reduced, repaired, with hydrogens."""
    topology, positions, notes = read_protein(path, fix=fix)
    modeller = app.Modeller(topology, positions * unit.nanometer)
    notes += complete_hydrogens(modeller, forcefield, 7.0)
    return modeller.topology, np.array(modeller.positions.value_in_unit(unit.nanometer)), notes


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
    topology, positions = read_structure(source)
    at = {(a.residue.index, a.name): positions[a.index] for a in topology.atoms()}

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
    topology, positions = read_structure(source)
    at = {(a.residue.index, a.name): positions[a.index] for a in topology.atoms()}
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


def test_gap_in_the_chain_is_bonded_across(inputs, tmp_path, amber):
    """Characterisation: OpenMM bonds consecutive residues of a chain whatever their distance,
    and the force field accepts the result, so three missing residues go unnoticed."""
    path = edited(tmp_path / "gap.pdb", inputs[0], skip=lambda atom: atom.residue.index in (14, 15, 16))
    topology, positions, notes = modelled(path, amber)
    lengths = [np.linalg.norm(positions[a.index] - positions[b.index])
               for a, b in topology.bonds() if a.residue is not b.residue]
    assert topology.getNumChains() == 1 and topology.getNumResidues() == 32
    assert max(lengths) == pytest.approx(0.447, abs=1e-3)              # ARG 14 C - ALA 18 N
    amber.createSystem(topology)
    assert notes == []


@pytest.mark.parametrize("fix", [False, True])
def test_caps_in_the_input_are_lost(inputs, tmp_path, amber, fix):
    """Characterisation: ACE and NME have no backbone, so they are removed with the hetero
    components. Without PDBFixer the uncapped ends then fit no template; with it they are
    completed as charged termini."""
    path = capped_fragment(tmp_path / "capped.pdb", inputs[0])
    if not fix:
        with pytest.raises(ValueError, match="does not fit the force field"):
            modelled(path, amber, fix=False)
        return
    pytest.importorskip("pdbfixer")
    topology, _, notes = modelled(path, amber, fix=True)
    assert [r.name for r in topology.residues()] == ["PHE", "LYS", "ALA", "VAL", "PHE", "GLY", "MET"]
    assert "removed from the protein file: ACE x1, NME x1" in notes
    assert {"H2", "H3"} <= names_of(topology, 0) and "OXT" in names_of(topology, 6)


def test_protonation_states_of_the_input(inputs, tmp_path, amber):
    """Characterisation: hydrogens that fit the force field keep the states of histidine and
    of the acids, but a neutral lysine is completed to the charged one (+5, not +4); once
    the hydrogens are rebuilt every state of the input is lost (+2)."""
    topology, _, notes = modelled(protonated(tmp_path / "states.pdb", inputs[0]), amber)
    assert {"HD1", "HE2"} <= names_of(topology, 26) and "HD2" in names_of(topology, 2)
    assert "HE2" in names_of(topology, 3) and {"HZ1", "HZ2", "HZ3"} <= names_of(topology, 6)
    assert net_charge(amber, topology) == pytest.approx(5.0, abs=1e-6)

    topology, _, notes = modelled(protonated(tmp_path / "foreign.pdb", inputs[0], foreign=True), amber)
    assert any("did not match the force field and were rebuilt" in note for note in notes)
    assert net_charge(amber, topology) == pytest.approx(2.0, abs=1e-6)


def test_cysteines_of_the_input(inputs, tmp_path, amber):
    """Characterisation: sulfur atoms 0.204 nm apart are bonded on reading and stay without
    HG (CYX); further apart they are free cysteines, which get their HG back even when the
    file has hydrogens and none on the sulfur."""
    topology, _, _ = modelled(with_cysteines(tmp_path / "bridge.pdb", inputs[0], 0.204), amber)
    assert sum(a.name == "SG" and b.name == "SG" for a, b in topology.bonds()) == 1
    assert "HG" not in names_of(topology, 20) | names_of(topology, 23)
    assert net_charge(amber, topology) == pytest.approx(1.0, abs=1e-6)

    topology, _, _ = modelled(with_cysteines(tmp_path / "free.pdb", inputs[0], 0.35), amber)
    assert sum(a.name == "SG" and b.name == "SG" for a, b in topology.bonds()) == 0
    assert "HG" in names_of(topology, 20) and "HG" in names_of(topology, 23)
    assert net_charge(amber, topology) == pytest.approx(1.0, abs=1e-6)


def test_first_listed_alternate_location_is_taken(inputs, tmp_path):
    """Characterisation: of the alternate locations of an atom OpenMM keeps the one listed
    first, whatever its letter or occupancy."""
    lines = []
    for line in open(inputs[0]).read().splitlines():
        if line.startswith("ATOM") and line[22:26].strip() == "5" and line[12:16].strip() in ("CB", "CG"):
            moved = f"{float(line[30:38]) + 1.0:8.3f}"
            lines += [line[:16] + "B" + line[17:54] + "  0.40" + line[60:],
                      line[:16] + "A" + line[17:30] + moved + line[38:54] + "  0.60" + line[60:]]
        else:
            lines.append(line)
    (tmp_path / "altloc.pdb").write_text("\n".join(lines) + "\n")
    topology, positions, notes = read_protein(tmp_path / "altloc.pdb", fix=False)
    assert np.abs(positions - read_structure(inputs[0])[1]).max() < 1e-6 and notes == []
