"""Reading structures: protein files, posed ligands, and complexes to split into the two."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import openmm.app as app
import openmm.unit as unit


def _nm(positions) -> np.ndarray:
    return np.array(positions.value_in_unit(unit.nanometer), dtype=float)


def read_structure(path) -> tuple[app.Topology, np.ndarray]:
    """Read a PDB or PDBx/mmCIF file. Positions are returned in nm."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".pdb", ".ent"):
        structure = app.PDBFile(str(path))
    elif suffix in (".cif", ".mmcif", ".pdbx"):
        structure = app.PDBxFile(str(path))
    else:
        raise ValueError(f"unsupported structure format: {path.name}")
    return structure.topology, _nm(structure.positions)


def _ligand_topology(elements, bonds, resname: str) -> app.Topology:
    topology = app.Topology()
    residue = topology.addResidue(resname, topology.addChain("L"))
    counts: dict[str, int] = {}
    atoms = []
    for symbol in elements:
        symbol = symbol.capitalize()
        counts[symbol] = counts.get(symbol, 0) + 1
        atoms.append(topology.addAtom(f"{symbol}{counts[symbol]}",
                                      app.Element.getBySymbol(symbol), residue))
    kinds = {1: app.Single, 2: app.Double, 3: app.Triple, 4: app.Aromatic}
    for first, second, order in bonds:
        topology.addBond(atoms[first], atoms[second], type=kinds.get(order),
                         order=order if order in (1, 2, 3) else None)
    return topology


def _read_molfile(path, resname: str):
    """Minimal MDL molfile / SDF (V2000) reader: first molecule only."""
    lines = Path(path).read_text().splitlines()
    counts = lines[3]
    if "V3000" in counts:
        raise ValueError("V3000 molfiles are not supported; convert the ligand to V2000 SDF")
    n_atoms, n_bonds = int(counts[0:3]), int(counts[3:6])
    elements, coords = [], []
    for line in lines[4:4 + n_atoms]:
        coords.append([float(line[0:10]), float(line[10:20]), float(line[20:30])])
        elements.append(line[31:34].strip())
    bonds = [(int(line[0:3]) - 1, int(line[3:6]) - 1, int(line[6:9]))
             for line in lines[4 + n_atoms:4 + n_atoms + n_bonds]]
    return _ligand_topology(elements, bonds, resname), np.array(coords) / 10.0


def _read_mol2(path, resname: str):
    """Minimal Tripos MOL2 reader: first molecule only."""
    section, elements, coords, bonds = None, [], [], []
    orders = {"1": 1, "2": 2, "3": 3, "ar": 4, "am": 1}
    for line in Path(path).read_text().splitlines():
        if line.startswith("@<TRIPOS>"):
            if line.strip() == "@<TRIPOS>MOLECULE" and elements:
                break
            section = line.strip()[9:]
            continue
        fields = line.split()
        if section == "ATOM" and len(fields) >= 6:
            coords.append([float(v) for v in fields[2:5]])
            elements.append(fields[5].split(".")[0])
        elif section == "BOND" and len(fields) >= 4:
            bonds.append((int(fields[1]) - 1, int(fields[2]) - 1, orders.get(fields[3].lower(), 1)))
    return _ligand_topology(elements, bonds, resname), np.array(coords) / 10.0


def read_ligand(path, resname: str = "LIG") -> tuple[app.Topology, np.ndarray]:
    """Read a posed ligand with bonds and all hydrogens (SDF/MOL V2000, MOL2, or
    a PDB with CONECT records). Coordinates must be in the protein's frame."""
    suffix = Path(path).suffix.lower()
    if suffix in (".sdf", ".mol"):
        topology, positions = _read_molfile(path, resname)
    elif suffix == ".mol2":
        topology, positions = _read_mol2(path, resname)
    elif suffix == ".pdb":
        topology, positions = read_structure(path)
        if topology.getNumBonds() == 0 and topology.getNumAtoms() > 1:
            raise ValueError("ligand PDB has no CONECT records; bonds are needed to assign parameters")
    else:
        raise ValueError(f"unsupported ligand format: {Path(path).name}")
    if not any(atom.element.symbol == "H" for atom in topology.atoms()):
        raise ValueError("the ligand has no hydrogens; provide the fully protonated pose")
    return topology, positions


# ---- protein clean-up and complexes -----------------------------------------

_BACKBONE = {"N", "CA", "C"}
_WATER = {"HOH", "WAT", "H2O", "TIP3", "TIP", "SOL", "DOD"}


def pdbfixer_available() -> bool:
    try:
        import pdbfixer  # noqa: F401
    except ImportError:
        return False
    return True


def read_protein(path, fix: Optional[bool] = None, keep_water: bool = False,
                 keep_hetero: Sequence[str] = ()) -> tuple[app.Topology, np.ndarray, list[str]]:
    """Read a protein file and reduce it to what will be simulated.

    Everything that is not an amino-acid residue is removed: other ligands,
    ions, cofactors, buffer molecules and (unless ``keep_water``) water. Name
    residues in ``keep_hetero`` to keep them; the force-field files must then
    cover them. With ``fix`` (default: whenever PDBFixer is installed)
    non-standard amino acids are replaced by their standard parents and missing
    heavy atoms are built; unresolved loops are not modelled.

    Returns the topology, the positions in nm and a list of notes saying what
    was changed.
    """
    topology, positions = read_structure(path)
    notes = []
    keep = {name.strip().upper() for name in keep_hetero}
    removed: dict[str, int] = {}
    doomed = []
    for residue in topology.residues():
        name = residue.name.strip().upper()
        amino_acid = _BACKBONE <= {atom.name for atom in residue.atoms()}
        if amino_acid or name in keep or (keep_water and name in _WATER):
            continue
        doomed.append(residue)
        removed[name] = removed.get(name, 0) + 1
    modeller = app.Modeller(topology, positions * unit.nanometer)
    if doomed:
        modeller.delete(doomed)
        listing = ", ".join(f"{name} x{count}" for name, count in sorted(removed.items()))
        notes.append(f"removed from the protein file: {listing}")
    if modeller.topology.getNumAtoms() == 0:
        raise ValueError(f"{path} contains no amino-acid residues")

    use_fixer = pdbfixer_available() if fix is None else fix
    if use_fixer:
        modeller, fixed = _fix_with_pdbfixer(modeller)
        notes += fixed
    return modeller.topology, _nm(modeller.positions), notes


def _fix_with_pdbfixer(modeller: app.Modeller) -> tuple[app.Modeller, list[str]]:
    try:
        from pdbfixer import PDBFixer
    except ImportError as error:
        raise ImportError("fix_protein needs PDBFixer (conda install -c conda-forge pdbfixer)") from error
    import io
    text = io.StringIO()
    app.PDBFile.writeFile(modeller.topology, modeller.positions, text, keepIds=True)
    fixer = PDBFixer(pdbfile=io.StringIO(text.getvalue()))
    fixer.findMissingResidues()
    fixer.missingResidues = {}                    # unresolved loops are not modelled
    fixer.findNonstandardResidues()
    notes = []
    if fixer.nonstandardResidues:
        swaps = sorted({f"{residue.name} -> {standard}" for residue, standard in fixer.nonstandardResidues})
        notes.append("non-standard residues replaced: " + ", ".join(swaps))
        fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    n_missing = (sum(len(atoms) for atoms in fixer.missingAtoms.values())
                 + sum(len(atoms) for atoms in fixer.missingTerminals.values()))
    if n_missing:
        notes.append(f"{n_missing} missing heavy atoms rebuilt")
        fixer.addMissingAtoms()
    return app.Modeller(fixer.topology, fixer.positions), notes


def complete_hydrogens(modeller: app.Modeller, forcefield: app.ForceField, ph: float) -> list[str]:
    """Add the protein's missing hydrogens in place. If the hydrogens already in
    the file do not fit the force field's naming, they are removed and rebuilt."""
    notes = []
    before = modeller.topology.getNumAtoms()
    try:
        modeller.addHydrogens(forcefield, pH=ph)
    except Exception:
        hydrogens = [a for a in modeller.topology.atoms()
                     if a.element is not None and a.element.symbol == "H"]
        modeller.delete(hydrogens)
        before = modeller.topology.getNumAtoms()
        try:
            modeller.addHydrogens(forcefield, pH=ph)
        except Exception as error:
            hint = "" if pdbfixer_available() else (
                " Installing PDBFixer (conda install -c conda-forge pdbfixer) lets fmeta repair "
                "missing atoms and non-standard residues automatically.")
            raise ValueError("the protein does not fit the force field's residue templates "
                             f"({error}).{hint}") from error
        notes.append(f"{len(hydrogens)} hydrogens in the protein file did not match the force "
                     "field and were rebuilt")
    added = modeller.topology.getNumAtoms() - before
    if added:
        notes.append(f"{added} hydrogens added to the protein at pH {ph:g}")
    return notes


def ensure_ligand_hydrogens(path, out_dir) -> tuple[str, list[str]]:
    """Return a ligand file that has hydrogens.

    If the given SDF has none, they are added with RDKit at the positions the
    heavy atoms imply, following the formal charges in the file, and the result
    is written to ``out_dir``. The heavy-atom pose is not changed.
    """
    path = Path(path)
    if path.suffix.lower() not in (".sdf", ".mol"):
        return str(path), []
    topology, _ = _read_molfile(path, "LIG")
    if any(atom.element.symbol == "H" for atom in topology.atoms()):
        return str(path), []
    try:
        from rdkit import Chem
    except ImportError as error:
        raise ImportError("the ligand SDF has no hydrogens; adding them needs RDKit "
                          "(pip install rdkit), or supply a protonated SDF") from error
    molecule = Chem.MolFromMolFile(str(path), removeHs=False)
    if molecule is None:
        raise ValueError(f"RDKit cannot read {path}; check the SDF")
    protonated = Chem.AddHs(molecule, addCoords=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}_with_hydrogens.sdf"
    out.write_text(Chem.MolToMolBlock(protonated) + "$$$$\n")
    added = protonated.GetNumAtoms() - molecule.GetNumAtoms()
    charge = Chem.GetFormalCharge(protonated)
    return str(out), [f"the ligand SDF had no hydrogens: {added} added for the structure as drawn "
                      f"(net charge {charge:+d}); check that this is the protonation state you want"]


_DROP = re.compile(r"(?P<chain>[^:]*)(?::(?P<first>-?\d+)(?P<first_code>[A-Za-z]?)"
                   r"(?:-(?P<last>-?\d+)(?P<last_code>[A-Za-z]?))?)?")
_LAST_CODE = "￿"      # sorts after every insertion code: a bound without a code takes them all


def _parse_drop(selection: str):
    """``CHAIN``, ``CHAIN:NUMBER`` or ``CHAIN:FIRST-LAST`` -> (chain, first key, last key).

    A key is (residue number, insertion code). A bound written without an insertion code
    stands for every residue of that number; ``52A`` is that one residue."""
    text = selection.strip()
    match = _DROP.fullmatch(text)
    if not text or match is None:
        raise ValueError(f"cannot read the drop selection {selection!r}: write CHAIN, CHAIN:NUMBER "
                         "or CHAIN:FIRST-LAST, e.g. A:1001-1106")
    chain = match["chain"].strip()
    if match["first"] is None:
        return chain, None, None
    first = (int(match["first"]), match["first_code"].upper())
    if match["last"] is None:
        last = (first[0], first[1] or _LAST_CODE)
    else:
        last = (int(match["last"]), match["last_code"].upper() or _LAST_CODE)
    if last < first:
        raise ValueError(f"the drop selection {selection!r} ends before it starts")
    return chain, first, last


def _residue_key(residue):
    """(number, insertion code) of a residue, or None if its id is not a number."""
    try:
        return int(residue.id), residue.insertionCode.strip().upper()
    except ValueError:
        return None


def drop_residues(topology: app.Topology, positions: np.ndarray,
                  selections: Sequence[str]) -> tuple[app.Topology, np.ndarray, list[str]]:
    """Remove the residues named by ``selections`` (see ``_parse_drop``).

    Residue numbers are compared as integers with the insertion code as a tie-break, so
    ``A:1001-1106`` takes 1050A with it and leaves residue 11 alone. A selection takes every
    residue of the chain in its range, protein or not. One that matches nothing is an error:
    a part that was meant to go must not stay unnoticed.

    Returns the new topology, the positions in nm and one note per selection."""
    doomed, notes = [], []
    for selection in selections:
        chain, first, last = _parse_drop(selection)
        matches = []
        for residue in topology.residues():
            if residue.chain.id.strip() != chain:
                continue
            key = _residue_key(residue)
            if first is None or (key is not None and first <= key <= last):
                matches.append(residue)
        if not matches:
            spans = []
            for candidate in topology.chains():
                numbers = [key[0] for key in map(_residue_key, candidate.residues()) if key]
                span = f" (residues {min(numbers)} to {max(numbers)})" if numbers else ""
                spans.append(f"{candidate.id.strip() or 'blank'}{span}")
            raise ValueError(f"the drop selection {selection!r} matches no residue; chains in the "
                             f"file: {', '.join(dict.fromkeys(spans))}")
        doomed += matches
        ends = " to ".join(dict.fromkeys(f"{r.name}{r.id}{r.insertionCode.strip()}"
                                         for r in (matches[0], matches[-1])))
        count = f"{len(matches)} residue{'' if len(matches) == 1 else 's'}"
        notes.append(f"dropped {selection.strip()}: {count} ({ends})")
    modeller = app.Modeller(topology, positions * unit.nanometer)
    modeller.delete(doomed)
    return modeller.topology, _nm(modeller.positions), notes


def split_complex(path, resname: str, out_dir, smiles: Optional[str] = None,
                  chain: Optional[str] = None, resid: Optional[str] = None,
                  drop: Sequence[str] = (), notes: Optional[list] = None) -> tuple[str, str]:
    """Split a complex (PDB or mmCIF) into ``protein.pdb`` and ``ligand.sdf``.

    The ligand is the residue called ``resname`` (narrow it down with ``chain``
    and ``resid`` if there are several copies). Crystal structures carry no
    ligand hydrogens or bond orders, so give its SMILES in the protonation state
    you want to simulate; RDKit then assigns bond orders from it and adds the
    hydrogens. Without ``smiles`` the ligand must already have hydrogens and bonds.

    ``drop`` removes parts of the file first (``drop_residues``), e.g. a fusion partner
    such as ``A:1001-1106``: nothing after this point sees them, neither the ligand search
    nor the membrane orientation estimate that reads ``protein.pdb``. What was dropped is
    appended to ``notes`` if a list is given.
    Returns the two file paths.
    """
    topology, positions = read_structure(path)
    if drop:
        topology, positions, dropped = drop_residues(topology, positions, drop)
        if notes is not None:
            notes += dropped
    wanted = resname.strip().upper()
    matches = [r for r in topology.residues() if r.name.strip().upper() == wanted
               and (chain is None or r.chain.id == chain)
               and (resid is None or str(r.id) == str(resid))]
    if len(matches) != 1:
        found = ", ".join(f"chain {r.chain.id} residue {r.id}" for r in matches) or "none"
        raise ValueError(f"expected exactly one ligand residue {resname!r}, found: {found}; "
                         "select one with chain= and resid=")
    ligand = matches[0]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modeller = app.Modeller(topology, positions * unit.nanometer)
    modeller.delete([ligand])
    protein_path = out_dir / "protein.pdb"
    with open(protein_path, "w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)

    ligand_atoms = list(ligand.atoms())
    index = {atom.index: k for k, atom in enumerate(ligand_atoms)}
    coords = positions[[atom.index for atom in ligand_atoms]]
    elements = [atom.element.symbol for atom in ligand_atoms]
    bonds = [(index[a.index], index[b.index], 1) for a, b in topology.bonds()
             if a.index in index and b.index in index]
    ligand_path = out_dir / "ligand.sdf"
    if smiles is None:
        if "H" not in elements or len(bonds) < len(elements) - 1:
            raise ValueError("the ligand in the complex has no hydrogens or bonds; pass its SMILES "
                             "(in the protonation state to simulate) so they can be added")
        _write_molfile(ligand_path, resname, elements, coords, bonds)
    else:
        _write_ligand_from_smiles(ligand_path, resname, elements, coords, smiles)
    return str(protein_path), str(ligand_path)


def _write_molfile(path, name: str, elements, coords_nm, bonds) -> None:
    lines = [name, f"{'':2s}{'fmeta':<8s}{'':10s}3D", "",
             f"{len(elements):3d}{len(bonds):3d}  0  0  0  0  0  0  0  0999 V2000"]
    for symbol, (x, y, z) in zip(elements, np.asarray(coords_nm) * 10.0):
        lines.append(f"{x:10.4f}{y:10.4f}{z:10.4f} {symbol:<3s} 0  0  0  0  0  0  0  0  0  0  0  0")
    lines += [f"{a + 1:3d}{b + 1:3d}{order:3d}  0" for a, b, order in bonds]
    lines += ["M  END", "$$$$"]
    Path(path).write_text("\n".join(lines) + "\n")


def _write_ligand_from_smiles(path, name: str, elements, coords_nm, smiles: str) -> None:
    """Heavy-atom coordinates + SMILES -> protonated molecule with bond orders (RDKit)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Geometry import Point3D
    except ImportError as error:
        raise ImportError("adding ligand hydrogens from a SMILES needs RDKit "
                          "(pip install rdkit)") from error
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"RDKit cannot parse the SMILES {smiles!r}")
    heavy = [k for k, symbol in enumerate(elements) if symbol != "H"]
    if template.GetNumAtoms() != len(heavy):
        raise ValueError(f"the SMILES has {template.GetNumAtoms()} heavy atoms but the ligand "
                         f"residue has {len(heavy)}")
    # bare heavy atoms with the crystal coordinates; connectivity from interatomic distances
    editable = Chem.RWMol()
    for k in heavy:
        editable.AddAtom(Chem.Atom(elements[k]))
    conformer = Chem.Conformer(len(heavy))
    for new, k in enumerate(heavy):
        x, y, z = np.asarray(coords_nm[k]) * 10.0
        conformer.SetAtomPosition(new, Point3D(float(x), float(y), float(z)))
    editable.AddConformer(conformer)
    from rdkit.Chem import rdDetermineBonds
    rdDetermineBonds.DetermineConnectivity(editable)
    mismatch = ("the ligand's connectivity in the structure does not match the SMILES "
                f"{smiles!r}")
    if editable.GetNumBonds() != template.GetNumBonds():   # e.g. a ring given as an open chain
        raise ValueError(mismatch)
    try:
        molecule = AllChem.AssignBondOrdersFromTemplate(template, editable.GetMol())
    except ValueError as error:
        raise ValueError(mismatch) from error
    for atom in molecule.GetAtoms():      # the bare atoms were created as hydrogen-less radicals
        atom.SetNoImplicit(False)
        atom.SetNumRadicalElectrons(0)
    Chem.SanitizeMol(molecule)
    molecule = Chem.AddHs(molecule, addCoords=True)
    molecule.SetProp("_Name", name)
    Path(path).write_text(Chem.MolToMolBlock(molecule) + "$$$$\n")
