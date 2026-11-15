"""Reading structures: protein files, posed ligands, and complexes to split into the two."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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

def fix_protein(path, keep_water: bool = False) -> tuple[app.Topology, np.ndarray]:
    """Repair a protein file with PDBFixer: non-standard residues are replaced by
    their standard parents and missing heavy atoms are built. Missing residues
    (unresolved loops, termini) are not modelled; hydrogens are added later with
    the force field. Hetero groups other than water are removed."""
    try:
        from pdbfixer import PDBFixer
    except ImportError as error:
        raise ImportError("fix_protein needs PDBFixer (conda install -c conda-forge pdbfixer)") from error
    path = Path(path)
    if path.suffix.lower() in (".cif", ".mmcif", ".pdbx"):
        with open(path) as handle:
            fixer = PDBFixer(pdbxfile=handle)
    else:
        fixer = PDBFixer(filename=str(path))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=keep_water)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    return fixer.topology, _nm(fixer.positions)


def split_complex(path, resname: str, out_dir, smiles: Optional[str] = None,
                  chain: Optional[str] = None, resid: Optional[str] = None) -> tuple[str, str]:
    """Split a complex (PDB or mmCIF) into ``protein.pdb`` and ``ligand.sdf``.

    The ligand is the residue called ``resname`` (narrow it down with ``chain``
    and ``resid`` if there are several copies). Crystal structures carry no
    ligand hydrogens or bond orders, so give its SMILES in the protonation state
    you want to simulate; RDKit then assigns bond orders from it and adds the
    hydrogens. Without ``smiles`` the ligand must already have hydrogens and bonds.
    Returns the two file paths.
    """
    topology, positions = read_structure(path)
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
    lines = [name, f"{'':2s}{'fmeta':<8s}{'':10s}3D", "", f"{len(elements):3d}{len(bonds):3d}  0  0  0  0  0  0  0  0999 V2000"]
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
