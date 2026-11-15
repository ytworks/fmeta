"""Reading structures: protein files, posed ligands, and pre-parameterised systems."""

from __future__ import annotations

from pathlib import Path

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
    elif suffix == ".gro":
        raise ValueError("a .gro file carries no bonds; use load_gromacs(top, gro)")
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


# ---- pre-parameterised systems ---------------------------------------------

def _system_kwargs(cutoff: float, hydrogen_mass) -> dict:
    kwargs = dict(nonbondedMethod=app.PME, nonbondedCutoff=cutoff * unit.nanometer,
                  constraints=app.HBonds, rigidWater=True)
    if hydrogen_mass:
        kwargs["hydrogenMass"] = hydrogen_mass * unit.amu
    return kwargs


def load_amber(prmtop, coordinates, cutoff: float = 1.0, hydrogen_mass=1.5):
    """Load an AMBER prmtop with inpcrd/rst7 coordinates. Returns (topology, system, positions)."""
    top = app.AmberPrmtopFile(str(prmtop))
    crd = app.AmberInpcrdFile(str(coordinates))
    if crd.boxVectors is not None:
        top.topology.setPeriodicBoxVectors(crd.boxVectors)
    system = top.createSystem(**_system_kwargs(cutoff, hydrogen_mass))
    return top.topology, system, _nm(crd.positions)


def load_gromacs(top, gro, include_dir=None, cutoff: float = 1.0, hydrogen_mass=1.5):
    """Load a GROMACS topology with a .gro file. Returns (topology, system, positions)."""
    coordinates = app.GromacsGroFile(str(gro))
    kwargs = dict(periodicBoxVectors=coordinates.getPeriodicBoxVectors())
    if include_dir is not None:
        kwargs["includeDir"] = str(include_dir)
    topology = app.GromacsTopFile(str(top), **kwargs)
    system = topology.createSystem(**_system_kwargs(cutoff, hydrogen_mass))
    return topology.topology, system, _nm(coordinates.positions)


def load_charmm(psf, coordinates, parameter_files, box_nm=None, cutoff: float = 1.2,
                switch: float = 1.0, hydrogen_mass=1.5):
    """Load a CHARMM PSF (e.g. from CHARMM-GUI) with a PDB or CRD and its
    parameter files. ``box_nm`` (three lengths) is needed if the coordinate
    file does not carry the box. Returns (topology, system, positions)."""
    psf_file = app.CharmmPsfFile(str(psf))
    path = Path(coordinates)
    if path.suffix.lower() == ".crd":
        positions = _nm(app.CharmmCrdFile(str(path)).positions)
        box = None
    else:
        pdb = app.PDBFile(str(path))
        positions, box = _nm(pdb.positions), pdb.topology.getUnitCellDimensions()
    if box_nm is not None:
        lengths = [float(v) for v in box_nm]
    elif box is not None:
        lengths = list(box.value_in_unit(unit.nanometer))
    else:
        raise ValueError("no periodic box found; pass box_nm=(x, y, z)")
    psf_file.setBox(*(length * unit.nanometer for length in lengths))
    parameters = app.CharmmParameterSet(*(str(p) for p in parameter_files))
    kwargs = _system_kwargs(cutoff, hydrogen_mass)
    kwargs["switchDistance"] = switch * unit.nanometer
    system = psf_file.createSystem(parameters, **kwargs)
    return psf_file.topology, system, positions
