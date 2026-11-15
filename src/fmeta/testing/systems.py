"""A small real system for tests and for ``fmeta check``: the 35-residue protein
shipped with OpenMM (app/data/test.pdb) with a benzene molecule placed against
its surface. The pose is not a real binding mode; it exercises the machinery."""

import os
from pathlib import Path

import numpy as np
import openmm.app as app
import openmm.unit as unit
from scipy.spatial import cKDTree

BENZENE_XML = str(Path(__file__).parent / "data" / "benzene.xml")


def write_protein(path) -> np.ndarray:
    """Write the protein (no solvent) and return its heavy-atom coordinates in nm."""
    source = app.PDBFile(os.path.join(os.path.dirname(app.__file__), "data", "test.pdb"))
    modeller = app.Modeller(source.topology, source.positions)
    modeller.delete([r for r in modeller.topology.residues() if r.name in ("HOH", "Cl")])
    modeller.topology.setPeriodicBoxVectors(None)
    with open(path, "w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle)
    positions = np.array(modeller.positions.value_in_unit(unit.nanometer))
    heavy = [a.index for a in modeller.topology.atoms() if a.element.symbol != "H"]
    return positions[heavy]


def pocket_point(heavy: np.ndarray, clearance: float = 0.5) -> np.ndarray:
    """The most enclosed point that keeps ``clearance`` nm from every heavy atom."""
    tree = cKDTree(heavy)
    lo, hi = heavy.min(axis=0) - 0.5, heavy.max(axis=0) + 0.5
    axes = [np.arange(a, b, 0.1) for a, b in zip(lo, hi)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    distance, _ = tree.query(grid)
    grid = grid[(distance >= clearance) & (distance <= clearance + 0.1)]
    neighbours = np.array([len(n) for n in tree.query_ball_point(grid, 1.0)])
    return grid[np.argmax(neighbours)]


def write_benzene(path, centre: np.ndarray, resname: str = "BNZ") -> None:
    """Benzene as a V2000 molfile, ring centred on ``centre`` (nm) in the xy plane."""
    angles = np.arange(6) * np.pi / 3
    ring = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(6)])
    coords = np.vstack([centre + 0.1397 * ring, centre + 0.2484 * ring]) * 10.0
    lines = [resname, f"{'':2s}{'fmeta':<8s}{'':10s}3D", "", f"{12:3d}{12:3d}  0  0  0  0  0  0  0  0999 V2000"]
    for index, (x, y, z) in enumerate(coords):
        lines.append(f"{x:10.4f}{y:10.4f}{z:10.4f} {'C' if index < 6 else 'H':<3s} 0  0  0  0  0  0  0  0  0  0  0  0")
    for i in range(6):
        lines.append(f"{i + 1:3d}{(i + 1) % 6 + 1:3d}{2 if i % 2 == 0 else 1:3d}  0")
    for i in range(6):
        lines.append(f"{i + 1:3d}{i + 7:3d}{1:3d}  0")
    lines += ["M  END", "$$$$"]
    Path(path).write_text("\n".join(lines) + "\n")


def write_complex_inputs(directory) -> tuple[str, str]:
    """Write protein.pdb and benzene.sdf into ``directory``; return their paths."""
    directory = Path(directory)
    heavy = write_protein(directory / "protein.pdb")
    write_benzene(directory / "benzene.sdf", pocket_point(heavy))
    return str(directory / "protein.pdb"), str(directory / "benzene.sdf")


def write_membrane_inputs(directory, centre_z: float = 1.5) -> tuple[str, str]:
    """The same protein and benzene, turned so the pocket faces +z and lifted so
    the lower half of the protein sits in a bilayer centred on z = 0 while the
    pocket opens to the water above it. Not a real membrane protein: it only
    gives ``addMembrane`` something to pack lipids around."""
    directory = Path(directory)
    source = app.PDBFile(os.path.join(os.path.dirname(app.__file__), "data", "test.pdb"))
    modeller = app.Modeller(source.topology, source.positions)
    modeller.delete([r for r in modeller.topology.residues() if r.name in ("HOH", "Cl")])
    modeller.topology.setPeriodicBoxVectors(None)
    positions = np.array(modeller.positions.value_in_unit(unit.nanometer))
    heavy = np.array([a.index for a in modeller.topology.atoms() if a.element.symbol != "H"])
    pocket = pocket_point(positions[heavy])
    centre = positions[heavy].mean(axis=0)

    # rotation taking (centre -> pocket) onto +z (Rodrigues)
    u = (pocket - centre) / np.linalg.norm(pocket - centre)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(u, z), float(u @ z)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation = np.eye(3) + k + k @ k / (1.0 + c)
    shift = np.array([0.0, 0.0, centre_z])
    positions = (positions - centre) @ rotation.T + shift
    pocket = (pocket - centre) @ rotation.T + shift

    with open(directory / "protein_oriented.pdb", "w") as handle:
        app.PDBFile.writeFile(modeller.topology, positions * unit.nanometer, handle)
    write_benzene(directory / "benzene_oriented.sdf", pocket)
    return str(directory / "protein_oriented.pdb"), str(directory / "benzene_oriented.sdf")
