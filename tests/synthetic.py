"""Synthetic "proteins" for funnel-placement tests: lattices of atoms with a
binding cavity and channels carved out, so the right answer is known."""

import numpy as np
import openmm.app as app

SPACING = 0.27   # nm; tight enough that a 0.3 nm probe cannot pass between atoms


def lattice(shape_test, extent, rng):
    axis = np.arange(-extent, extent + SPACING / 2, SPACING)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    points = points[shape_test(points)]
    return points + rng.uniform(-0.02, 0.02, size=points.shape)


def carve(points, site, channels, cavity_radius=0.5):
    """Remove a cavity around ``site`` and straight channels (direction, radius) from it."""
    keep = np.linalg.norm(points - site, axis=1) > cavity_radius
    for direction, radius in channels:
        direction = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        relative = points - site
        along = relative @ direction
        off_axis = np.linalg.norm(relative - np.outer(along, direction), axis=1)
        keep &= ~((along > 0) & (off_axis < radius))
    return points[keep]


def build_topology(protein_points, site, rng, ca_fraction=0.2, n_ligand=8):
    """Protein atoms as 5-atom ALA residues (some named CA) plus an 8-atom ligand at ``site``."""
    topology = app.Topology()
    chain = topology.addChain("A")
    carbon = app.Element.getBySymbol("C")
    is_ca = rng.random(len(protein_points)) < ca_fraction
    residue = None
    for index in range(len(protein_points)):
        if index % 5 == 0:
            residue = topology.addResidue("ALA", chain)
        topology.addAtom("CA" if is_ca[index] else f"C{index % 5}", carbon, residue)
    ligand_chain = topology.addChain("L")
    ligand_residue = topology.addResidue("LIG", ligand_chain)
    ligand_points = site + rng.normal(scale=0.1, size=(n_ligand, 3))
    for index in range(n_ligand):
        topology.addAtom(f"C{index}", carbon, ligand_residue)
    return topology, np.vstack([protein_points, ligand_points])


def ball_with_channel(direction, seed=0, radius=1.8, depth=1.2, channel_radius=0.5,
                      channel=True):
    """A ball with a cavity ``depth`` nm below the surface and, optionally, a
    straight channel from it to the surface along ``direction``."""
    rng = np.random.default_rng(seed)
    direction = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    site = (radius - depth) * direction
    points = lattice(lambda p: np.linalg.norm(p, axis=1) < radius, radius, rng)
    points = carve(points, site, [(direction, channel_radius)] if channel else [])
    return build_topology(points, site, rng) + (site,)


def membrane_barrel(seed=0, radius=1.8, half_length=3.2, site_z=1.2):
    """A cylinder along z, standing in for a membrane protein. From the site a
    narrow channel leads up to the +z face and a wider, shorter one sideways (+x)
    into what would be the bilayer."""
    rng = np.random.default_rng(seed)
    site = np.array([0.0, 0.0, site_z])
    points = lattice(lambda p: (np.hypot(p[:, 0], p[:, 1]) < radius) & (np.abs(p[:, 2]) < half_length),
                     half_length, rng)
    points = carve(points, site, [([0, 0, 1], 0.42), ([1, 0, 0], 0.6)])
    return build_topology(points, site, rng) + (site,)
