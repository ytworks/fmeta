"""Undo periodic wrapping in input coordinates.

The funnel CVs take centroids of atom groups, and a centroid is only meaningful
if the group's atoms sit in the same periodic image. Simulation output is often
written with atoms or molecules wrapped into the box, so pre-parameterised
input is made whole before a funnel is placed on it.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import openmm.app as app
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components
from scipy.spatial import cKDTree


def _lattice(box) -> np.ndarray:
    box = np.asarray(box, dtype=float)
    return np.diag(box) if box.ndim == 1 else box


def _wrap(delta: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Remove lattice vectors from displacements (rows), one box vector after the other.

    The result is the nearest image while every component of the true displacement is below
    half of the box width along its axis. A bond always is; a vector as long as the funnel is
    not, in a rhombic dodecahedron (FM-D2 in ``docs/design/engine.md``).
    """
    delta = np.array(delta, dtype=float)
    for axis in (2, 1, 0):
        delta -= np.outer(np.round(delta[:, axis] / lattice[axis, axis]), lattice[axis])
    return delta


def make_whole(topology: app.Topology, positions: np.ndarray, box,
               assemble: Optional[Sequence[int]] = None) -> np.ndarray:
    """Return coordinates in which every molecule is whole.

    Each bonded atom is moved to the periodic image nearest its neighbour,
    walking outwards from one atom per molecule. Then the molecules that contain
    the atoms in ``assemble`` (typically protein chains, ligand and cofactors)
    are translated, largest first, to the image in contact with those already
    placed, so that a complex split over several molecules ends up together.
    """
    positions = np.array(positions, dtype=float)
    lattice = _lattice(box)
    n = topology.getNumAtoms()
    bonds = np.array([(a.index, b.index) for a, b in topology.bonds()], dtype=int).reshape(-1, 2)
    graph = coo_matrix((np.ones(len(bonds)), (bonds[:, 0], bonds[:, 1])), shape=(n, n))
    n_molecules, labels = connected_components(graph, directed=False)

    # One breadth-first walk for all molecules: a virtual root bonded to one atom of each.
    roots = np.unique(labels, return_index=True)[1]
    rows = np.concatenate([bonds[:, 0], np.full(len(roots), n)])
    cols = np.concatenate([bonds[:, 1], roots])
    rooted = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n + 1, n + 1)).tocsr()
    order, parent = breadth_first_order(rooted, n, directed=False)
    depth = np.zeros(n + 1, dtype=int)
    for atom in order[1:]:
        depth[atom] = depth[parent[atom]] + 1
    for level in range(2, depth.max() + 1):          # level 1 are the molecule roots
        atoms = np.flatnonzero(depth[:n] == level)
        delta = positions[atoms] - positions[parent[atoms]]
        positions[atoms] = positions[parent[atoms]] + _wrap(delta, lattice)

    if assemble is not None and len(assemble):
        chosen = np.unique(labels[np.asarray(assemble, dtype=int)])
        members = [np.flatnonzero(labels == label) for label in chosen]
        members.sort(key=len, reverse=True)
        placed = positions[members[0]]
        shifts = [i * lattice[0] + j * lattice[1] + k * lattice[2]
                  for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
        for atoms in members[1:]:
            tree = cKDTree(placed)
            best = min(shifts, key=lambda shift: tree.query(positions[atoms] + shift)[0].min())
            positions[atoms] += best
            placed = np.vstack([placed, positions[atoms]])
    return positions
