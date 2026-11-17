"""What the ligand can bump into: solute atoms and, optionally, a membrane slab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class MembraneSlab:
    """A membrane that is not built yet, as an impenetrable slab normal to z (nm)."""

    center_z: float = 0.0
    half_thickness: float = 2.2


class Obstacles:
    """Distance from arbitrary points to the nearest obstacle."""

    def __init__(self, coordinates: np.ndarray, slab: Optional[MembraneSlab] = None):
        self.coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 3)
        self.slab = slab
        self._tree = cKDTree(self.coordinates) if len(self.coordinates) else None

    def clearance(self, points: np.ndarray) -> np.ndarray:
        """Distance (nm) to the nearest obstacle for each point; 0 inside the slab."""
        points = np.asarray(points, dtype=float)
        flat = points.reshape(-1, 3)
        if self._tree is not None:
            distance, _ = self._tree.query(flat)
        else:
            distance = np.full(len(flat), np.inf)
        if self.slab is not None:
            to_slab = np.abs(flat[:, 2] - self.slab.center_z) - self.slab.half_thickness
            distance = np.minimum(distance, np.maximum(to_slab, 0.0))
        return distance.reshape(points.shape[:-1])

    def nearest_image_distance(self, points: np.ndarray, box: np.ndarray) -> float:
        """Smallest distance from ``points`` to any obstacle atom in the 26
        neighbouring periodic images. ``box`` holds three lattice vectors (rows)."""
        if self._tree is None:
            return float("inf")
        box = np.asarray(box, dtype=float)
        if box.ndim == 1:
            box = np.diag(box)
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        best = np.inf
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    if i == j == k == 0:
                        continue
                    shift = i * box[0] + j * box[1] + k * box[2]
                    distance, _ = self._tree.query(points - shift)
                    best = min(best, float(distance.min()))
        return best
