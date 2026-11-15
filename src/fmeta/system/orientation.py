"""Where does the membrane go? A hydrophobic-belt fit from alpha carbons.

``Modeller.addMembrane`` puts the bilayer in the xy plane and trusts that the
protein is already oriented. This module estimates the membrane plane from the
structure itself, to check an input orientation or to produce one.

The fit follows the idea of slab-search methods such as ANVIL: among the
surface residues, the lipid-facing ones are hydrophobic and the ones in water
are polar. A slab of fixed thickness is swept over all orientations and
positions, and the one whose inside/outside split correlates best with residue
hydrophobicity wins. It is a coarse estimate (tilt to roughly ten degrees,
position to a few angstroms) and it is not a substitute for a curated
orientation such as the OPM database provides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openmm.app as app
from scipy.spatial import cKDTree

from ..funnel.autodetect import fibonacci_sphere

# Normalised consensus hydrophobicity scale of Eisenberg et al. (1984).
HYDROPHOBICITY = {
    "ILE": 1.38, "PHE": 1.19, "VAL": 1.08, "LEU": 1.06, "TRP": 0.81, "MET": 0.64, "ALA": 0.62,
    "GLY": 0.48, "CYS": 0.29, "TYR": 0.26, "PRO": 0.12, "THR": -0.05, "SER": -0.18, "HIS": -0.40,
    "GLU": -0.74, "ASN": -0.78, "GLN": -0.85, "ASP": -0.90, "LYS": -1.50, "ARG": -2.53,
}
_ALIASES = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
            "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS", "MSE": "MET"}

MIN_SCORE = 0.35       # correlation below which no belt is claimed
MIN_CONTRAST = 4.0     # ... and by how many standard deviations it must beat shuffled sequences


@dataclass
class MembraneFit:
    normal: np.ndarray        # unit vector, pointing to +z half space
    midpoint: np.ndarray      # point on the fitted midplane closest to the protein centroid (nm)
    score: float              # correlation between hydrophobicity and being inside the slab
    contrast: float           # z-score of the fit against the same structure with shuffled residues
    n_residues: int           # surface residues the fit is based on

    @property
    def confident(self) -> bool:
        return self.score >= MIN_SCORE and self.contrast >= MIN_CONTRAST

    def tilt(self) -> float:
        """Angle (degrees) between the fitted normal and the z axis."""
        return float(np.degrees(np.arccos(min(1.0, abs(self.normal[2])))))

    def offset(self, center_z: float = 0.0) -> float:
        """Distance (nm) from the fitted midplane to the plane z = center_z,
        measured under the protein centroid."""
        return float(self.midpoint[2] - center_z)


def _scores(projection: np.ndarray, values: np.ndarray, centres: np.ndarray,
            half_thickness: float) -> np.ndarray:
    """Correlation between ``values`` and slab membership, for each slab centre."""
    inside = (np.abs(projection[None, :] - centres[:, None]) <= half_thickness).astype(float)
    inside -= inside.mean(axis=1, keepdims=True)
    norm = np.sqrt((inside**2).sum(axis=1)) * np.sqrt((values**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        score = (inside @ values) / norm
    return np.nan_to_num(score, nan=-1.0)


def fit_slab(ca_positions: np.ndarray, residue_names, half_thickness: float = 1.5,
             n_directions: int = 2000) -> MembraneFit:
    """Fit a hydrophobic slab to alpha-carbon positions (nm) and residue names."""
    ca_positions = np.asarray(ca_positions, dtype=float)
    names = [_ALIASES.get(n.strip().upper(), n.strip().upper()) for n in residue_names]
    known = np.array([n in HYDROPHOBICITY for n in names])
    xyz = ca_positions[known]
    hydrophobicity = np.array([HYDROPHOBICITY[n] for n, k in zip(names, known) if k])
    if len(xyz) < 30:
        raise ValueError("too few residues to estimate a membrane orientation")

    # surface residues: the less crowded 60% by alpha-carbon neighbour count
    crowding = cKDTree(xyz).query_ball_point(xyz, 1.0, return_length=True)
    surface = crowding <= np.percentile(crowding, 60)
    xyz_s = xyz[surface]
    values = hydrophobicity[surface] - hydrophobicity[surface].mean()
    centroid = xyz.mean(axis=0)

    def best_for(direction: np.ndarray, step: float):
        projection = (xyz_s - centroid) @ direction
        centres = np.arange(projection.min(), projection.max() + step, step)
        score = _scores(projection, values, centres, half_thickness)
        index = int(np.argmax(score))
        return float(score[index]), float(centres[index])

    directions = fibonacci_sphere(n_directions)
    directions = directions[directions[:, 2] >= 0.0]          # a slab has no up or down
    coarse = np.array([best_for(d, 0.1) for d in directions])
    best = int(np.argmax(coarse[:, 0]))

    # How good a slab can be found by chance in this shape? Shuffle which residue
    # sits where and repeat the search (on fewer directions, compared like for like).
    rng = np.random.default_rng(0)
    few = fibonacci_sphere(400)
    few = few[few[:, 2] >= 0.0]
    projections = (xyz_s - centroid) @ few.T                   # (residues, directions)

    def search(sample: np.ndarray) -> float:
        best_score = -1.0
        for column in range(few.shape[0]):
            projection = projections[:, column]
            centres = np.arange(projection.min(), projection.max() + 0.2, 0.2)
            best_score = max(best_score, float(_scores(projection, sample, centres,
                                                       half_thickness).max()))
        return best_score

    shuffled = np.array([search(rng.permutation(values)) for _ in range(24)])
    contrast = float((search(values) - shuffled.mean()) / max(shuffled.std(), 1e-9))

    # refine around the best direction
    normal, (score, centre) = directions[best], coarse[best]
    for spread in (0.10, 0.03):
        candidates = normal + spread * rng.normal(size=(200, 3))
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        results = np.array([best_for(d, 0.05) for d in candidates])
        top = int(np.argmax(results[:, 0]))
        if results[top, 0] > score:
            normal, score, centre = candidates[top], float(results[top, 0]), float(results[top, 1])
    if normal[2] < 0:
        normal, centre = -normal, -centre
    return MembraneFit(normal, centroid + centre * normal, float(score), contrast, int(surface.sum()))


def fit_membrane(topology: app.Topology, positions: np.ndarray,
                 half_thickness: float = 1.5) -> MembraneFit:
    """Fit the membrane plane to the protein in ``topology``."""
    atoms = [a for a in topology.atoms() if a.name == "CA" and a.residue.name.strip().upper()
             in set(HYDROPHOBICITY) | set(_ALIASES)]
    positions = np.asarray(positions, dtype=float)
    return fit_slab(positions[[a.index for a in atoms]], [a.residue.name for a in atoms],
                    half_thickness)


def rotation_onto_z(normal: np.ndarray) -> np.ndarray:
    """Rotation matrix R with R @ normal = z (the smallest such rotation)."""
    normal = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(normal, z), float(normal @ z)
    if c < -1.0 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


def orient(positions: np.ndarray, fit: MembraneFit, center_z: float = 0.0):
    """Rotate and translate coordinates so the fitted membrane is the plane
    z = center_z, with the midplane point under the protein at x = y = 0.

    Returns ``(new_positions, rotation, translation)`` with
    ``new = old @ rotation.T + translation``.
    """
    rotation = rotation_onto_z(fit.normal)
    translation = np.array([0.0, 0.0, center_z]) - rotation @ fit.midpoint
    return np.asarray(positions, dtype=float) @ rotation.T + translation, rotation, translation
