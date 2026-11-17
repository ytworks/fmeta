"""Funnel definition and the pure-numpy geometry that goes with it.

The funnel is defined by three atom groups: the ligand (L), an origin inside the
protein behind the pocket (p0) and a second protein group (p1) that fixes the
axis direction p0 -> p1. Two collective variables follow:

    s   (projection): position of the ligand centroid along the axis, from p0
    rho (extent):     distance of the ligand centroid from the axis

The side wall has radius ``R(s) = width / (1 + exp(steepness*(s - inflection))) + buffer``,
a cone that narrows into a thin cylinder in bulk solvent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np
import yaml

KB = 0.0083144626  # Boltzmann constant, kJ/mol/K
STANDARD_VOLUME = 1.6605390  # nm^3 per molecule at 1 mol/L


@dataclass
class Funnel:
    """A funnel restraint and its collective variables.

    Lengths are in nm, force constants in kJ/mol/nm^2, ``steepness`` in 1/nm.
    Group weights are always explicit so the CVs do not depend on particle
    masses (which hydrogen mass repartitioning changes).
    """

    ligand_atoms: list[int]
    p0_atoms: list[int]
    p1_atoms: list[int]
    ligand_weights: Optional[list[float]] = None
    p0_weights: Optional[list[float]] = None
    p1_weights: Optional[list[float]] = None
    width: float = 0.6
    buffer: float = 0.15
    steepness: float = 1.5
    inflection: float = 2.0
    lower: float = 0.5
    upper: float = 4.0
    k_wall: float = 10000.0
    k_lower: float = 10000.0
    k_upper: float = 10000.0
    # Regularisation of rho at the axis: rho_reg = sqrt(rho^2 + eps^2).
    eps: float = 1.0e-3
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("ligand", "p0", "p1"):
            atoms = [int(i) for i in getattr(self, f"{name}_atoms")]
            if not atoms:
                raise ValueError(f"{name}_atoms is empty")
            weights = getattr(self, f"{name}_weights")
            weights = [1.0] * len(atoms) if weights is None else [float(w) for w in weights]
            if len(weights) != len(atoms):
                raise ValueError(f"{name}_weights and {name}_atoms differ in length")
            if min(weights) < 0 or sum(weights) <= 0:
                raise ValueError(f"{name}_weights must be non-negative with a positive sum")
            setattr(self, f"{name}_atoms", atoms)
            setattr(self, f"{name}_weights", weights)
        # Plain floats only: these values are formatted into OpenMM expressions.
        for name in ("width", "buffer", "steepness", "inflection", "lower", "upper",
                     "k_wall", "k_lower", "k_upper", "eps"):
            setattr(self, name, float(getattr(self, name)))
        if not self.lower < self.upper:
            raise ValueError("lower wall must be below the upper wall")
        if min(self.width, self.buffer, self.steepness, self.eps) < 0:
            raise ValueError("width, buffer, steepness and eps must be non-negative")
        if min(self.k_wall, self.k_lower, self.k_upper) <= 0:
            raise ValueError("force constants must be positive")

    # ---- geometry -------------------------------------------------------
    def wall_radius(self, s):
        """Side-wall radius R(s) in nm."""
        s = np.asarray(s, dtype=float)
        return self.width / (1.0 + np.exp(self.steepness * (s - self.inflection))) + self.buffer

    def restraint_energy(self, s, rho):
        """Energy of the side, lower and upper walls (kJ/mol)."""
        s = np.asarray(s, dtype=float)
        rho = np.asarray(rho, dtype=float)
        side = np.maximum(rho - self.wall_radius(s), 0.0)
        low = np.maximum(self.lower - s, 0.0)
        up = np.maximum(s - self.upper, 0.0)
        return 0.5 * (self.k_wall * side**2 + self.k_lower * low**2 + self.k_upper * up**2)

    def effective_area(self, s, kT):
        """Cross-section available to the ligand centroid at projection ``s`` (nm^2).

        This is ``int 2*pi*rho*exp(-U_wall/kT) drho``: the hard-wall disc plus
        the thermally populated skin of the harmonic wall.
        """
        radius = self.wall_radius(s)
        skin = np.sqrt(np.pi * kT / (2.0 * self.k_wall))
        return np.pi * radius**2 + 2.0 * np.pi * radius * skin + 2.0 * np.pi * kT / self.k_wall

    @property
    def max_radius(self) -> float:
        """Largest wall radius inside [lower, upper]."""
        return float(self.wall_radius(self.lower))

    # ---- (de)serialisation ----------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Funnel":
        return cls(**data)

    def save(self, path) -> None:
        with open(path, "w") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, default_flow_style=None)

    @classmethod
    def load(cls, path) -> "Funnel":
        with open(path) as handle:
            return cls.from_dict(yaml.safe_load(handle))


def centroid(positions: np.ndarray, atoms: Sequence[int], weights: Sequence[float]) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    return (positions[list(atoms)] * weights[:, None]).sum(axis=0) / weights.sum()


def funnel_frame(funnel: Funnel, positions: np.ndarray):
    """Return (origin, unit axis, ligand centroid) for a set of coordinates."""
    positions = np.asarray(positions, dtype=float)
    p0 = centroid(positions, funnel.p0_atoms, funnel.p0_weights)
    p1 = centroid(positions, funnel.p1_atoms, funnel.p1_weights)
    lig = centroid(positions, funnel.ligand_atoms, funnel.ligand_weights)
    axis = p1 - p0
    return p0, axis / np.linalg.norm(axis), lig


def compute_cvs(funnel: Funnel, positions: np.ndarray, regularize: bool = True):
    """Numpy reference for the two collective variables: returns ``(s, rho)``.

    The quantities of the OpenMM expressions, from the same two vectors p0 -> ligand and
    p0 -> p1: the component of the first along the second, and the length of its component
    across it. Like the forces, it takes the coordinates as they are, without periodic images.
    """
    p0, axis, lig = funnel_frame(funnel, positions)
    to_ligand = lig - p0
    rho_sq = float(np.sum(np.cross(to_ligand, axis) ** 2))
    if regularize:
        rho_sq += funnel.eps**2
    return float(np.dot(to_ligand, axis)), float(np.sqrt(rho_sq))
