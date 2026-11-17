"""The metadynamics bias as a 2D grid over (s, rho). Pure numpy.

``values[j, i]`` is the bias at ``(s[i], rho[j])`` so that ``values.ravel()``
is in the order OpenMM's 2D tabulated functions expect (first variable
fastest).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from ..funnel.geometry import KB, Funnel


@dataclass
class BiasSettings:
    """Well-tempered metadynamics settings (nm, kJ/mol, K)."""

    temperature: float = 300.0
    bias_factor: Optional[float] = 10.0   # None: plain (non-tempered) metadynamics
    height: float = 1.5                   # initial hill height
    sigma_s: float = 0.05
    sigma_rho: float = 0.1
    pace: int = 250                       # MD steps between hills
    points_per_sigma: float = 3.0         # grid resolution (see note below)
    grid_margin: float = 0.3              # grid extends this far past the walls
    reflect: bool = True                  # mirror hills at rho = 0
    s_min: Optional[float] = None         # explicit grid limits override the margin rule
    s_max: Optional[float] = None
    rho_max: Optional[float] = None

    # Grid resolution: the cubic spline through the grid reproduces a Gaussian to about
    # 2e-4 of its height at 3 points per sigma. A finer grid only costs time: the spline
    # is refitted and sent to the GPU after every hill.

    def __post_init__(self):
        if self.bias_factor is not None and self.bias_factor <= 1.0:
            raise ValueError("bias_factor must be > 1 (or None for plain metadynamics)")
        if min(self.height, self.sigma_s, self.sigma_rho, self.temperature) <= 0:
            raise ValueError("height, sigmas and temperature must be positive")
        if self.pace < 1:
            raise ValueError("pace must be at least 1 step")

    @property
    def kT(self) -> float:
        return KB * self.temperature

    @property
    def delta_kT(self) -> Optional[float]:
        """kB * (bias_factor - 1) * T, the well-tempered energy scale."""
        return None if self.bias_factor is None else self.kT * (self.bias_factor - 1.0)

    @property
    def fes_scale(self) -> float:
        """Factor s.t. F = -fes_scale * V."""
        return 1.0 if self.bias_factor is None else self.bias_factor / (self.bias_factor - 1.0)

    def to_dict(self) -> dict:
        return asdict(self)


class BiasGrid:
    def __init__(self, s_min: float, s_max: float, n_s: int, rho_max: float, n_rho: int,
                 sigma_s: float, sigma_rho: float, reflect: bool = True):
        if n_s < 4 or n_rho < 4:
            raise ValueError("the bias grid needs at least 4 points per axis")
        self.s_min, self.s_max, self.rho_max = float(s_min), float(s_max), float(rho_max)
        self.sigma_s, self.sigma_rho = float(sigma_s), float(sigma_rho)
        self.reflect = bool(reflect)
        self.s = np.linspace(self.s_min, self.s_max, int(n_s))
        self.rho = np.linspace(0.0, self.rho_max, int(n_rho))
        self.values = np.zeros((int(n_rho), int(n_s)))

    @classmethod
    def from_funnel(cls, funnel: Funnel, settings: BiasSettings) -> "BiasGrid":
        s_min = settings.s_min if settings.s_min is not None else funnel.lower - settings.grid_margin
        s_max = settings.s_max if settings.s_max is not None else funnel.upper + settings.grid_margin
        rho_max = (settings.rho_max if settings.rho_max is not None
                   else funnel.max_radius + settings.grid_margin)
        n_s = math.ceil((s_max - s_min) * settings.points_per_sigma / settings.sigma_s) + 1
        n_rho = math.ceil(rho_max * settings.points_per_sigma / settings.sigma_rho) + 1
        return cls(s_min, s_max, n_s, rho_max, n_rho, settings.sigma_s, settings.sigma_rho,
                   settings.reflect)

    @property
    def n_s(self) -> int:
        return self.s.size

    @property
    def n_rho(self) -> int:
        return self.rho.size

    def spec(self) -> dict:
        return dict(s_min=self.s_min, s_max=self.s_max, n_s=self.n_s, rho_max=self.rho_max,
                    n_rho=self.n_rho, sigma_s=self.sigma_s, sigma_rho=self.sigma_rho,
                    reflect=self.reflect)

    def empty_copy(self) -> "BiasGrid":
        return BiasGrid(**self.spec())

    def _kernels(self, s, rho):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        rho = np.atleast_1d(np.asarray(rho, dtype=float))
        g_s = np.exp(-0.5 * ((self.s[None, :] - s[:, None]) / self.sigma_s) ** 2)
        g_rho = np.exp(-0.5 * ((self.rho[None, :] - rho[:, None]) / self.sigma_rho) ** 2)
        if self.reflect:
            g_rho += np.exp(-0.5 * ((self.rho[None, :] + rho[:, None]) / self.sigma_rho) ** 2)
        return g_s, g_rho

    def add_hill(self, s: float, rho: float, height: float) -> None:
        g_s, g_rho = self._kernels(s, rho)
        self.values += height * np.outer(g_rho[0], g_s[0])

    def add_hills(self, s, rho, height, chunk: int = 2000) -> None:
        """Add many hills at once (used when replaying a HILLS file)."""
        s, rho, height = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (s, rho, height))
        for start in range(0, s.size, chunk):
            sl = slice(start, start + chunk)
            g_s, g_rho = self._kernels(s[sl], rho[sl])
            self.values += (g_rho * height[sl, None]).T @ g_s

    def interpolate(self, s, rho) -> np.ndarray:
        """Bilinear interpolation with the same edge clamping as the OpenMM force."""
        s = np.clip(np.asarray(s, dtype=float), self.s_min, self.s_max)
        rho = np.clip(np.abs(np.asarray(rho, dtype=float)), 0.0, self.rho_max)
        x = (s - self.s_min) / (self.s_max - self.s_min) * (self.n_s - 1)
        y = rho / self.rho_max * (self.n_rho - 1)
        i = np.clip(np.floor(x).astype(int), 0, self.n_s - 2)
        j = np.clip(np.floor(y).astype(int), 0, self.n_rho - 2)
        fx, fy = x - i, y - j
        v = self.values
        return ((1 - fy) * ((1 - fx) * v[j, i] + fx * v[j, i + 1])
                + fy * ((1 - fx) * v[j + 1, i] + fx * v[j + 1, i + 1]))
