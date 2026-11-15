"""Free-energy surfaces from the accumulated bias."""

from __future__ import annotations

import numpy as np

from ..engine.grid import BiasGrid


def free_energy_surface(grid: BiasGrid, fes_scale: float) -> np.ndarray:
    """F(s, rho) = -fes_scale * V, shifted so its minimum is zero. Shape (n_rho, n_s)."""
    surface = -fes_scale * grid.values
    return surface - surface.min()


def projection_profile(grid: BiasGrid, fes_scale: float, kT: float) -> np.ndarray:
    """W(s) = -kT ln int exp(-F(s, rho)/kT) drho, shifted so its minimum is zero.

    No extra Jacobian is applied: rho is itself a biased CV, so F(s, rho)
    already contains the 2*pi*rho measure. With mirrored hills the grid still
    covers rho >= 0 only, which is the physical domain.
    """
    exponent = fes_scale * grid.values / kT          # = -F/kT up to a constant
    shift = exponent.max()
    integral = np.trapezoid(np.exp(exponent - shift), grid.rho, axis=0)
    profile = -kT * (np.log(integral) + shift)
    return profile - profile.min()


def reweighted_profile(s_samples, rho_samples, grid: BiasGrid, kT: float, s_edges) -> np.ndarray:
    """W(s) from samples reweighted with the final bias, exp(+V/kT).

    This treats the final bias as static, so pass samples from the late part of
    the run only. Bins without samples are returned as ``inf``.
    """
    bias = grid.interpolate(s_samples, rho_samples)
    weights = np.exp((bias - bias.max()) / kT)
    histogram, _ = np.histogram(s_samples, bins=s_edges, weights=weights)
    with np.errstate(divide="ignore"):
        profile = -kT * np.log(histogram / np.diff(s_edges))
    finite = np.isfinite(profile)
    if finite.any():
        profile = profile - profile[finite].min()
    return profile
