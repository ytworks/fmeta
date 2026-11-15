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


def reweighting_offset(grid: BiasGrid, fes_scale: float, kT: float) -> float:
    """The time-dependent offset c(t) of well-tempered metadynamics for the bias
    currently on ``grid`` (Tiwary and Parrinello, J. Phys. Chem. B 2015):

        c = kT ln [ int exp(-F/kT) / int exp(-(F + V)/kT) ],   F = -fes_scale * V

    For plain metadynamics (fes_scale = 1) the denominator is flat, as it should be.
    """
    bias = grid.values / kT
    top = fes_scale * bias
    bottom = (fes_scale - 1.0) * bias
    log_top = top.max() + np.log(np.exp(top - top.max()).sum())
    log_bottom = bottom.max() + np.log(np.exp(bottom - bottom.max()).sum())
    return float(kT * (log_top - log_bottom))


def reweighted_profile(s_samples, log_weights, kT: float, s_edges) -> np.ndarray:
    """W(s) from a weighted histogram of samples, shifted so its minimum is zero.

    For metadynamics the weight of a sample taken at time t is
    exp((V(s, rho, t) - c(t))/kT), with V the bias the system felt at that
    moment and c(t) from :func:`reweighting_offset`. Bins without samples are
    returned as ``inf``.
    """
    log_weights = np.asarray(log_weights, dtype=float)
    weights = np.exp(log_weights - log_weights.max())
    histogram, _ = np.histogram(s_samples, bins=s_edges, weights=weights)
    with np.errstate(divide="ignore"):
        profile = -kT * np.log(histogram / np.diff(s_edges))
    finite = np.isfinite(profile)
    if finite.any():
        profile = profile - profile[finite].min()
    return profile
