"""Free-energy surfaces from the accumulated bias."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

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


def sliced_log_weights(reduced_bias: np.ndarray, counts: np.ndarray, *, fast: bool = False,
                       evaluation_bias: np.ndarray | None = None) -> np.ndarray:
    """Log-weights that turn samples drawn under several static biases into the unbiased ensemble.

    ``reduced_bias[k, i]`` is bias ``k`` at sample ``i`` over kT; ``counts[k]`` is the number of
    samples drawn under bias ``k``. The free energies ``f_k`` of the biased ensembles solve

        exp(-f_k) = sum_i exp(-V_k(x_i)/kT) / sum_l N_l exp(f_l - V_l(x_i)/kT)

    (the multistate Bennett acceptance ratio equations: Shirts and Chodera, J. Chem. Phys. 129,
    124105, 2008), here as the minimum of a convex function of ``f``. The weight of sample ``i``
    is then ``1 / sum_l N_l exp(f_l - V_l(x_i)/kT)``. Only the samples enter: the offsets come
    from them and not from a model of how the bias grew. Raises ``ArithmeticError`` when the
    equations are not solved. An optional matrix of biases at other coordinates evaluates
    weights there with the same fitted offsets. Those coordinates do not enter the fit.
    """
    reduced_bias = np.asarray(reduced_bias, dtype=float)
    counts = np.asarray(counts, dtype=float)
    log_counts = np.log(counts)
    fraction = counts / counts.sum()

    def log_objective(f):
        terms = (log_counts + f)[:, None] - reduced_bias
        total = logsumexp(terms, axis=0)
        return total.mean() - fraction @ f, np.exp(terms - total).mean(axis=1) - fraction

    objective = log_objective
    # A campaign fits hundreds of time slices at many historical cuts. When rescaling every
    # column keeps *all* kernels representable, the same objective and gradient require only
    # two matrix-vector products per iteration. Very large bias differences use logsumexp.
    if fast and np.max(np.ptp(reduced_bias, axis=0)) < 500:
        floor = reduced_bias.min(axis=0)
        kernel = np.exp(floor - reduced_bias)

        def kernel_objective(f):
            shift = f.max()
            coefficients = counts * np.exp(f - shift)
            totals = coefficients @ kernel
            value = np.log(totals).mean() + shift - fraction @ f
            gradient = coefficients * (kernel @ (1.0 / totals)) / totals.size - fraction
            return value, gradient

        objective = kernel_objective

    found = minimize(objective, np.zeros(counts.size), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=5000, maxfun=20000, gtol=1e-10, ftol=1e-15))
    residual = float(np.abs(objective(found.x)[1]).max())
    if not np.isfinite(residual) or residual > 1e-6:
        raise ArithmeticError(f"the reweighting equations were not solved (residual {residual:.1e})")
    evaluate = reduced_bias if evaluation_bias is None else evaluation_bias
    return -logsumexp((log_counts + found.x)[:, None] - evaluate, axis=0)


def reweighted_profile(s_samples, log_weights, kT: float, s_edges) -> np.ndarray:
    """W(s) from a weighted histogram of samples, shifted so its minimum is zero.

    The weights of the samples of a metadynamics run come from :func:`sliced_log_weights`.
    Bins without samples are returned as ``inf``. The sum of a bin is taken relative to its own
    largest weight: the weights in a deep site and in the unbound region differ by more than the
    digits of a float, and a histogram of the weights themselves returned the lighter bins as
    empty.
    """
    s_samples = np.asarray(s_samples, dtype=float)
    log_weights = np.asarray(log_weights, dtype=float)
    s_edges = np.asarray(s_edges, dtype=float)
    n_bins = s_edges.size - 1
    # The bins of numpy.histogram: half-open, the last one closed.
    index = np.searchsorted(s_edges, s_samples, side="right") - 1
    index[s_samples == s_edges[-1]] = n_bins - 1
    inside = (index >= 0) & (index < n_bins) & (log_weights > -np.inf)
    index, log_weights = index[inside], log_weights[inside]
    peak = np.full(n_bins, -np.inf)
    np.maximum.at(peak, index, log_weights)
    total = np.bincount(index, weights=np.exp(log_weights - peak[index]), minlength=n_bins)
    with np.errstate(divide="ignore"):
        profile = -kT * (np.log(total) + peak - np.log(np.diff(s_edges)))
    finite = np.isfinite(profile)
    if finite.any():
        profile = profile - profile[finite].min()
    return profile
