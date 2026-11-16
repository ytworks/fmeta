"""Benchmark sets (FM-D17): the ligands of one target through the ordinary stages, and their
binding free energies beside experiment and FEP.

A set is a directory with ``set.yaml`` (where the structures come from, pinned by commit and
sha256; the ligands; the differences from the default configuration), ``reference.csv`` (the
experimental and the FEP values per ligand) and a ``README.md``. ``fmeta benchmark prepare``
fetches and verifies the inputs and writes one ordinary fmeta work directory per ligand;
``run`` gives every stage of every ligand to one call of the scheduler (FM-D14); ``status``
shows where it stands; ``report`` writes the comparison. A report carries no verdict.
"""

from __future__ import annotations

import warnings

import numpy as np

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20261116
ENERGY_MEASURES = ("mue", "rmse", "mean_signed_error", "mue_offset_removed", "rmse_offset_removed")
MEASURES = ENERGY_MEASURES + ("pearson_r", "kendall_tau")


# ---- statistics -------------------------------------------------------------

def _measures(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``MEASURES`` of arrays of shape ``(..., n)``, as an array of shape ``(..., 7)``.

    A correlation of a series without spread is NaN. Kendall's tau is tau-b (ties are taken
    out of the pairs that count)."""
    error = predicted - reference
    centred = error - error.mean(-1, keepdims=True)
    dp = predicted - predicted.mean(-1, keepdims=True)
    dr = reference - reference.mean(-1, keepdims=True)
    sp = np.sign(predicted[..., :, None] - predicted[..., None, :]).astype(np.int8)
    sr = np.sign(reference[..., :, None] - reference[..., None, :]).astype(np.int8)
    with np.errstate(invalid="ignore", divide="ignore"):
        pearson = (dp * dr).sum(-1) / np.sqrt((dp ** 2).sum(-1) * (dr ** 2).sum(-1))
        pairs = np.count_nonzero(sp, axis=(-1, -2)) * np.count_nonzero(sr, axis=(-1, -2))
        tau = (sp * sr).sum((-1, -2), dtype=np.int64) / np.sqrt(pairs)
    flat = (np.ptp(predicted, axis=-1) == 0) | (np.ptp(reference, axis=-1) == 0)
    pearson = np.where(flat, np.nan, pearson)                 # 0 / 0 up to rounding otherwise
    return np.stack([np.abs(error).mean(-1), np.sqrt((error ** 2).mean(-1)), error.mean(-1),
                     np.abs(centred).mean(-1), np.sqrt((centred ** 2).mean(-1)), pearson, tau], axis=-1)


def agreement(predicted, reference, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    """How ``predicted`` agrees with ``reference``: one value per ligand, both in one unit.

    Returns ``n`` and, for each of ``MEASURES``, ``dict(value, ci95)``. The error of a ligand
    is predicted - reference. ``*_offset_removed`` are the measures after the mean error of
    these same ligands is taken out: what is left when only differences between the ligands
    count, as for relative free energies that were shifted to the experimental mean.

    ``ci95`` is the 2.5 - 97.5 percentile interval of a bootstrap over the ligands (drawn with
    replacement, ``resamples`` times, from ``seed``): it tells how much the measure depends
    on the choice of ligands, not how uncertain each value is. A measure that does not exist
    is None: every interval and both correlations below three ligands, the correlations of a
    series without spread, the offset-removed measures of a single ligand.
    """
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if predicted.ndim != 1 or predicted.shape != reference.shape:
        raise ValueError("agreement needs two series of the same length")
    n = len(predicted)
    result: dict = dict(n=n)
    if n == 0:
        result.update({name: dict(value=None, ci95=None) for name in MEASURES})
        return result
    values = _measures(predicted, reference)
    low = high = np.full(len(MEASURES), np.nan)
    if n >= 3 and resamples > 0:
        rng = np.random.default_rng(seed)
        samples = []
        for start in range(0, resamples, 1000):
            index = rng.integers(0, n, size=(min(1000, resamples - start), n))
            samples.append(_measures(predicted[index], reference[index]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # a measure that no resample has
            low, high = np.nanpercentile(np.concatenate(samples), [2.5, 97.5], axis=0)
    needs = dict(pearson_r=3, kendall_tau=3, mue_offset_removed=2, rmse_offset_removed=2)
    for i, name in enumerate(MEASURES):
        value = float(values[i]) if n >= needs.get(name, 1) and np.isfinite(values[i]) else None
        interval = [float(low[i]), float(high[i])] if value is not None and np.isfinite(low[i]) else None
        result[name] = dict(value=value, ci95=interval)
    return result
