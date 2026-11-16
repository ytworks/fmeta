"""Benchmark sets (FM-D17): the statistics of a report, the set definition, the order of the
jobs and the restart, the report and its check."""

import numpy as np
import pytest
from scipy import stats

from fmeta.benchmark import agreement

# ---- statistics -------------------------------------------------------------


def test_a_constant_offset_is_the_whole_error_and_vanishes_when_the_offset_is_removed():
    reference = np.array([-40.0, -35.0, -30.0, -25.0])
    result = agreement(reference + 1.0, reference)
    assert result["n"] == 4
    for name in ("mue", "rmse", "mean_signed_error"):
        assert result[name] == dict(value=1.0, ci95=[1.0, 1.0])       # every resample has it too
    for name in ("mue_offset_removed", "rmse_offset_removed"):
        assert result[name] == dict(value=0.0, ci95=[0.0, 0.0])
    assert result["pearson_r"]["value"] == pytest.approx(1.0, abs=1e-12)
    assert result["kendall_tau"]["value"] == 1.0


def test_errors_of_both_signs_give_the_known_measures():
    reference = np.array([-40.0, -35.0, -30.0, -25.0])
    result = agreement(reference + np.array([1.0, -1.0, 2.0, -2.0]), reference)
    assert result["mue"]["value"] == pytest.approx(1.5)
    assert result["rmse"]["value"] == pytest.approx(np.sqrt(2.5))
    assert result["mean_signed_error"]["value"] == pytest.approx(0.0, abs=1e-12)
    assert result["mue_offset_removed"]["value"] == pytest.approx(1.5)
    assert result["rmse_offset_removed"]["value"] == pytest.approx(np.sqrt(2.5))


def test_the_correlations_are_those_of_scipy_also_with_ties():
    reference = np.array([-45.0, -42.0, -42.0, -40.0, -38.0, -38.0, -38.0, -33.0, -30.0, -30.0])
    predicted = np.array([-44.0, -40.0, -43.0, -40.0, -35.0, -39.0, -36.0, -36.0, -31.0, -28.0])
    result = agreement(predicted, reference, resamples=0)
    assert result["pearson_r"]["value"] == pytest.approx(stats.pearsonr(predicted, reference)[0], abs=1e-12)
    assert result["kendall_tau"]["value"] == pytest.approx(stats.kendalltau(predicted, reference)[0], abs=1e-12)
    assert result["kendall_tau"]["ci95"] is None                      # no resamples, no interval


def test_the_bootstrap_interval_is_that_of_the_mean_of_the_errors():
    """40 errors of mean 0.5 and standard deviation 1: the 95% interval of their mean is the
    sample mean -+ 1.96 s / sqrt(40), to the precision of 10000 resamples."""
    rng = np.random.default_rng(7)
    reference = rng.uniform(-50.0, -25.0, 40)
    error = rng.normal(0.5, 1.0, 40)
    result = agreement(reference + error, reference)["mean_signed_error"]
    half = 1.96 * error.std() / np.sqrt(40)
    assert result["value"] == pytest.approx(error.mean())
    assert result["ci95"] == pytest.approx([error.mean() - half, error.mean() + half], abs=0.03)
    assert result == agreement(reference + error, reference)["mean_signed_error"]     # fixed seed


def test_measures_that_do_not_exist_are_none():
    two = agreement([-30.0, -40.0], [-31.0, -38.0])
    assert two["mue"] == dict(value=1.5, ci95=None)                   # no interval below three ligands
    assert two["pearson_r"] == two["kendall_tau"] == dict(value=None, ci95=None)
    flat = agreement([-30.0, -30.0, -30.0], [-31.0, -38.0, -35.0])
    assert flat["pearson_r"]["value"] is None and flat["kendall_tau"]["value"] is None
    assert flat["mue"]["ci95"] is not None
    assert agreement([], [])["mue"] == dict(value=None, ci95=None)
