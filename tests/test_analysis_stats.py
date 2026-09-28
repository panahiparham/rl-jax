"""Tests for the aggregates over runs (``analysis.stats``)."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.stats import (
    best_by_mean,
    mean_ci,
    median_ti,
    min_max_normalize,
)

# --- mean_ci ------------------------------------------------------------------


def test_mean_ci_mean_matches_constant_return_across_seeds():
    """Identical runs average to their shared value."""
    stack = np.full((5, 4), -4.0)

    mean, _ci_lo, _ci_hi = mean_ci(stack, n_boot=200)

    assert np.allclose(mean, -4)


def test_mean_ci_band_has_zero_width_when_seeds_agree():
    """Resampling identical runs cannot move the mean, so the band collapses."""
    stack = np.full((5, 4), -4.0)

    _mean, ci_lo, ci_hi = mean_ci(stack, n_boot=200)

    assert np.allclose(ci_lo, -4)
    assert np.allclose(ci_hi, -4)



def test_mean_ci_of_per_run_scalars_is_one_interval():
    """One lifetime score per run gives a single mean and band, as for a bar."""
    center, low, high = mean_ci([1.0, 2.0, 6.0], n_boot=200)
    assert center.shape == ()
    assert np.isclose(center, 3.0)
    assert 1.0 <= low <= center <= high <= 6.0


def test_mean_ci_keeps_the_trailing_shape_of_its_samples():
    """Runs are the first axis; every other axis is kept point by point."""
    samples = np.random.default_rng(0).normal(size=(4, 2, 3))
    center, low, high = mean_ci(samples, n_boot=200)
    assert center.shape == low.shape == high.shape == (2, 3)
    assert np.allclose(center, samples.mean(axis=0))


def test_mean_ci_band_widens_with_confidence():
    """A higher confidence level asks for a wider bootstrap band."""
    samples = np.random.default_rng(0).normal(size=20)
    _, low_90, high_90 = mean_ci(samples, confidence=0.9, n_boot=2000)
    _, low_99, high_99 = mean_ci(samples, confidence=0.99, n_boot=2000)
    assert low_99 < low_90 < high_90 < high_99


@pytest.mark.parametrize("aggregate", [mean_ci, median_ti])
def test_aggregating_zero_runs_raises(aggregate):
    """Missing data is skipped before aggregating, never drawn as an empty band."""
    with pytest.raises(ValueError, match="no runs"):
        aggregate(np.empty((0, 3)))

# --- min_max_normalize --------------------------------------------------------


def test_min_max_normalize_shares_bounds_across_arrays():
    """The lowest value anywhere maps to 0 and the highest to 1, so arrays from
    different agents on one environment stay comparable."""
    weak, strong = min_max_normalize([[-20.0, 0.0], [10.0, 20.0]])
    assert np.allclose(weak, [0.0, 0.5])
    assert np.allclose(strong, [0.75, 1.0])


def test_min_max_normalize_keeps_stack_shape_in_unit_range():
    """A seed stack keeps its shape, and every value lands in [0, 1]."""
    stack = np.random.default_rng(0).normal(size=(5, 7)) * 100
    (normalized,) = min_max_normalize([stack])
    assert normalized.shape == stack.shape
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0


def test_min_max_normalize_ignores_nan_for_bounds():
    """NaN gaps in a curve neither set the bounds nor get filled in."""
    (normalized,) = min_max_normalize([[np.nan, 2.0, 4.0, np.nan]])
    assert np.isnan(normalized[[0, 3]]).all()
    assert np.allclose(normalized[1:3], [0.0, 1.0])


def test_min_max_normalize_rejects_no_arrays():
    """An environment with no data is skipped, not normalized."""
    with pytest.raises(ValueError, match="no arrays"):
        min_max_normalize([])


def test_min_max_normalize_rejects_a_constant():
    """Equal bounds leave no range, which would otherwise divide by zero."""
    with pytest.raises(ValueError, match="no range"):
        min_max_normalize([[3.0, 3.0], [3.0]])


# --- median_ti ----------------------------------------------------------------


def _shuffled_ranks(n_runs: int, n_points: int = 3):
    ranks = np.tile(np.arange(n_runs, dtype=float)[:, None], n_points)
    return np.random.default_rng(0).permuted(ranks, axis=0)


def test_tolerance_interval_reaches_its_stated_confidence():
    """Across many independent point estimates, the 95%/95% interval covers at
    least 95% of the run distribution in at least 95% of them."""
    runs = np.random.default_rng(0).random((300, 4000))  # uniform: coverage = width
    _median, ti_lo, ti_hi = median_ti(runs)
    assert ((ti_hi - ti_lo) >= 0.95).mean() >= 0.95


def test_tolerance_interval_trims_order_statistics_with_enough_runs():
    """With 300 runs the interval drops the 3 most extreme runs on each side."""
    median, ti_lo, ti_hi = median_ti(_shuffled_ranks(300))
    assert np.allclose(median, 149.5)
    assert np.allclose(ti_lo, 3.0)
    assert np.allclose(ti_hi, 296.0)


@pytest.mark.parametrize("n_runs", [3, 100])
def test_tolerance_interval_spans_all_runs_when_too_few_to_trim(n_runs: int):
    """Below the runs needed to trim anything, the interval is [min, max]."""
    _median, ti_lo, ti_hi = median_ti(_shuffled_ranks(n_runs))
    assert np.allclose(ti_lo, 0.0)
    assert np.allclose(ti_hi, n_runs - 1)


def test_tolerance_interval_is_nan_where_a_run_is_missing():
    """Like the mean, the median and band are only defined where every run
    contributes."""
    stack = _shuffled_ranks(5)
    stack[2, 1] = np.nan
    median, ti_lo, ti_hi = median_ti(stack)
    for values in (median, ti_lo, ti_hi):
        assert np.isnan(values[1])
        assert not np.isnan(values[[0, 2]]).any()


def test_median_ti_of_per_run_scalars_is_one_interval():
    """Too few runs to trim, so the band spans the smallest and largest run."""
    center, low, high = median_ti([1.0, 2.0, 6.0])
    assert center.shape == ()
    assert (center, low, high) == (2.0, 1.0, 6.0)


# --- best_by_mean -------------------------------------------------------------


def test_best_by_mean_picks_the_highest_mean():
    """The chosen key has the highest mean over its runs, not the highest run."""
    scores = {0.1: [1.0, 5.0], 0.01: [4.0, 4.0], 0.001: [0.0, 2.0]}
    assert best_by_mean(scores) == 0.01


def test_best_by_mean_never_picks_a_key_with_a_nan_run():
    """A run that finished no episode leaves its key's mean undefined."""
    scores = {"diverged": [9.0, np.nan], "stable": [1.0, 1.0]}
    assert best_by_mean(scores) == "stable"


def test_best_by_mean_breaks_ties_by_order():
    """Equal means resolve to the first key, so the choice is deterministic."""
    assert best_by_mean({"b": [2.0], "a": [2.0]}) == "b"


@pytest.mark.parametrize("scores", [{}, {"only": [np.nan]}])
def test_best_by_mean_raises_without_a_defined_mean(scores):
    """With nothing to choose from there is no best key to report."""
    with pytest.raises(ValueError, match="no key"):
        best_by_mean(scores)
