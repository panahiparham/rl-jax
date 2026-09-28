"""Aggregates over runs: a center and a band at each point, and normalization.

Every function takes a stack with one row per run.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import binom


class Interval(NamedTuple):
    """A center line and the band around it, one value per point."""

    center: NDArray[np.float64]
    low: NDArray[np.float64]
    high: NDArray[np.float64]


def _per_point(samples: ArrayLike) -> NDArray[np.float64]:
    """``[n_runs, ...]`` samples as a ``[n_runs, n_points]`` stack."""
    samples = np.asarray(samples, dtype=float)
    if len(samples) == 0:
        raise ValueError("no runs to aggregate")
    return samples.reshape(len(samples), -1)


def _shaped_like(interval: Interval, samples: ArrayLike) -> Interval:
    shape = np.shape(samples)[1:]
    center, low, high = interval
    return Interval(center.reshape(shape), low.reshape(shape), high.reshape(shape))


def mean_ci(
    samples: ArrayLike,
    confidence: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> Interval:
    """Bootstrap a mean and confidence interval over seeds at each point.

    ``samples`` is ``[n_runs]`` or ``[n_runs, ...]``, and the interval takes its
    trailing shape. NaN wherever not every seed is present. With a single seed
    the band collapses onto the mean.
    """
    stack = _per_point(samples)
    n, m = stack.shape
    valid = (~np.isnan(stack)).sum(axis=0) == n
    mean = np.full(m, np.nan)
    ci_lo = np.full(m, np.nan)
    ci_hi = np.full(m, np.nan)
    sub = stack[:, valid]  # [n_seeds, n_valid], no NaNs
    mean[valid] = sub.mean(axis=0)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, sub.shape[1]))
    for s in range(0, n_boot, 1000):  # chunked to bound memory
        e = min(s + 1000, n_boot)
        idx = rng.integers(0, n, size=(e - s, n))  # resample seed indices
        boot[s:e] = sub[idx].mean(axis=1)
    tail = (100 - 100 * confidence) / 2
    ci_lo[valid], ci_hi[valid] = np.percentile(boot, [tail, 100 - tail], axis=0)
    return _shaped_like(Interval(mean, ci_lo, ci_hi), samples)


def median_ti(
    samples: ArrayLike, coverage: float = 0.95, confidence: float = 0.95
) -> Interval:
    """Median and tolerance interval over runs at each point.

    ``samples`` is ``[n_runs]`` or ``[n_runs, ...]``, and the interval takes its
    trailing shape. NaN wherever not every run is present. The interval is a
    pair of order statistics that covers at least ``coverage`` of the run
    distribution with probability ``confidence``. Too few runs (93 for 95%/95%)
    cannot reach that confidence; the interval then falls back to the runs' min
    and max.
    """
    stack = _per_point(samples)
    n, m = stack.shape
    valid = (~np.isnan(stack)).sum(axis=0) == n
    median = np.full(m, np.nan)
    ti_lo = np.full(m, np.nan)
    ti_hi = np.full(m, np.nan)

    # [r-th smallest, r-th largest] covers >= coverage with probability
    # P(Binomial(n, coverage) <= n - 2r).
    trimmed = max((n - int(binom.ppf(confidence, n, coverage))) // 2 - 1, 0)
    ordered = np.sort(stack[:, valid], axis=0)
    median[valid] = np.median(ordered, axis=0)
    ti_lo[valid] = ordered[trimmed]
    ti_hi[valid] = ordered[n - 1 - trimmed]
    return _shaped_like(Interval(median, ti_lo, ti_hi), samples)


def min_max_normalize(arrays: Sequence[ArrayLike]) -> list[NDArray[np.float64]]:
    """Rescale one environment's arrays onto ``[0, 1]`` with shared bounds.

    Pass every array measured on the same environment - e.g. each agent's
    seed stack - so they share its lowest and highest observed value and stay
    comparable. NaNs are ignored when finding the bounds and stay NaN.
    """
    values = [np.asarray(a, dtype=float) for a in arrays]
    low = min(np.nanmin(v) for v in values)
    high = max(np.nanmax(v) for v in values)
    return [(v - low) / (high - low) for v in values]
