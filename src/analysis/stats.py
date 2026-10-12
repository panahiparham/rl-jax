"""Aggregates over runs: intervals, normalization and picking the best setting.

Runs are the first axis of every array.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import binom


class Interval(NamedTuple):
    """A center line and the band around it, one value per point."""

    center: NDArray[np.float64]
    low: NDArray[np.float64]
    high: NDArray[np.float64]


# (random, human) Atari-57 scores from DQN Zoo's atari_data.py: episodes start after
# up to 30 no-ops and are capped at 108,000 frames.
ATARI_REFERENCE_SCORES = {
    "alien": (227.8, 7127.7),
    "amidar": (5.8, 1719.5),
    "assault": (222.4, 742.0),
    "asterix": (210.0, 8503.3),
    "asteroids": (719.1, 47388.7),
    "atlantis": (12850.0, 29028.1),
    "bank_heist": (14.2, 753.1),
    "battle_zone": (2360.0, 37187.5),
    "beam_rider": (363.9, 16926.5),
    "berzerk": (123.7, 2630.4),
    "bowling": (23.1, 160.7),
    "boxing": (0.1, 12.1),
    "breakout": (1.7, 30.5),
    "centipede": (2090.9, 12017.0),
    "chopper_command": (811.0, 7387.8),
    "crazy_climber": (10780.5, 35829.4),
    "defender": (2874.5, 18688.9),
    "demon_attack": (152.1, 1971.0),
    "double_dunk": (-18.6, -16.4),
    "enduro": (0.0, 860.5),
    "fishing_derby": (-91.7, -38.7),
    "freeway": (0.0, 29.6),
    "frostbite": (65.2, 4334.7),
    "gopher": (257.6, 2412.5),
    "gravitar": (173.0, 3351.4),
    "hero": (1027.0, 30826.4),
    "ice_hockey": (-11.2, 0.9),
    "jamesbond": (29.0, 302.8),
    "kangaroo": (52.0, 3035.0),
    "krull": (1598.0, 2665.5),
    "kung_fu_master": (258.5, 22736.3),
    "montezuma_revenge": (0.0, 4753.3),
    "ms_pacman": (307.3, 6951.6),
    "name_this_game": (2292.3, 8049.0),
    "phoenix": (761.4, 7242.6),
    "pitfall": (-229.4, 6463.7),
    "pong": (-20.7, 14.6),
    "private_eye": (24.9, 69571.3),
    "qbert": (163.9, 13455.0),
    "riverraid": (1338.5, 17118.0),
    "road_runner": (11.5, 7845.0),
    "robotank": (2.2, 11.9),
    "seaquest": (68.4, 42054.7),
    "skiing": (-17098.1, -4336.9),
    "solaris": (1236.3, 12326.7),
    "space_invaders": (148.0, 1668.7),
    "star_gunner": (664.0, 10250.0),
    "surround": (-10.0, 6.5),
    "tennis": (-23.8, -8.3),
    "time_pilot": (3568.0, 5229.2),
    "tutankham": (11.4, 167.6),
    "up_n_down": (533.4, 11693.2),
    "venture": (0.0, 1187.5),
    "video_pinball": (16256.9, 17667.9),
    "wizard_of_wor": (563.5, 4756.5),
    "yars_revenge": (3092.9, 54576.9),
    "zaxxon": (32.5, 9173.3),
}


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
    if not values:
        raise ValueError("no arrays to normalize")
    low = min(np.nanmin(v) for v in values)
    high = max(np.nanmax(v) for v in values)
    if low == high:
        raise ValueError(f"every value is {low}, so there is no range to normalize")
    return [(v - low) / (high - low) for v in values]


def human_normalize(scores: ArrayLike, game: str) -> NDArray[np.float64]:
    """Rescale a game's scores so random play is 0 and human play is 1.

    Uses the game's ``ATARI_REFERENCE_SCORES``. NaNs stay NaN.
    """
    if game not in ATARI_REFERENCE_SCORES:
        raise ValueError(f"no random and human scores for {game!r}")
    random, human = ATARI_REFERENCE_SCORES[game]
    return (np.asarray(scores, dtype=float) - random) / (human - random)


def best_by_mean[K](scores: Mapping[K, ArrayLike]) -> K:
    """The key whose runs score highest on average, e.g. the best learning rate.

    A key with a NaN run has no mean and is never chosen. Ties go to the key
    that comes first.
    """
    means = {
        key: float(np.asarray(runs, dtype=float).mean()) for key, runs in scores.items()
    }
    defined = {key: mean for key, mean in means.items() if not np.isnan(mean)}
    if not defined:
        raise ValueError("no key has a mean over its runs")
    return max(defined, key=defined.__getitem__)
