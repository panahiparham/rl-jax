"""Signals over time from the per-timestep ``reward``/``done`` each run stores."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import lfilter


def ema(reward: ArrayLike, beta: float = 0.99) -> NDArray[np.float64]:
    """Exponential moving average of per-timestep reward, along the last axis.

    A continuing task (e.g. Catch) has no episode to derive a return from, so
    its learning curve smooths the raw reward stream instead. ``beta`` is in
    ``[0, 1)``; higher means more smoothing, following
    ``ema[i] = beta * ema[i - 1] + (1 - beta) * reward[i]``. The first value
    is left unsmoothed - there is nothing to average against yet.
    """
    reward = np.asarray(reward, dtype=float)
    if reward.shape[-1] == 0:
        return reward
    # A filter state of beta * reward[0] makes the first output reward[0].
    initial = beta * reward[..., :1]
    smoothed, _ = lfilter([1 - beta], [1, -beta], reward, axis=-1, zi=initial)
    return np.asarray(smoothed, dtype=np.float64)


def return_curve(reward: ArrayLike, done: ArrayLike) -> NDArray[np.float64]:
    """Each timestep's episode return, along the last axis.

    Every step of an episode gets the return of that whole episode, so the
    curve's time average is the step-weighted mean episode return. Steps after
    the last completed episode are NaN.
    """
    reward = np.asarray(reward, dtype=float)
    ended = np.asarray(done) > 0
    curve = np.full(reward.shape, np.nan)
    for run in np.ndindex(reward.shape[:-1]):
        ends = np.flatnonzero(ended[run])
        if ends.size == 0:
            continue
        returns = np.diff(np.cumsum(reward[run])[ends], prepend=0.0)
        lengths = np.diff(ends, prepend=-1)
        curve[run][: ends[-1] + 1] = np.repeat(returns, lengths)
    return curve


def lifetime_average(curve: ArrayLike) -> NDArray[np.float64]:
    """Each run's curve averaged over its defined timesteps, along the last axis.

    This is the area under the curve per timestep: on a return curve it is the
    step-weighted mean episode return, and on raw reward the average reward.
    NaN for a run with no defined timestep.
    """
    curve = np.asarray(curve, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN runs
        return np.asarray(np.nanmean(curve, axis=-1), dtype=np.float64)
