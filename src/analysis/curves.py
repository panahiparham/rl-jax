"""One run's signal over time, from the per-timestep ``reward``/``done`` it stores."""

from __future__ import annotations

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
