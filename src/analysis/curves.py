"""One run's signal over time, from the per-timestep ``reward``/``done`` it stores."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def ema(reward: ArrayLike, beta: float = 0.99) -> NDArray[np.float64]:
    """Exponential moving average of per-timestep reward.

    A continuing task (e.g. Catch) has no episode to derive a return from, so
    its learning curve smooths the raw reward stream instead. ``beta`` is in ``[0, 1)``; higher means more smoothing, following
    ``ema[i] = beta * ema[i - 1] + (1 - beta) * reward[i]``. The first value
    is left unsmoothed - there is nothing to average against yet.
    """
    reward = np.asarray(reward, dtype=float)
    smoothed = np.empty_like(reward)
    if reward.size:
        smoothed[0] = reward[0]
        for i in range(1, reward.size):
            smoothed[i] = beta * smoothed[i - 1] + (1 - beta) * reward[i]
    return smoothed
