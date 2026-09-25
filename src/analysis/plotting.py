"""Shared plotting helpers for experiment analysis notebooks.

Kept out of the harness, which stores curves without knowing what they mean;
everything here reads them as this project's reward/done signals. Import it explicitly in a notebook::

    from analysis.plotting import seed_grids_for, plot_mean_ci, style

The pipeline turns the per-timestep ``reward``/``done`` curves
stored per run into per-seed *return-over-time* stacks, then a mean ± bootstrap-CI
band:

* :func:`episode_returns` - per-episode ``(end_timestep, return)`` for one run.
* :func:`interp_on_grid`  - one seed's curve as a function of timestep (linear
  interpolation onto a shared grid; NaN outside its observed range, no smoothing).
* :func:`seed_grids_for`  - the ``[n_seeds, len(grid)]`` raw-return stack for one
  component dir; :func:`ema_reward_grids_for` is the continuing-task alternative
  (no episode to derive a return from - smooths reward directly instead).
* :func:`weighted_lifetime_return` / :func:`weighted_lifetime_return_stack` - a run's
  episode returns collapsed to one length-weighted scalar, stacked per swept hyper
  value (e.g. learning rate) for a sensitivity curve.
  :func:`weighted_lifetime_returns_for` collects that scalar for every run of a
  component instead.
* :func:`average_lifetime_reward` / :func:`average_lifetime_reward_stack` - the same
  idea for a continuing task (e.g. Catch), which has no episode to derive a return
  from: mean reward rate over the whole run instead. :func:`ema_reward` is its
  per-timestep (not collapsed) version, feeding :func:`ema_reward_grids_for`.
* :func:`mean_over_seeds` / :func:`bootstrap_mean_ci` - pointwise aggregates, defined
  only where *every* seed contributes (so the mean is always over the same seeds).
* :func:`median_tolerance_interval` - a pointwise median and nonparametric
  tolerance interval over runs, showing how widely individual runs vary rather
  than how certain the mean is.
* :func:`min_max_normalize` - one environment's curves or scalars rescaled onto
  ``[0, 1]`` with shared bounds, so stacks from different environments can be
  pooled into one aggregate.
* :func:`plot_mean_ci` / :func:`plot_median_ti` / :func:`style` - draw a band +
  center line, and shared axes styling.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import polars as pl
from experiment.design import Experiment
from experiment.results import load_result, load_runs
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray
from scipy.stats import binom

CurveFn = Callable[[dict[str, Any]], tuple[NDArray[np.float64], NDArray[np.float64]]]
MetricFn = Callable[[dict[str, Any]], float]


def episode_returns(
    reward: ArrayLike, done: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Find each completed episode's end timestep and return.

    Returns an ``(ends, returns)`` pair. An episode ends where ``done``
    fires, and its return is the reward summed since the previous end.
    """
    reward = np.asarray(reward, dtype=float)
    idx = np.flatnonzero(np.asarray(done) > 0)
    if idx.size == 0:
        return np.array([]), np.array([])
    cumr = np.cumsum(reward)
    prev = np.concatenate(([0.0], cumr[idx[:-1]]))
    return idx.astype(np.float64), cumr[idx] - prev


def episode_lengths(ends: ArrayLike) -> NDArray[np.float64]:
    """Each episode's length in timesteps.

    The inclusive end indices diffed with ``prepend=-1`` (timestep -1:
    "before the run starts"). Every timestep belongs to exactly one episode,
    since the environments autoreset in place.
    """
    return np.diff(np.asarray(ends), prepend=-1)


def weighted_lifetime_return(reward: ArrayLike, done: ArrayLike) -> float:
    """Summarize one run as its length-weighted mean episode return.

    Longer episodes count for more than shorter ones, unlike a plain mean over
    episodes - useful when episode length itself varies with how well the
    agent is doing (e.g. Acrobot, MountainCar). Returns
    ``sum(return_i * length_i) / sum(length_i)`` over completed episodes, or
    NaN if the run completed none.
    """
    ends, rets = episode_returns(reward, done)
    if ends.size == 0:
        return float("nan")
    return float(np.average(rets, weights=episode_lengths(ends)))


def average_lifetime_reward(reward: ArrayLike) -> float:
    """Summarize one run as its mean per-step reward.

    A continuing task (e.g. Catch) never terminates or truncates within a
    run, so there is no episode to derive a return from at all - a
    length-weighted episode return (:func:`weighted_lifetime_return`) does
    not apply. Mean reward rate over the whole run is the summary instead.
    """
    return float(np.mean(np.asarray(reward, dtype=float)))


def ema_reward(reward: ArrayLike, beta: float = 0.99) -> NDArray[np.float64]:
    """Exponential moving average of per-timestep reward.

    The learning-curve counterpart of :func:`average_lifetime_reward` for a
    continuing task: with no episode boundaries to interpolate a return
    between, smoothing the raw reward stream directly is the only option.
    ``beta`` is in ``[0, 1)``; higher means more smoothing, following
    ``ema[i] = beta * ema[i - 1] + (1 - beta) * reward[i]``. The first value
    is left unsmoothed - there is nothing to average against yet.
    """
    reward = np.asarray(reward, dtype=float)
    ema = np.empty_like(reward)
    if reward.size:
        ema[0] = reward[0]
        for i in range(1, reward.size):
            ema[i] = beta * ema[i - 1] + (1 - beta) * reward[i]
    return ema


def interp_on_grid(
    ends: NDArray[np.float64], rets: NDArray[np.float64], grid: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Interpolate one seed's episode returns onto a shared timestep grid.

    Sharing a grid lets seeds be averaged pointwise. The result is return as
    a function of timestep, NaN outside the observed range - there is no
    extrapolation and no smoothing.
    """
    if ends.size == 0:
        return np.full(grid.shape, np.nan)
    return np.interp(grid, ends, rets, left=np.nan, right=np.nan)


def _run_ids(
    experiment: Experiment, component: str, run_ids: Sequence[str] | None
) -> list[str]:
    if run_ids is not None:
        return list(run_ids)
    return load_runs(experiment, component)["run_id"].to_list()


def _interp_stack_for(
    experiment: Experiment,
    component: str,
    grid: NDArray[np.float64],
    run_ids: Sequence[str] | None,
    curve_fn: CurveFn,
) -> NDArray[np.float64]:
    """Stack every seed's ``curve_fn(result)`` interpolated onto a shared grid.

    Shared by :func:`seed_grids_for` and :func:`ema_reward_grids_for` - only
    the per-run ``(xs, ys)`` curve differs.

    Each component has its own table, so no per-config filtering is needed
    unless the component itself sweeps a hyper (e.g. a learning-rate sweep),
    in which case ``run_ids`` restricts to one sweep point's runs. Returns an
    ``[n_seeds, len(grid)]`` array, one row per run.
    """
    grids: list[NDArray[np.float64]] = []
    for rid in _run_ids(experiment, component, run_ids):
        xs, ys = curve_fn(load_result(experiment, component, rid))
        grids.append(interp_on_grid(xs, ys, grid))
    return np.vstack(grids) if grids else np.empty((0, len(grid)))


def seed_grids_for(
    experiment: Experiment,
    component: str,
    grid: NDArray[np.float64],
    run_ids: Sequence[str] | None = None,
) -> NDArray[np.float64]:
    """Stack every seed's return-over-time for one component.

    For a continuing task with no episode to derive a return from, use
    :func:`ema_reward_grids_for` instead. Returns an ``[n_seeds, len(grid)]``
    array, one row per run.
    """

    def curve(c: dict[str, Any]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return episode_returns(c["reward"], c["done"])

    return _interp_stack_for(experiment, component, grid, run_ids, curve)


def ema_reward_grids_for(
    experiment: Experiment,
    component: str,
    grid: NDArray[np.float64],
    beta: float = 0.99,
    run_ids: Sequence[str] | None = None,
) -> NDArray[np.float64]:
    """Stack every seed's EMA-smoothed reward-over-time for one component.

    The learning-curve counterpart of :func:`seed_grids_for` for a continuing
    task (e.g. Catch), which has no episode boundaries to derive a return
    from at all. Returns an ``[n_seeds, len(grid)]`` array, one row per run.
    """

    def curve(c: dict[str, Any]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        reward = np.asarray(c["reward"], dtype=float)
        return np.arange(reward.size, dtype=np.float64), ema_reward(reward, beta=beta)

    return _interp_stack_for(experiment, component, grid, run_ids, curve)


def weighted_lifetime_returns_for(
    experiment: Experiment,
    component: str,
    run_ids: Sequence[str] | None = None,
) -> NDArray[np.float64]:
    """Every run's weighted lifetime return for one component, one per run.

    ``run_ids`` restricts to a subset of runs, e.g. one game of a sweep.
    """
    results = (
        load_result(experiment, component, rid)
        for rid in _run_ids(experiment, component, run_ids)
    )
    return np.array([weighted_lifetime_return(r["reward"], r["done"]) for r in results])


def _metric_stack(
    experiment: Experiment,
    component: str,
    column: str,
    values: Sequence[Any],
    metric_fn: MetricFn,
) -> NDArray[np.float64]:
    """Per-seed ``metric_fn(curves)``, one column per swept hyper value.

    Shared by :func:`weighted_lifetime_return_stack` and
    :func:`average_lifetime_reward_stack` - only the per-run scalar differs.
    ``column`` is the dotted config column swept (e.g. ``"AGENT_HYPERS.LR"``),
    as flattened by :func:`~experiment.results.load_runs`; ``values`` are
    pulled in output-column order. Returns an ``[n_seeds, len(values)]`` array;
    a value with fewer runs than the widest column is padded with NaN.
    """
    df = load_runs(experiment, component)
    columns: list[list[float]] = []
    for value in values:
        rids = df.filter(pl.col(column) == value).sort("seed")["run_id"].to_list()
        columns.append(
            [metric_fn(load_result(experiment, component, rid)) for rid in rids]
        )
    width = max((len(c) for c in columns), default=0)
    padded = [c + [float("nan")] * (width - len(c)) for c in columns]
    return np.array(padded).T if width else np.empty((0, len(values)))


def weighted_lifetime_return_stack(
    experiment: Experiment, component: str, column: str, values: Sequence[Any]
) -> NDArray[np.float64]:
    """Per-seed weighted-lifetime-return, one column per swept hyper value.

    For a sensitivity curve: feed the result straight into
    :func:`bootstrap_mean_ci` / :func:`plot_mean_ci` with ``values`` as the
    x-axis grid. For a continuing task (e.g. Catch), use
    :func:`average_lifetime_reward_stack` instead.
    """

    def metric(c: dict[str, Any]) -> float:
        return weighted_lifetime_return(c["reward"], c["done"])

    return _metric_stack(experiment, component, column, values, metric)


def average_lifetime_reward_stack(
    experiment: Experiment, component: str, column: str, values: Sequence[Any]
) -> NDArray[np.float64]:
    """Per-seed average-lifetime-reward, one column per swept hyper value.

    The continuing-task counterpart of :func:`weighted_lifetime_return_stack`
    - see :func:`average_lifetime_reward`.
    """

    def metric(c: dict[str, Any]) -> float:
        return average_lifetime_reward(c["reward"])

    return _metric_stack(experiment, component, column, values, metric)


def mean_over_seeds(stack: ArrayLike) -> NDArray[np.float64]:
    """Average a seed stack pointwise.

    NaN wherever not every seed contributes. Holding the seed set fixed
    avoids an early bias toward the seeds that finish first.
    """
    stack = np.asarray(stack)
    with warnings.catch_warnings():
        # all-NaN edge columns
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
    mean[(~np.isnan(stack)).sum(axis=0) < stack.shape[0]] = np.nan
    return mean


def median_tolerance_interval(
    stack: ArrayLike, coverage: float = 0.95, confidence: float = 0.95
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Median and tolerance interval over runs at each point.

    Returns a ``(median, ti_lo, ti_hi)`` triple, NaN wherever not every run is
    present. The interval is a pair of order statistics that covers at least
    ``coverage`` of the run distribution with probability ``confidence``. Too
    few runs (93 for 95%/95%) cannot reach that confidence; the interval then
    falls back to the runs' min and max.
    """
    stack = np.asarray(stack, dtype=float)
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
    return median, ti_lo, ti_hi


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


def bootstrap_mean_ci(
    stack: ArrayLike,
    n_boot: int = 10_000,
    lo: float = 2.5,
    hi: float = 97.5,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Bootstrap a mean and confidence interval over seeds at each timestep.

    Returns a ``(mean, ci_lo, ci_hi)`` triple, NaN wherever not every seed is
    present. With a single seed the band collapses onto the mean.
    """
    stack = np.asarray(stack)
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
    ci_lo[valid], ci_hi[valid] = np.percentile(boot, [lo, hi], axis=0)
    return mean, ci_lo, ci_hi


def plot_mean_ci(
    ax: Axes,
    grid: NDArray[np.float64],
    stack: ArrayLike,
    label: str,
    color: str,
    n_boot: int = 10_000,
) -> NDArray[np.float64]:
    """Draw a mean line and its shaded bootstrap CI band onto an axis.

    Returns the plotted ``mean`` array. The band is masked to where every
    seed is present, so it has zero width for a single seed.
    """
    mean, ci_lo, ci_hi = bootstrap_mean_ci(stack, n_boot=n_boot)
    m = ~np.isnan(mean)
    ax.fill_between(grid[m], ci_lo[m], ci_hi[m], color=color, alpha=0.2)  # band
    ax.plot(grid[m], mean[m], lw=2.5, color=color, label=label)  # thick mean
    return mean


def plot_median_ti(
    ax: Axes,
    grid: NDArray[np.float64],
    stack: ArrayLike,
    label: str,
    color: str,
) -> NDArray[np.float64]:
    """Draw a median line and its shaded 95%/95% tolerance band onto an axis.

    Returns the plotted ``median`` array. The band is masked to where every
    run is present.
    """
    median, ti_lo, ti_hi = median_tolerance_interval(stack)
    m = ~np.isnan(median)
    ax.fill_between(grid[m], ti_lo[m], ti_hi[m], color=color, alpha=0.2)  # band
    ax.plot(grid[m], median[m], lw=2.5, color=color, label=label)  # thick median
    return median


def style(
    ax: Axes,
    ylim: tuple[float, float] | None = None,
    xlabel: str = "Timestep",
    ylabel: str = "Return",
) -> None:
    """Apply the shared axes styling: no grid, no top/right spines.

    ``ylim`` fixes the return range, which differs per env, e.g. Cartpole
    ``(0, 500)``, Acrobot ``(-500, 0)``, Pinball ``(-1000, 0)``. ``xlabel``
    is e.g. ``"Learning rate"`` for a sensitivity curve instead of the
    default learning-curve timestep axis. ``ylabel`` is drawn horizontal and
    right-aligned.
    """
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=0, ha="right", va="center", labelpad=12)
    if ylim is not None:
        ax.set_ylim(*ylim)
