"""Learning-curve rendering for a benchmark run.

Turns a benchmark experiment's stored results into one plot per environment:
every agent run on that environment overlaid as a mean +/- 95% bootstrap CI
band, the same shape as ``experiments/tuned``'s learning curves. Plots are
saved wherever the caller points ``plots_dir`` - point it at a directory git
actually tracks, since ``**/plots/`` is gitignored except where a benchmark
negates it (see ``benchmarks/core/plots/``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from experiment.design import Experiment
from experiment.results import load_result, load_runs
from numpy.typing import NDArray

from analysis.plotting import (
    ema_reward_grids_for,
    plot_mean_ci,
    seed_grids_for,
    style,
)

__all__ = ["Environment", "Series", "plot_environment", "render_plots"]

METRIC_RETURN = "return"
METRIC_REWARD = "reward"

_GRIDS_FOR = {METRIC_RETURN: seed_grids_for, METRIC_REWARD: ema_reward_grids_for}
_YLABEL = {METRIC_RETURN: "Return", METRIC_REWARD: "Reward\n(EMA)"}


@dataclasses.dataclass(frozen=True)
class Series:
    """One agent's curve on an environment's learning curve."""

    label: str
    color: str
    component: str


@dataclasses.dataclass(frozen=True)
class Environment:
    """One environment's learning curve: every agent run on it, overlaid."""

    key: str
    title: str
    series: tuple[Series, ...]
    metric: str = METRIC_RETURN
    ylim: tuple[float, float] | None = None


def _grid_for(
    experiment: Experiment, component: str, points: int
) -> NDArray[np.float64]:
    """A shared [0, total_steps] timestep grid, read off any of the runs."""
    df = load_runs(experiment, component)
    total_steps = len(load_result(experiment, component, df["run_id"][0])["reward"])
    return np.linspace(0, total_steps, points)


def plot_environment(
    experiment: Experiment,
    environment: Environment,
    plots_dir: Path,
    *,
    grid_points: int = 500,
) -> Path | None:
    """Save one environment's learning curve, a mean +/- 95% CI band per agent.

    Each series gets its own timestep grid, matching
    ``experiments/tuned/analysis.ipynb``. A series with no completed runs is
    skipped; an environment with no completed runs at all renders nothing.
    """
    grids_for = _GRIDS_FOR[environment.metric]
    drawn = []
    for series in environment.series:
        if load_runs(experiment, series.component).is_empty():
            continue
        grid = _grid_for(experiment, series.component, grid_points)
        drawn.append((series, grid, grids_for(experiment, series.component, grid)))
    if not drawn:
        return None

    fig, ax = plt.subplots(figsize=(9, 6))
    for series, grid, stack in drawn:
        plot_mean_ci(ax, grid, stack, series.label, series.color)
    seeds = min(stack.shape[0] for _, _, stack in drawn)
    ax.set_title(f"{environment.title} ({seeds} seeds, 95% CI)")
    ax.legend(loc="lower right", frameon=False)
    style(ax, ylim=environment.ylim, ylabel=_YLABEL[environment.metric])
    ax.set_xticks([0, max(grid[-1] for _, grid, _ in drawn)])
    fig.tight_layout()

    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / f"{environment.key}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_plots(
    experiment: Experiment, environments: Sequence[Environment], plots_dir: Path
) -> list[Path]:
    """Re-render a benchmark run's plots, dropping whatever was there before.

    Wiping first is what makes a run's plot set exactly its own: a plot whose
    environment has since been renamed or dropped would otherwise linger and
    read as part of this week's results.
    """
    plots_dir = Path(plots_dir)
    for stale in plots_dir.glob("*.png"):
        stale.unlink()
    rendered = [plot_environment(experiment, env, plots_dir) for env in environments]
    return [path for path in rendered if path is not None]
