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
from numpy.typing import NDArray

from analysis.curves import ema, return_curve, subsample
from analysis.stats import mean_ci
from main import Results, load_results

__all__ = [
    "Environment",
    "Series",
    "environment_curves",
    "plot_environment",
    "render_plots",
]

METRIC_RETURN = "return"
METRIC_REWARD = "reward"

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


def _signal(runs: Results, metric: str) -> NDArray[np.float64]:
    if metric == METRIC_REWARD:
        return ema(runs.reward)
    return return_curve(runs.reward, runs.done)


def environment_curves(
    experiment: Experiment, environment: Environment, points: int = 500
) -> list[tuple[Series, NDArray[np.int64], NDArray[np.float64]]]:
    """Each series' ``(timesteps, per-run curves)``, skipping series with no runs."""
    curves = []
    for series in environment.series:
        runs = load_results(experiment, series.component)
        if runs is None:
            continue
        timesteps, values = subsample(_signal(runs, environment.metric), points)
        curves.append((series, timesteps, values))
    return curves


def plot_environment(
    experiment: Experiment, environment: Environment, plots_dir: Path
) -> Path | None:
    """Save one environment's learning curve, a mean +/- 95% CI band per agent.

    The same shape as ``experiments/tuned/analysis.ipynb``. An environment with
    no completed runs at all renders nothing.
    """
    drawn = environment_curves(experiment, environment)
    if not drawn:
        return None

    fig, ax = plt.subplots(figsize=(9, 6))
    for series, timesteps, values in drawn:
        mean, low, high = mean_ci(values)
        ax.fill_between(timesteps, low, high, color=series.color, alpha=0.2)
        ax.plot(timesteps, mean, lw=2.5, color=series.color, label=series.label)
    seeds = min(len(values) for _, _, values in drawn)
    ax.set_title(f"{environment.title} ({seeds} seeds, 95% CI)")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(
        _YLABEL[environment.metric], rotation=0, ha="right", va="center", labelpad=12
    )
    if environment.ylim is not None:
        ax.set_ylim(*environment.ylim)
    ax.set_xticks([0, max(timesteps[-1] for _, timesteps, _ in drawn)])
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
