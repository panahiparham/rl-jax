"""End-to-end tests for the benchmark plot renderer (benchmarks/core/report.py)."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "benchmarks" / "core"))

import report
from experiment.design import Component, Experiment
from experiment.plan import Run, Shard
from experiment.results import ResultWriter

T = 20


@dataclasses.dataclass(frozen=True)
class Cfg:
    NAME: str = "dqn"


def _experiment(results_dir: Path, *names: str) -> Experiment:
    return Experiment(
        name="bench_core",
        results_dir=results_dir,
        components=[Component(name=n, config=Cfg()) for n in names],
    )


def _seed_component(experiment: Experiment, name: str, returns: list[float]) -> None:
    """A fake component: one seed per return, each a single-episode run."""
    results, runs = [], []
    for seed, ret in enumerate(returns):
        result = {"reward": np.zeros(T), "done": np.zeros(T)}
        result["done"][-1] = 1
        result["reward"][-1] = ret
        results.append(result)
        runs.append(Run(config=Cfg(), seed=seed))

    with ResultWriter(experiment) as writer:
        writer.save(Shard(component=name, runs=tuple(runs)), results)


def _seed_continuing_component(
    experiment: Experiment, name: str, rewards: list[float]
) -> None:
    """A fake continuing task: one seed per reward, no episode ever ending."""
    results, runs = [], []
    for seed, reward in enumerate(rewards):
        results.append({"reward": np.full(T, reward), "done": np.zeros(T)})
        runs.append(Run(config=Cfg(), seed=seed))

    with ResultWriter(experiment) as writer:
        writer.save(Shard(component=name, runs=tuple(runs)), results)


def _environment(key: str, *components: str, metric: str = report.METRIC_RETURN):
    series = tuple(
        report.Series(label=name, color=f"C{i}", component=name)
        for i, name in enumerate(components)
    )
    return report.Environment(key=key, title=key, series=series, metric=metric)


def _spy_on_curves(monkeypatch) -> list[tuple[str, np.ndarray]]:
    drawn: list[tuple[str, np.ndarray]] = []
    real = report.plot_mean_ci

    def spy(ax, grid, stack, label, color, **kwargs):
        drawn.append((label, stack))
        return real(ax, grid, stack, label, color, **kwargs)

    monkeypatch.setattr(report, "plot_mean_ci", spy)
    return drawn


def test_plot_environment_overlays_every_agent_on_one_png(tmp_path):
    experiment = _experiment(tmp_path / "results", "dqn_catch", "random_catch")
    _seed_component(experiment, "dqn_catch", [10.0, 11.0, 12.0])
    _seed_component(experiment, "random_catch", [1.0, 2.0, 3.0])
    plots_dir = tmp_path / "plots"

    path = report.plot_environment(
        experiment, _environment("catch", "dqn_catch", "random_catch"), plots_dir
    )

    assert path == plots_dir / "catch.png"
    assert sorted(p.name for p in plots_dir.glob("*.png")) == ["catch.png"]


def test_plot_environment_returns_none_without_runs(tmp_path):
    experiment = _experiment(tmp_path / "results", "dqn_catch")

    path = report.plot_environment(
        experiment, _environment("catch", "dqn_catch"), tmp_path / "plots"
    )

    assert path is None


def test_plot_environment_draws_only_the_agents_that_have_runs(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path / "results", "dqn_catch", "ddqn_catch")
    _seed_component(experiment, "dqn_catch", [10.0, 11.0, 12.0])
    drawn = _spy_on_curves(monkeypatch)

    report.plot_environment(
        experiment, _environment("catch", "dqn_catch", "ddqn_catch"), tmp_path / "plots"
    )

    assert [label for label, _ in drawn] == ["dqn_catch"]


def test_plot_environment_smooths_reward_for_a_continuing_task(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path / "results", "dqn_catch")
    _seed_continuing_component(experiment, "dqn_catch", [0.5, 0.5, 0.5])
    drawn = _spy_on_curves(monkeypatch)

    report.plot_environment(
        experiment,
        _environment("catch", "dqn_catch", metric=report.METRIC_REWARD),
        tmp_path / "plots",
    )

    assert drawn[0][1][:, 0] == pytest.approx(0.5)


def test_plot_environment_finds_no_episode_return_in_a_continuing_task(
    tmp_path, monkeypatch
):
    experiment = _experiment(tmp_path / "results", "dqn_catch")
    _seed_continuing_component(experiment, "dqn_catch", [0.5, 0.5, 0.5])
    drawn = _spy_on_curves(monkeypatch)

    report.plot_environment(
        experiment, _environment("catch", "dqn_catch"), tmp_path / "plots"
    )

    assert np.isnan(drawn[0][1]).all()


def test_render_plots_saves_one_png_per_environment(tmp_path):
    experiment = _experiment(tmp_path / "results", "dqn_catch", "dqn_cartpole")
    _seed_component(experiment, "dqn_catch", [10.0, 11.0, 12.0])
    _seed_component(experiment, "dqn_cartpole", [1.0, 2.0, 3.0])
    plots_dir = tmp_path / "plots"

    paths = report.render_plots(
        experiment,
        [_environment("catch", "dqn_catch"), _environment("cartpole", "dqn_cartpole")],
        plots_dir,
    )

    assert paths == [plots_dir / "catch.png", plots_dir / "cartpole.png"]


def test_render_plots_deletes_a_plot_left_by_an_earlier_run(tmp_path):
    experiment = _experiment(tmp_path / "results", "dqn_catch")
    _seed_component(experiment, "dqn_catch", [10.0, 11.0, 12.0])
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    (plots_dir / "pinball_medium.png").write_bytes(b"last week")

    report.render_plots(experiment, [_environment("catch", "dqn_catch")], plots_dir)

    assert sorted(p.name for p in plots_dir.glob("*.png")) == ["catch.png"]
