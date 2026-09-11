"""End-to-end tests for the weekly benchmark suite's definition (benchmarks/core)."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

from experiment import commands

from benchmarking import report

_REPO = Path(__file__).resolve().parents[1]
_BENCH_CORE = _REPO / "benchmarks" / "core"

sys.path.insert(0, str(_BENCH_CORE))
import config as bench_core_config


def test_components_are_dqn_ddqn_random_trios_on_ten_offset_seeds():
    experiment = bench_core_config.EXPERIMENT
    names = {c.name for c in experiment.components}
    assert names == {
        "dqn_acrobot",
        "ddqn_acrobot",
        "random_acrobot",
        "dqn_cartpole",
        "ddqn_cartpole",
        "random_cartpole",
        "dqn_mountaincar",
        "ddqn_mountaincar",
        "random_mountaincar",
        "dqn_catch",
        "ddqn_catch",
        "random_catch",
        "dqn_pinball_box",
        "ddqn_pinball_box",
        "random_pinball_box",
        "dqn_pinball_easy",
        "ddqn_pinball_easy",
        "random_pinball_easy",
        "dqn_pinball_empty",
        "ddqn_pinball_empty",
        "random_pinball_empty",
    }
    for c in experiment.components:
        assert c.seeds == tuple(range(10, 20)), c.name
        assert c.name.split("_")[0] == c.config.AGENT, c.name


def test_pinball_medium_is_left_to_experiments_tuned():
    names = {c.name for c in bench_core_config.EXPERIMENT.components}

    assert not [n for n in names if "medium" in n]


def test_status_reports_twenty_one_components_and_two_hundred_ten_runs(
    tmp_path, capsys
):
    experiment = dataclasses.replace(bench_core_config.EXPERIMENT, results_dir=tmp_path)

    commands.run(experiment, lambda *args, **kwargs: None, argv=["status"])

    out = capsys.readouterr().out
    assert "bench_core: 210 run(s), 0 done, 210 pending" in out


def test_every_plotted_series_names_a_component_the_suite_runs():
    experiment = bench_core_config.EXPERIMENT
    names = {c.name for c in experiment.components}
    plotted = {
        series.component
        for environment in bench_core_config.ENVIRONMENTS
        for series in environment.series
    }

    assert plotted == names


def test_catch_is_plotted_as_a_continuing_task():
    metrics = {e.key: e.metric for e in bench_core_config.ENVIRONMENTS}

    assert metrics["catch"] == report.METRIC_REWARD
