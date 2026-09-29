"""Tests for reading stored runs back (``main.load_results``).

Runs a tiny Cartpole sweep through the real harness and ``process_shard``, so
what is loaded is exactly what a sweep stores.
"""

from __future__ import annotations

import numpy as np
import pytest
from experiment.design import Component, Experiment
from experiment.plan import plan_experiment
from experiment.runner import run_shards

from agents.dqn import DQNConfig
from environments.classic_control import CartpoleConfig
from main import ExperimentConfig, load_results, process_shard

STEPS = 200
LRS = [1e-3, 1e-2]


def cartpole(lr: float = LRS[0]) -> ExperimentConfig:
    return ExperimentConfig(
        AGENT="dqn",
        ENV="cartpole",
        AGENT_HYPERS=DQNConfig(
            TOTAL_TIMESTEPS=STEPS,
            LR=lr,
            LEARNING_STARTS=50,
            BUFFER_SIZE=500,
            BATCH_SIZE=16,
            HIDDEN_SIZE=8,
        ),
        ENV_HYPERS=CartpoleConfig(),
    )


def lr_sweep(results_dir, lrs=LRS) -> Experiment:
    return Experiment(
        name="toy",
        results_dir=results_dir,
        components=[
            Component(
                name="dqn",
                config=cartpole(),
                sweep={"AGENT_HYPERS.LR": list(lrs)},
                seeds=[1, 0],
            )
        ],
    )


@pytest.fixture(scope="module")
def stored(tmp_path_factory) -> Experiment:
    experiment = lr_sweep(tmp_path_factory.mktemp("results"))
    run_shards(experiment, plan_experiment(experiment), process_shard)
    return experiment


# --- stored runs --------------------------------------------------------------


def test_loaded_runs_are_what_process_shard_computed(stored):
    """Loading returns each run's reward and done exactly as computed, in seed
    order rather than the order the component lists its seeds."""
    expected = process_shard([cartpole(LRS[1])] * 2, [0, 1])

    loaded = load_results(stored, "dqn", where={"AGENT_HYPERS.LR": LRS[1]})

    assert loaded is not None
    assert list(loaded.seed) == [0, 1]
    assert loaded.reward.shape == loaded.done.shape == (2, STEPS)
    for row, run in enumerate(expected):
        assert np.array_equal(loaded.reward[row], run["reward"])
        assert np.array_equal(loaded.done[row], run["done"])


def test_without_where_every_run_is_loaded(stored):
    """No filter stacks every stored run of the component."""
    loaded = load_results(stored, "dqn")
    assert loaded is not None
    assert loaded.reward.shape == (len(LRS) * 2, STEPS)


# --- missing runs -------------------------------------------------------------


def test_an_empty_store_loads_nothing(tmp_path):
    """Before any run is stored there is nothing to plot."""
    assert load_results(lr_sweep(tmp_path), "dqn") is None


def test_a_sweep_point_with_no_stored_run_loads_nothing(stored):
    """A value the sweep never stored reads as missing, not as an error."""
    assert load_results(stored, "dqn", where={"AGENT_HYPERS.LR": 0.5}) is None


def test_a_partial_sweep_loads_only_the_stored_points(tmp_path):
    """With one sweep point stored, the other still reads as missing."""
    stored_lr, missing_lr = LRS
    first_point = lr_sweep(tmp_path, lrs=[stored_lr])
    run_shards(first_point, plan_experiment(first_point), process_shard)
    experiment = lr_sweep(tmp_path)

    assert load_results(experiment, "dqn", where={"AGENT_HYPERS.LR": stored_lr})
    missing = load_results(experiment, "dqn", where={"AGENT_HYPERS.LR": missing_lr})
    assert missing is None


def test_an_unknown_where_path_raises_before_anything_is_stored(tmp_path):
    """A typo fails loudly instead of reading as missing data."""
    with pytest.raises(AttributeError, match=r"AGENT_HYPERS\.RL"):
        load_results(lr_sweep(tmp_path), "dqn", where={"AGENT_HYPERS.RL": 0.1})
