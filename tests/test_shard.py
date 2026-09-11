"""Tests for this project's shard processing (``shard``).

Cartpole throughout: it is cheap and its episodes end at different steps for
different hyperparameters and seeds, so a run's output actually reflects what it
was asked to compute. The serial path is exercised by marking an environment
stateful in the registry rather than by pulling in Atari, which needs a wheel
the test environment does not have.
"""

from __future__ import annotations

import numpy as np
import pytest
from experiment.hypers import split_traced

from agents.dqn import DQNConfig
from agents.random import RandomConfig
from environments import ENVIRONMENTS
from environments.classic_control import CartpoleConfig
from main import ExperimentConfig, process_shard

STEPS = 300
METRICS = {"reward", "done"}


def cartpole(**hypers) -> ExperimentConfig:
    defaults = {
        "TOTAL_TIMESTEPS": STEPS,
        "LEARNING_STARTS": 50,
        "BUFFER_SIZE": 500,
        "BATCH_SIZE": 16,
        "HIDDEN_SIZE": 8,
    }
    return ExperimentConfig(
        AGENT="dqn",
        ENV="cartpole",
        AGENT_HYPERS=DQNConfig(**{**defaults, **hypers}),
        ENV_HYPERS=CartpoleConfig(),
    )


@pytest.fixture
def stateful_cartpole(monkeypatch):
    """Mark cartpole stateful, so a shard takes the one-at-a-time path."""
    spec = ENVIRONMENTS["cartpole"]
    monkeypatch.setitem(ENVIRONMENTS, "cartpole", spec._replace(vmappable=False))


# --- the contract -----------------------------------------------------------


def test_one_result_per_run():
    results = process_shard([cartpole(), cartpole()], [0, 1])
    assert len(results) == 2
    assert all(set(r) == METRICS for r in results)


def test_only_metrics_come_back():
    """The rest of the run's pytree is never fetched to the host."""
    assert set(process_shard([cartpole()], [0])[0]) == METRICS


def test_results_have_one_value_per_step():
    result = process_shard([cartpole()], [0])[0]
    assert all(array.shape == (STEPS,) for array in result.values())


def test_an_empty_shard_computes_nothing():
    assert process_shard([], []) == []


def test_a_seed_is_required_for_every_config():
    with pytest.raises(ValueError, match="config"):
        process_shard([cartpole(), cartpole()], [0])


def test_configs_must_share_their_static_fields():
    with pytest.raises(ValueError, match="static"):
        process_shard([cartpole(HIDDEN_SIZE=8), cartpole(HIDDEN_SIZE=16)], [0, 1])


# --- what a shard actually computes -----------------------------------------


def test_seeds_produce_different_runs():
    first, second = process_shard([cartpole(), cartpole()], [0, 1])
    assert not np.array_equal(first["done"], second["done"])


def test_traced_hypers_produce_different_runs():
    greedy, random = process_shard(
        [
            cartpole(EPSILON_START=0.0, EPSILON_END=0.0),
            cartpole(EPSILON_START=1.0, EPSILON_END=1.0),
        ],
        [0, 0],
    )
    assert not np.array_equal(greedy["done"], random["done"])


def test_results_come_back_in_the_order_asked_for():
    configs = [cartpole(LR=1e-1), cartpole(LR=1e-4), cartpole(LR=1e-1)]
    seeds = [0, 0, 1]
    batched = process_shard(configs, seeds)
    for index, (config, seed) in enumerate(zip(configs, seeds, strict=True)):
        alone = process_shard([config], [seed])[0]
        for name in METRICS:
            np.testing.assert_array_equal(alone[name], batched[index][name])


def test_an_agent_with_no_traced_hypers_still_batches():
    config = ExperimentConfig(
        AGENT="random",
        ENV="cartpole",
        AGENT_HYPERS=RandomConfig(TOTAL_TIMESTEPS=STEPS),
        ENV_HYPERS=CartpoleConfig(),
    )
    results = process_shard([config, config], [0, 1])
    assert len(results) == 2
    assert not np.array_equal(results[0]["done"], results[1]["done"])


# --- what shares a shard ----------------------------------------------------


def test_configs_differing_in_a_traced_hyper_share_a_static():
    """Which is what lets the planner batch them into one shard."""
    first, _ = split_traced(cartpole(LR=1e-3))
    second, _ = split_traced(cartpole(LR=5e-4))
    assert first == second


def test_a_static_difference_keeps_configs_apart():
    first, _ = split_traced(cartpole(HIDDEN_SIZE=8))
    second, _ = split_traced(cartpole(HIDDEN_SIZE=16))
    assert first != second


# --- the serial path --------------------------------------------------------


def test_a_stateful_environment_runs_one_at_a_time(stateful_cartpole):
    results = process_shard([cartpole(), cartpole()], [0, 1])
    assert len(results) == 2
    assert all(set(r) == METRICS for r in results)


def test_the_serial_path_computes_the_same_results(stateful_cartpole, monkeypatch):
    """Batching is an optimisation, so it must not change a run's output."""
    configs = [cartpole(LR=1e-1), cartpole(LR=1e-4)]
    serial = process_shard(configs, [0, 1])

    monkeypatch.undo()  # cartpole is vmappable again
    batched = process_shard(configs, [0, 1])

    for one, other in zip(serial, batched, strict=True):
        for name in METRICS:
            np.testing.assert_array_equal(one[name], other[name])
