"""End-to-end tests for the Catch environment integration.

Unlike Atari, ``catch_jax.Catch`` is a pure, non-stateful jax env, so these
tests drive the real env (no fakes) through the registry and the full
interaction loop.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from agents.dqn import DQNAgent, DQNConfig
from agents.random import RandomConfig
from environments import ENVIRONMENTS
from environments.catch import CatchConfig
from main import ExperimentConfig, interaction, process_shard


def test_spaces_and_dtype():
    env = ENVIRONMENTS["catch"].build(CatchConfig(ROWS=10, COLUMNS=5))
    assert env.observation_space().shape == (10, 5)
    assert env.observation_space().dtype == jnp.float32
    assert env.action_space().n == 3


def test_init_and_step_shapes():
    env = ENVIRONMENTS["catch"].build(CatchConfig())
    state, obs = env.init(jax.random.key(0))
    assert obs.shape == (10, 5) and obs.dtype == jnp.float32
    _state2, reward, termination, truncation, _discount, next_obs = env.step(
        state, jax.random.key(1), jnp.int32(1)
    )
    assert next_obs.shape == (10, 5)
    assert reward.shape == ()
    assert termination.shape == () and truncation.shape == ()


def test_episode_cutoff_truncates_not_terminates():
    """Catch never reaches a real MDP terminal - only the configured cutoff ends
    an episode, as a truncation."""
    env = ENVIRONMENTS["catch"].build(CatchConfig(EPISODE_CUTOFF=5))
    state, _obs = env.init(jax.random.key(0))
    flags = []
    for n in range(5):
        state, _r, term, trunc, _discount, _obs = env.step(
            state, jax.random.key(n), jnp.int32(1)
        )
        flags.append((bool(term), bool(trunc)))
    assert flags == [(False, False)] * 4 + [(False, True)]


def test_registered_as_vmappable():
    assert ENVIRONMENTS["catch"].vmappable


def test_random_agent_vmapped_e2e():
    config = ExperimentConfig(
        AGENT="random",
        ENV="catch",
        AGENT_HYPERS=RandomConfig(TOTAL_TIMESTEPS=50),
        ENV_HYPERS=CatchConfig(EPISODE_CUTOFF=10),
    )
    runs = process_shard([config] * 3, [0, 1, 2])

    assert [run["reward"].shape for run in runs] == [(50,)] * 3
    # a 10-step cutoff ends an episode every 10th step, and the env resets
    # in place, so all 5 episodes fit in the 50-step budget
    assert [int(run["done"].sum()) for run in runs] == [5, 5, 5]


def test_dqn_agent_e2e():
    env = ENVIRONMENTS["catch"].build(CatchConfig(EPISODE_CUTOFF=10))
    agent = DQNAgent(
        DQNConfig(
            TOTAL_TIMESTEPS=100,
            BUFFER_SIZE=200,
            BATCH_SIZE=8,
            LEARNING_STARTS=10,
            TARGET_NETWORK_FREQUENCY=20,
        )
    )
    run = jax.jit(lambda key: interaction(key, agent, env, 100))
    metrics, final_carry = jax.block_until_ready(run(jax.random.key(0)))
    assert metrics["reward"].shape == (100,)
    assert np.isfinite(np.asarray(final_carry[1].q.layer3.weight)).all()
