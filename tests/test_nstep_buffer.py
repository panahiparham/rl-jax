from __future__ import annotations

import equinox as eqx
import jax
import numpy as np

from agents.ddqn import DDQNAgent, DDQNConfig
from agents.dqn import DQNAgent, DQNConfig
from agents.random_buffered import RandomBufferAgent, RandomBufferConfig
from environments import ENVIRONMENTS
from environments.catch import CatchConfig
from main import interaction


def _run_random_buffered(total):
    env = ENVIRONMENTS["catch"].build(CatchConfig(EPISODE_CUTOFF=5))
    agent = RandomBufferAgent(
        RandomBufferConfig(TOTAL_TIMESTEPS=total, BUFFER_SIZE=32, BATCH_SIZE=8)
    )
    run = jax.jit(lambda key: interaction(key, agent, env, total))
    _metrics, final_carry = jax.block_until_ready(run(jax.random.key(0)))
    return agent, final_carry[1]


def test_random_buffered_runs_and_can_sample_is_correct():
    agent, agent_state = _run_random_buffered(20)
    assert agent.can_sample(agent_state)


def test_random_buffered_can_sample_false_before_min_length():
    agent, agent_state = _run_random_buffered(3)
    assert not agent.can_sample(agent_state)


def test_dqn_runs_under_jit_and_vmap_with_n_step():
    env = ENVIRONMENTS["catch"].build(CatchConfig(EPISODE_CUTOFF=5))
    agent = DQNAgent(
        DQNConfig(
            TOTAL_TIMESTEPS=20,
            BUFFER_SIZE=32,
            BATCH_SIZE=4,
            LEARNING_STARTS=4,
            HIDDEN_SIZE=8,
            N_STEP=3,
        )
    )
    run = jax.jit(jax.vmap(lambda key: interaction(key, agent, env, 20)))
    metrics, final_carry = jax.block_until_ready(
        run(jax.random.split(jax.random.key(0), 3))
    )
    assert metrics["reward"].shape == (3, 20)
    assert np.isfinite(np.asarray(final_carry[1].q.layer3.weight)).all()


def test_ddqn_runs_under_jit_and_vmap_with_n_step():
    env = ENVIRONMENTS["catch"].build(CatchConfig(EPISODE_CUTOFF=5))
    agent = DDQNAgent(
        DDQNConfig(
            TOTAL_TIMESTEPS=20,
            BUFFER_SIZE=32,
            BATCH_SIZE=4,
            LEARNING_STARTS=4,
            HIDDEN_SIZE=8,
            N_STEP=2,
        )
    )
    run = jax.jit(jax.vmap(lambda key: interaction(key, agent, env, 20)))
    metrics, final_carry = jax.block_until_ready(
        run(jax.random.split(jax.random.key(0), 3))
    )
    assert metrics["reward"].shape == (3, 20)
    assert np.isfinite(np.asarray(final_carry[1].q.layer3.weight)).all()


def _trained_q_leaves(agent_cls, config_cls):
    """Train one agent on Catch with a fixed all-random policy, return q leaves."""
    env = ENVIRONMENTS["catch"].build(CatchConfig(EPISODE_CUTOFF=10))
    agent = agent_cls(
        config_cls(
            TOTAL_TIMESTEPS=200,
            BUFFER_SIZE=256,
            BATCH_SIZE=8,
            LEARNING_STARTS=8,
            HIDDEN_SIZE=16,
            LR=1e-2,
            N_STEP=2,
            TARGET_NETWORK_FREQUENCY=100,
            # all-random actions -> both agents see identical data
            EPSILON_START=1.0,
            EPSILON_END=1.0,
        )
    )
    run = jax.jit(lambda key: interaction(key, agent, env, 200))
    _metrics, final_carry = jax.block_until_ready(run(jax.random.key(0)))
    return jax.tree.leaves(eqx.filter(final_carry[1].q, eqx.is_array))


def test_ddqn_bootstraps_differently_from_dqn():
    """``ddqn`` values the online net's argmax under the target net, where
    ``dqn`` takes the target net's own max. Fed identical transitions from the
    same seed, the two rules therefore learn different Q-functions."""
    dqn_leaves = _trained_q_leaves(DQNAgent, DQNConfig)
    ddqn_leaves = _trained_q_leaves(DDQNAgent, DDQNConfig)
    assert any(
        not np.allclose(a, b) for a, b in zip(dqn_leaves, ddqn_leaves, strict=True)
    )
