"""``REWARD_CLIP`` on ``dqn``/``ddqn``: sign-clips the reward stored in the
replay buffer (and so used for training) without touching the reward reported
back to the caller for experiment results.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.ddqn import DDQNAgent, DDQNConfig
from agents.dqn import DQNAgent, DQNConfig
from environments.autoreset import AutoresetImmediate
from main import interaction

AGENTS = [(DQNAgent, DQNConfig), (DDQNAgent, DDQNConfig)]


class _Box:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _Discrete:
    def __init__(self, n):
        self.n = int(n)


class _FakeRewardEnv:
    """A DISABLED-mode env whose reward sweeps through negative, zero, and
    positive values of varying magnitude, to exercise sign-clipping."""

    def __init__(self, n: int = 4) -> None:
        self._n = n

    def observation_space(self, params=None):
        return _Box((1,), jnp.float32)

    def action_space(self, params=None):
        return _Discrete(self._n)

    def reset(self, key, params=None):
        return jnp.zeros((1,), jnp.float32), jnp.int32(0)

    def step(self, key, state, action, params=None):
        t = state + 1
        reward = (t.astype(jnp.float32) - 3.0) * 2.5
        obs = jnp.zeros((1,), jnp.float32)
        return obs, t, reward, jnp.asarray(False), jnp.asarray(False), {}


def _fake_env():
    return AutoresetImmediate(_FakeRewardEnv())


def _run(agent_cls, config_cls, *, reward_clip: bool, steps: int = 6):
    agent = agent_cls(
        config_cls(
            TOTAL_TIMESTEPS=steps,
            BUFFER_SIZE=steps,
            BATCH_SIZE=1,
            LEARNING_STARTS=steps,  # no training: pure buffer/metrics inspection
            HIDDEN_SIZE=8,
            REWARD_CLIP=reward_clip,
        )
    )
    env = _fake_env()
    run = jax.jit(lambda key: interaction(key, agent, env, steps))
    metrics, final_carry = jax.block_until_ready(run(jax.random.key(0)))
    return agent, metrics, final_carry[1]


@pytest.mark.parametrize("agent_cls,config_cls", AGENTS)
def test_reward_clip_defaults_to_disabled(agent_cls, config_cls):
    assert config_cls().REWARD_CLIP is False


@pytest.mark.parametrize("agent_cls,config_cls", AGENTS)
def test_buffer_stores_sign_clipped_reward(agent_cls, config_cls):
    agent, metrics, state = _run(agent_cls, config_cls, reward_clip=True)
    stored = np.asarray(agent._buffer.stored_transitions(state.buffer_state).reward)
    raw = np.asarray(metrics["reward"])
    np.testing.assert_array_equal(stored, np.sign(raw))
    # the fake env's rewards span negative, zero, and >1-magnitude positive,
    # so this is not a vacuous check
    assert (raw < -1).any() and (raw > 1).any() and (raw == 0).any()


@pytest.mark.parametrize("agent_cls,config_cls", AGENTS)
def test_buffer_stores_raw_reward_when_disabled(agent_cls, config_cls):
    agent, metrics, state = _run(agent_cls, config_cls, reward_clip=False)
    stored = np.asarray(agent._buffer.stored_transitions(state.buffer_state).reward)
    raw = np.asarray(metrics["reward"])
    np.testing.assert_array_equal(stored, raw)


@pytest.mark.parametrize("agent_cls,config_cls", AGENTS)
def test_recorded_metrics_are_never_clipped(agent_cls, config_cls):
    """The value reported to the caller matches the env's true reward,
    whether or not REWARD_CLIP is enabled for the agent's own buffer/update."""
    _, clipped_metrics, _ = _run(agent_cls, config_cls, reward_clip=True)
    _, unclipped_metrics, _ = _run(agent_cls, config_cls, reward_clip=False)
    np.testing.assert_array_equal(
        np.asarray(clipped_metrics["reward"]), np.asarray(unclipped_metrics["reward"])
    )
