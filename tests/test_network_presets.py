"""``NETWORK_PRESET`` dispatch for the LN variants: DQNAgent (and DDQNAgent,
which inherits ``_build_q``) actually build and step with "mlp_ln" and
"nature_cnn_ln", not just the plain "mlp"/"nature_cnn" presets already
exercised elsewhere.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from agents.dqn import DQNAgent, DQNConfig
from environments import ENVIRONMENTS
from environments.autoreset import AutoresetImmediate
from environments.catch import CatchConfig
from main import interaction


class _Box:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _Discrete:
    def __init__(self, n):
        self.n = int(n)


class _FakeImageEnv:
    """A DISABLED-mode env with (84,84,4) uint8 obs, for the nature_cnn_ln
    preset - no ale-py dependency, mirroring test_atari.py's fake image env."""

    def __init__(self, h: int = 84, w: int = 84, c: int = 4, n: int = 6) -> None:
        self._shape, self._n = (h, w, c), n

    def observation_space(self, params=None):
        return _Box(self._shape, jnp.uint8)

    def action_space(self, params=None):
        return _Discrete(self._n)

    def reset(self, key, params=None):
        return jnp.zeros(self._shape, jnp.uint8), jnp.int32(0)

    def step(self, key, state, action, params=None):
        t = state + 1
        obs = jnp.full(self._shape, (t % 256).astype(jnp.uint8), jnp.uint8)
        return obs, t, jnp.float32(1.0), jnp.asarray(False), jnp.asarray(False), {}


def test_mlp_ln_preset_runs_under_jit():
    env = ENVIRONMENTS["catch"].build(CatchConfig(EPISODE_CUTOFF=10))
    agent = DQNAgent(
        DQNConfig(
            TOTAL_TIMESTEPS=20,
            BUFFER_SIZE=32,
            BATCH_SIZE=4,
            LEARNING_STARTS=4,
            HIDDEN_SIZE=8,
            NETWORK_PRESET="mlp_ln",
        )
    )
    run = jax.jit(lambda key: interaction(key, agent, env, 20))
    metrics, final_carry = jax.block_until_ready(run(jax.random.key(0)))
    assert metrics["reward"].shape == (20,)
    assert np.isfinite(np.asarray(final_carry[1].q.layer3.weight)).all()


def test_nature_cnn_ln_preset_runs_under_jit():
    env = AutoresetImmediate(_FakeImageEnv())
    agent = DQNAgent(
        DQNConfig(
            TOTAL_TIMESTEPS=20,
            BUFFER_SIZE=32,
            BATCH_SIZE=4,
            LEARNING_STARTS=4,
            NETWORK_PRESET="nature_cnn_ln",
        )
    )
    run = jax.jit(lambda key: interaction(key, agent, env, 20))
    metrics, final_carry = jax.block_until_ready(run(jax.random.key(0)))
    assert metrics["reward"].shape == (20,)
    assert np.isfinite(np.asarray(final_carry[1].q.out.weight)).all()
