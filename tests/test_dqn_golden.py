"""Golden values pinning DQN-family initialization and short training runs.

The values were recorded before the Q networks were restructured. Parameter
digests after init must match exactly; training summaries match to a float
tolerance, since accumulated matmul rounding can differ across platforms.
"""

from __future__ import annotations

import hashlib

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from agents import AGENTS
from environments import ENVIRONMENTS
from environments.autoreset import AutoresetImmediate
from environments.catch import CatchConfig
from environments.minatar import MinAtarConfig
from main import interaction

_STEPS = 40
_SMALL = {
    "TOTAL_TIMESTEPS": _STEPS,
    "BUFFER_SIZE": 32,
    "BATCH_SIZE": 4,
    "LEARNING_STARTS": 8,
    "TARGET_NETWORK_FREQUENCY": 10,
    "EPSILON_DECAY_STEPS": 20,
    "HIDDEN_SIZE": 16,
    "LR": 1e-2,
}


class _Box:
    def __init__(self, shape: tuple[int, ...], dtype: jnp.dtype):
        self.shape = shape
        self.dtype = dtype


class _Discrete:
    def __init__(self, n: int):
        self.n = n


class _FakeImageEnv:
    """(36, 36, 4) uint8 frames, rewarding the action that matches the step."""

    _SHAPE = (36, 36, 4)
    _ACTIONS = 4

    def observation_space(self, params=None):
        return _Box(self._SHAPE, jnp.uint8)

    def action_space(self, params=None):
        return _Discrete(self._ACTIONS)

    def reset(self, key: jax.Array, params=None):
        return jnp.zeros(self._SHAPE, jnp.uint8), jnp.int32(0)

    def step(self, key: jax.Array, state: jax.Array, action: jax.Array, params=None):
        t = state + 1
        obs = jnp.full(self._SHAPE, (t * 37 % 256).astype(jnp.uint8), jnp.uint8)
        reward = (action == t % self._ACTIONS).astype(jnp.float32)
        return obs, t, reward, jnp.asarray(False), jnp.asarray(False), {}


def _environment(preset: str):
    if preset.startswith("mlp"):
        return ENVIRONMENTS["catch"].build(CatchConfig(EPISODE_CUTOFF=10))
    if preset.startswith("minatar"):
        return ENVIRONMENTS["minatar"].build(MinAtarConfig(GAME="breakout"))
    return AutoresetImmediate(_FakeImageEnv())


def _build(agent_name: str, preset: str):
    spec = AGENTS[agent_name]
    return spec.build(spec.config_cls(NETWORK_PRESET=preset, **_SMALL))


def _digest(net: eqx.Module):
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(eqx.filter(net, eqx.is_array)):
        array = jnp.asarray(leaf)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _init_digest(agent_name: str, preset: str):
    environment = _environment(preset)
    state = _build(agent_name, preset).init(
        jax.random.key(3),
        environment.observation_space(),
        environment.action_space(),
    )
    return _digest(state.q)


def _training_summary(agent_name: str, preset: str):
    agent = _build(agent_name, preset)
    run = jax.jit(
        lambda key: interaction(key, agent, _environment(preset), _STEPS)
    )
    metrics, (_, state, _, _) = run(jax.random.key(3))
    leaves = jax.tree.leaves(eqx.filter(state.q, eqx.is_array))
    return jnp.asarray(
        [
            jnp.sum(metrics["reward"]),
            sum(jnp.sum(leaf) for leaf in leaves),
            sum(jnp.sum(jnp.square(leaf)) for leaf in leaves),
        ]
    )


_MLP_DIGEST = "1273f8e4411fe55ac1936565534cfb15f5a27dff028be66f74cb8a5550d7c658"
_NATURE_DIGEST = "f0f213c67b459a5785469020e0e6c900d573c59a84b9187d24bfe7a6b24fe2eb"
_INIT_DIGESTS = {
    ("dqn", "mlp"): _MLP_DIGEST,
    ("dqn", "mlp_ln"): _MLP_DIGEST,
    ("dqn", "nature_cnn"): _NATURE_DIGEST,
    ("dqn", "nature_cnn_ln"): _NATURE_DIGEST,
    ("dqn", "minatar_cnn"): (
        "d424663abe816dc6c93cfe08baf816e4acc3d0d8be24dc9d01fce9414bc696c5"
    ),
    ("ddqn", "mlp"): _MLP_DIGEST,
    ("dqn_noheadbias", "mlp"): (
        "5102ea09680d8a0fcd2d16f0753fb4248f1d7dcda506622641a81b3d88320359"
    ),
    ("dqn_noheadbias", "nature_cnn"): (
        "ae1cd27100119d461017983f5f025ccb55b76dd9f1354afc27d70510ef292736"
    ),
}
_TRAINING_SUMMARIES = {
    ("dqn", "mlp"): [-4.0, -2.57589, 13.48174],
    ("dqn", "mlp_ln"): [-4.0, -1.628649, 14.18648],
    ("dqn", "nature_cnn"): [10.0, -1221.526, 382.9796],
    ("dqn", "nature_cnn_ln"): [11.0, -120.87, 438.1038],
    ("dqn", "minatar_cnn"): [1.0, -1837.5, 212.1719],
    ("ddqn", "mlp"): [-4.0, -5.010958, 13.09796],
    ("dqn_noheadbias", "mlp"): [-4.0, -5.422059, 12.74723],
    ("dqn_noheadbias", "nature_cnn"): [10.0, -1312.93, 384.8335],
}


@pytest.mark.parametrize(("agent_name", "preset"), sorted(_INIT_DIGESTS))
def test_initial_parameters_match_golden(agent_name: str, preset: str):
    """The same init key yields bit-identical Q-network parameters."""
    assert _init_digest(agent_name, preset) == _INIT_DIGESTS[agent_name, preset]


@pytest.mark.parametrize(("agent_name", "preset"), sorted(_TRAINING_SUMMARIES))
def test_training_run_matches_golden(agent_name: str, preset: str):
    """A short seeded run collects the same rewards and ends at the same
    parameters, summarized as their sum and sum of squares."""
    expected = jnp.asarray(_TRAINING_SUMMARIES[agent_name, preset])
    summary = _training_summary(agent_name, preset)
    assert jnp.allclose(summary, expected, rtol=1e-4), summary
