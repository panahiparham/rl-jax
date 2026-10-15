"""QRC: DQN without a target network, plus an h head correcting the Q update."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from agents.dqn import DQNAgent, DQNConfig
from agents.qrc import QRCAgent, QRCConfig, qrc_loss
from components import Batch, DualHeadNetwork, build_q_network
from environments import ENVIRONMENTS
from environments.autoreset import AutoresetImmediate
from environments.catch import CatchConfig
from environments.minatar import MinAtarConfig
from main import interaction

# --- loss gradient ------------------------------------------------------------

_OBS_DIM, _ACTIONS = 3, 2


def _network_with_random_h():
    q = build_q_network("mlp", (_OBS_DIM,), _ACTIONS, 8, jax.random.key(0))
    network = DualHeadNetwork(q)
    h_weight = jax.random.normal(jax.random.key(1), network.h_out.weight.shape)
    h_bias = jax.random.normal(jax.random.key(2), (_ACTIONS,))
    return eqx.tree_at(
        lambda n: (n.h_out.weight, n.h_out.bias), network, (h_weight, h_bias)
    )


def _batch():
    obs_key, boot_key = jax.random.split(jax.random.key(3))
    return Batch(
        obs=jax.random.normal(obs_key, (4, _OBS_DIM)),
        action=jnp.asarray([0, 1, 1, 0]),
        ret=jnp.asarray([1.0, -0.5, 0.0, 100.0]),
        discount=jnp.asarray([0.9, 0.0, 0.81, 0.9]),
        boot_obs=jax.random.normal(boot_key, (4, _OBS_DIM)),
        boot_action=jnp.zeros(4, jnp.int32),
        mask=jnp.asarray([True, True, True, False]),
    )


def _q_value(net: DualHeadNetwork, obs: jax.Array, action: jax.Array):
    return net(obs)[0][action]


def _boot_value(net: DualHeadNetwork, obs: jax.Array):
    return jnp.max(net(obs)[0])


def _h_value(net: DualHeadNetwork, obs: jax.Array, action: jax.Array):
    return net(obs)[1][action]


def _sample_direction(
    network: DualHeadNetwork, batch: Batch, i: int, correction_scale: float
):
    """-delta grad(v) + c discount h grad(v') - (delta - h) grad(h) for one
    sample, computed without stop-gradients."""
    obs, action, boot_obs = batch.obs[i], batch.action[i], batch.boot_obs[i]
    v, h = _q_value(network, obs, action), _h_value(network, obs, action)
    delta = batch.ret[i] + batch.discount[i] * _boot_value(network, boot_obs) - v
    correction = correction_scale * batch.discount[i] * h
    return jax.tree.map(
        lambda gv, gb, gh: -delta * gv + correction * gb - (delta - h) * gh,
        eqx.filter_grad(_q_value)(network, obs, action),
        eqx.filter_grad(_boot_value)(network, boot_obs),
        eqx.filter_grad(_h_value)(network, obs, action),
    )


def _expected_gradient(
    network: DualHeadNetwork, batch: Batch, beta: float, correction_scale: float
):
    """QRC's update averaged over unmasked samples, with beta pulling the h
    head's parameters towards zero."""
    valid = [i for i in range(len(batch.mask)) if batch.mask[i]]
    directions = [_sample_direction(network, batch, i, correction_scale) for i in valid]
    mean = jax.tree.map(lambda *gs: sum(gs) / len(valid), *directions)
    penalized_h = jax.tree.map(lambda g, p: g + beta * p, mean.h_out, network.h_out)
    return eqx.tree_at(lambda g: g.h_out, mean, penalized_h)


@pytest.mark.parametrize(("beta", "correction_scale"), [(1.0, 1.0), (0.3, 0.5)])
def test_loss_gradient_is_the_qrc_update(beta: float, correction_scale: float):
    """The loss gradient equals QRC's update for the Q head, torso and h head.

    The batch mixes a terminal transition and a masked-out outlier, which must
    not contribute.
    """
    network, batch = _network_with_random_h(), _batch()

    grads = eqx.filter_grad(qrc_loss)(network, batch, beta, correction_scale)

    expected = _expected_gradient(network, batch, beta, correction_scale)
    for got, want in zip(
        jax.tree.leaves(grads), jax.tree.leaves(expected), strict=True
    ):
        assert jnp.allclose(got, want, rtol=1e-5, atol=1e-6)


def test_correction_vanishes_when_h_is_zero():
    """With h at zero and no penalty, the Q gradient is plain semi-gradient
    Q-learning without a target network."""
    q = build_q_network("mlp", (_OBS_DIM,), _ACTIONS, 8, jax.random.key(0))
    batch = _batch()

    def q_learning_loss(net: DualHeadNetwork):
        q_sa = jax.vmap(net.q)(batch.obs)
        v = jnp.take_along_axis(q_sa, batch.action[:, None], axis=-1).squeeze(-1)
        boot = jnp.max(jax.vmap(net.q)(batch.boot_obs), axis=-1)
        target = jax.lax.stop_gradient(batch.ret + batch.discount * boot)
        sq_err = jnp.where(batch.mask, 0.5 * jnp.square(v - target), 0.0)
        return jnp.sum(sq_err) / jnp.sum(batch.mask)

    network = DualHeadNetwork(q)
    grads = eqx.filter_grad(qrc_loss)(network, batch, 0.0, 1.0)
    expected = eqx.filter_grad(q_learning_loss)(network)

    for got, want in zip(
        jax.tree.leaves(grads.q), jax.tree.leaves(expected.q), strict=True
    ):
        assert jnp.allclose(got, want, rtol=1e-5, atol=1e-7)


# --- initialization -----------------------------------------------------------


def test_init_matches_dqn_q_network_with_zero_h():
    """For the same key, QRC's Q network is DQN's and h predicts zero."""
    env = ENVIRONMENTS["catch"].build(CatchConfig())
    spaces = (env.observation_space(), env.action_space())
    key = jax.random.key(4)

    qrc_state = QRCAgent(QRCConfig()).init(key, *spaces)
    dqn_state = DQNAgent(DQNConfig()).init(key, *spaces)

    q_leaves = jax.tree.leaves(qrc_state.network.q)
    for got, want in zip(q_leaves, jax.tree.leaves(dqn_state.q), strict=True):
        assert jnp.array_equal(got, want)
    for leaf in jax.tree.leaves(qrc_state.network.h_out):
        assert jnp.all(leaf == 0)


# --- runs ---------------------------------------------------------------------


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

    def observation_space(self, params=None):
        return _Box(self._SHAPE, jnp.uint8)

    def action_space(self, params=None):
        return _Discrete(4)

    def reset(self, key: jax.Array, params=None):
        return jnp.zeros(self._SHAPE, jnp.uint8), jnp.int32(0)

    def step(self, key: jax.Array, state: jax.Array, action: jax.Array, params=None):
        t = state + 1
        obs = jnp.full(self._SHAPE, (t * 37 % 256).astype(jnp.uint8), jnp.uint8)
        reward = (action == t % 4).astype(jnp.float32)
        return obs, t, reward, jnp.asarray(False), jnp.asarray(False), {}


def _environment(preset: str):
    if preset.startswith("mlp"):
        return ENVIRONMENTS["catch"].build(CatchConfig(EPISODE_CUTOFF=10))
    if preset.startswith("minatar"):
        return ENVIRONMENTS["minatar"].build(MinAtarConfig(GAME="breakout"))
    return AutoresetImmediate(_FakeImageEnv())


_PRESETS = [
    "mlp",
    "mlp_ln",
    "nature_cnn",
    "nature_cnn_ln",
    "minatar_cnn",
    "minatar_cnn_ln",
]


@pytest.mark.parametrize("n_step", [1, 3])
@pytest.mark.parametrize("preset", _PRESETS)
def test_runs_under_jit_and_vmap(preset: str, n_step: int):
    """Vmapped seeds train every preset with n-step returns, moving h off
    zero and keeping all parameters finite."""
    steps = 40
    agent = QRCAgent(
        QRCConfig(
            TOTAL_TIMESTEPS=steps,
            BUFFER_SIZE=32,
            BATCH_SIZE=4,
            LEARNING_STARTS=8,
            HIDDEN_SIZE=16,
            NETWORK_PRESET=preset,
            N_STEP=n_step,
            LR=1e-2,
        )
    )
    env = _environment(preset)
    run = jax.jit(jax.vmap(lambda key: interaction(key, agent, env, steps)))

    metrics, (_, state, _, _) = run(jax.random.split(jax.random.key(5), 2))

    assert metrics["reward"].shape == (2, steps)
    for leaf in jax.tree.leaves(eqx.filter(state.network, eqx.is_array)):
        assert jnp.isfinite(leaf).all()
    h_weight = state.network.h_out.weight
    assert jnp.any(h_weight.reshape(2, -1) != 0, axis=-1).all()
