"""``QNetworkLN``/``NatureCNNLN``: same shapes as their plain counterparts, and
the LayerNorm layers carry no learnable gain or bias - the whole point of the
"no-affine" variant is otherwise silently regressable.

Every network also uses the Nature DQN initializer: each weight and bias of a
layer is drawn from U(+/- 1 / sqrt(fan_in)).
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from components import (
    DualHeadNetwork,
    MinAtarCNN,
    MinAtarCNNLN,
    NatureCNN,
    NatureCNNLN,
    QNetwork,
    QNetworkLN,
    build_q_network,
)


def test_qnetwork_ln_output_shape_and_finite():
    net = QNetworkLN(obs_dim=6, action_dim=4, hidden_size=8, key=jax.random.key(0))
    out = net(jnp.ones((6,)))
    assert out.shape == (4,)
    assert np.isfinite(np.asarray(out)).all()


def test_qnetwork_ln_layers_have_no_affine_params():
    net = QNetworkLN(obs_dim=6, action_dim=4, hidden_size=8, key=jax.random.key(0))
    for ln in (net.ln1, net.ln2):
        assert ln.weight is None
        assert ln.bias is None


def test_nature_cnn_ln_output_shape_and_finite():
    obs = jnp.zeros((84, 84, 4), jnp.uint8)
    net = NatureCNNLN(obs_shape=obs.shape, action_dim=6, key=jax.random.key(0))
    out = net(obs)
    assert out.shape == (6,)
    assert np.isfinite(np.asarray(out)).all()


def test_nature_cnn_ln_head_has_no_affine_params():
    obs_shape = (84, 84, 4)
    net = NatureCNNLN(obs_shape=obs_shape, action_dim=6, key=jax.random.key(0))
    assert net.head_ln.weight is None
    assert net.head_ln.bias is None


def test_minatar_cnn_output_shape_and_finite():
    obs = jnp.zeros((10, 10, 4), jnp.bool_)
    net = MinAtarCNN(obs_shape=obs.shape, action_dim=3, key=jax.random.key(0))
    out = net(obs)
    assert out.shape == (3,)
    assert np.isfinite(np.asarray(out)).all()


def test_minatar_cnn_ln_output_shape_and_finite():
    obs = jnp.zeros((10, 10, 4), jnp.bool_)
    net = MinAtarCNNLN(obs_shape=obs.shape, action_dim=3, key=jax.random.key(0))
    out = net(obs)
    assert out.shape == (3,)
    assert np.isfinite(np.asarray(out)).all()


def test_minatar_cnn_ln_head_has_no_affine_params():
    net = MinAtarCNNLN(obs_shape=(10, 10, 4), action_dim=3, key=jax.random.key(0))
    assert net.head_ln.weight is None
    assert net.head_ln.bias is None


def test_minatar_cnn_parameter_count():
    """Breakout's 4 channels and 3 actions give 592 conv, 131,200 hidden and
    387 output parameters."""
    net = MinAtarCNN(obs_shape=(10, 10, 4), action_dim=3, key=jax.random.key(0))
    leaves = jax.tree.leaves(eqx.filter(net, eqx.is_array))
    assert sum(leaf.size for leaf in leaves) == 592 + 131_200 + 387


def test_minatar_cnn_reads_bool_observations_as_zeros_and_ones():
    """A bool observation gives the same values as its float encoding."""
    net = MinAtarCNN(obs_shape=(10, 10, 4), action_dim=3, key=jax.random.key(0))
    obs = jax.random.bernoulli(jax.random.key(1), 0.3, (10, 10, 4))
    assert jnp.allclose(net(obs), net(obs.astype(jnp.float32)))


@pytest.mark.parametrize("channel", range(4))
def test_minatar_cnn_uses_every_channel_of_the_last_axis(channel: int):
    """A single active cell changes the output whichever channel it is in, so
    the (H, W, C) layout is read as channels last."""
    net = MinAtarCNN(obs_shape=(10, 10, 4), action_dim=3, key=jax.random.key(0))
    empty = jnp.zeros((10, 10, 4), jnp.bool_)
    one_cell = empty.at[3, 5, channel].set(True)
    assert not jnp.allclose(net(empty), net(one_cell))


# --- initialization -----------------------------------------------------------

_KEY = jax.random.key(0)
_NETWORKS = {
    "mlp": QNetwork(obs_dim=6, action_dim=4, hidden_size=64, key=_KEY),
    "mlp_ln": QNetworkLN(obs_dim=6, action_dim=4, hidden_size=64, key=_KEY),
    "nature_cnn": NatureCNN(obs_shape=(84, 84, 4), action_dim=6, key=_KEY),
    "nature_cnn_ln": NatureCNNLN(obs_shape=(84, 84, 4), action_dim=6, key=_KEY),
    "minatar_cnn": MinAtarCNN(obs_shape=(10, 10, 4), action_dim=3, key=_KEY),
}


def _layers(net: eqx.Module):
    def is_layer(node):
        return isinstance(node, eqx.nn.Linear | eqx.nn.Conv2d)

    return [leaf for leaf in jax.tree.leaves(net, is_leaf=is_layer) if is_layer(leaf)]


@pytest.mark.parametrize("name", list(_NETWORKS))
def test_layers_are_drawn_from_the_fan_in_uniform(name: str):
    """Weights fill U(+/- 1 / sqrt(fan_in)) and biases stay inside it, rather
    than being zero or using another scale."""
    layers = _layers(_NETWORKS[name])
    assert layers

    for layer in layers:
        bound = 1 / math.sqrt(layer.weight[0].size)
        weight, bias = np.asarray(layer.weight), np.asarray(layer.bias)
        assert np.abs(weight).max() <= bound
        assert np.abs(weight).max() > 0.9 * bound
        assert np.abs(bias).max() <= bound
        assert np.abs(bias).max() > 0


# --- preset builder -----------------------------------------------------------

_PRESET_OBS_SHAPES = {
    "mlp": (6,),
    "mlp_ln": (6,),
    "nature_cnn": (36, 36, 4),
    "nature_cnn_ln": (36, 36, 4),
    "minatar_cnn": (10, 10, 4),
    "minatar_cnn_ln": (10, 10, 4),
}


@pytest.mark.parametrize("preset", sorted(_PRESET_OBS_SHAPES))
def test_every_preset_builds_a_q_network(preset: str):
    """Each preset name yields a network giving one Q-value per action."""
    obs_shape = _PRESET_OBS_SHAPES[preset]
    net = build_q_network(preset, obs_shape, 5, 8, _KEY)
    assert net(jnp.zeros(obs_shape)).shape == (5,)


def test_unknown_preset_is_rejected():
    """A misspelled preset fails loudly instead of picking a default."""
    with pytest.raises(ValueError, match="unknown NETWORK_PRESET"):
        build_q_network("mlp_typo", (6,), 5, 8, _KEY)


# --- dual head ----------------------------------------------------------------


def _dual_head_with_random_h(preset: str):
    obs_shape = _PRESET_OBS_SHAPES[preset]
    net = DualHeadNetwork(build_q_network(preset, obs_shape, 5, 8, _KEY))
    weight = jax.random.normal(jax.random.key(1), net.h_out.weight.shape)
    return eqx.tree_at(lambda n: n.h_out.weight, net, weight), obs_shape


@pytest.mark.parametrize("preset", sorted(_PRESET_OBS_SHAPES))
def test_dual_head_starts_with_zero_h_and_unchanged_q(preset: str):
    """h predicts zero everywhere at init, and the Q head is the wrapped
    network itself."""
    obs_shape = _PRESET_OBS_SHAPES[preset]
    q_net = build_q_network(preset, obs_shape, 5, 8, _KEY)
    obs = jax.random.uniform(jax.random.key(2), obs_shape)

    q_values, h_values = DualHeadNetwork(q_net)(obs)

    assert jnp.array_equal(q_values, q_net(obs))
    assert jnp.array_equal(h_values, jnp.zeros(5))


@pytest.mark.parametrize("preset", sorted(_PRESET_OBS_SHAPES))
def test_h_gradient_reaches_only_the_h_head(preset: str):
    """Training h leaves the shared torso and the Q head untouched."""
    net, obs_shape = _dual_head_with_random_h(preset)
    obs = jax.random.uniform(jax.random.key(2), obs_shape)

    grads = eqx.filter_grad(lambda n: jnp.sum(n(obs)[1]))(net)

    q_grads = jax.tree.leaves(grads.q)
    assert q_grads
    assert all(jnp.all(leaf == 0) for leaf in q_grads)
    assert jnp.any(grads.h_out.weight != 0)


@pytest.mark.parametrize("preset", sorted(_PRESET_OBS_SHAPES))
def test_q_gradient_skips_the_h_head(preset: str):
    """Training Q reaches the torso but not the h head."""
    net, obs_shape = _dual_head_with_random_h(preset)
    obs = jax.random.uniform(jax.random.key(2), obs_shape)

    grads = eqx.filter_grad(lambda n: jnp.sum(n(obs)[0]))(net)

    assert all(jnp.all(leaf == 0) for leaf in jax.tree.leaves(grads.h_out))
    torso_grads = eqx.tree_at(lambda q: q.out, grads.q, None)
    assert any(jnp.any(leaf != 0) for leaf in jax.tree.leaves(torso_grads))
