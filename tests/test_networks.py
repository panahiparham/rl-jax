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
    MinAtarCNN,
    MinAtarCNNLN,
    NatureCNN,
    NatureCNNLN,
    QNetwork,
    QNetworkLN,
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
