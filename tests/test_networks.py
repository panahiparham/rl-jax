"""``QNetworkLN``/``NatureCNNLN``: same shapes as their plain counterparts, and
the LayerNorm layers carry no learnable gain or bias - the whole point of the
"no-affine" variant is otherwise silently regressable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from components import NatureCNNLN, QNetworkLN


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
