"""Golden values pinning every Q network's initial parameters and outputs.

The digests and outputs were recorded before the networks were restructured;
a refactor must reproduce them exactly from the same key.
"""

from __future__ import annotations

import hashlib

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

_KEY = jax.random.key(7)
_MLP_DIGEST = "cab249a17c3d791bad7d7dff61a322584e38cea18c71cfd00ef4e459dc09e3fb"
_NATURE_DIGEST = "f9ff6ae5ac6e9135e4bb73ce42f1328b90b6436eaf1415fd6dbbc006987f0eaf"
_MINATAR_DIGEST = "c7cb9962f4c9276bbb39b641e04108fe7d6a9bec1ba51bc5ffe74e66cb6490bf"

_GOLDEN = {
    "mlp": (
        lambda: QNetwork(6, 4, 16, _KEY),
        (6,),
        _MLP_DIGEST,
        [0.239677, -0.145461, -0.0810494, -0.235757],
    ),
    "mlp_ln": (
        lambda: QNetworkLN(6, 4, 16, _KEY),
        (6,),
        _MLP_DIGEST,
        [0.404917, 0.23066, 0.0895011, -0.546741],
    ),
    "nature_cnn": (
        lambda: NatureCNN((36, 36, 4), 6, _KEY),
        (36, 36, 4),
        _NATURE_DIGEST,
        [-0.0314351, -0.0315781, 0.0703174, -0.0234301, 0.0389371, -0.0023361],
    ),
    "nature_cnn_ln": (
        lambda: NatureCNNLN((36, 36, 4), 6, _KEY),
        (36, 36, 4),
        _NATURE_DIGEST,
        [-0.446573, 0.0332801, 0.892748, -0.744118, 0.185906, 0.403762],
    ),
    "minatar_cnn": (
        lambda: MinAtarCNN((10, 10, 4), 3, _KEY),
        (10, 10, 4),
        _MINATAR_DIGEST,
        [0.106085, -0.0464029, 0.00605495],
    ),
    "minatar_cnn_ln": (
        lambda: MinAtarCNNLN((10, 10, 4), 3, _KEY),
        (10, 10, 4),
        _MINATAR_DIGEST,
        [0.317858, -0.114571, 0.34299],
    ),
}


def _digest(net: eqx.Module):
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(eqx.filter(net, eqx.is_array)):
        array = np.asarray(leaf)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("name", sorted(_GOLDEN))
def test_initial_parameters_match_golden(name: str):
    """The same key yields bit-identical parameters, in the same leaf order.

    LayerNorm variants carry no parameters, so they share their plain
    counterpart's digest.
    """
    build, _, digest, _ = _GOLDEN[name]
    assert _digest(build()) == digest


@pytest.mark.parametrize("name", sorted(_GOLDEN))
def test_initial_outputs_match_golden(name: str):
    """A fixed input maps to the recorded Q-values."""
    build, obs_shape, _, outputs = _GOLDEN[name]
    obs = jax.random.uniform(jax.random.key(1), obs_shape)
    assert jnp.allclose(build()(obs), jnp.asarray(outputs), rtol=1e-5, atol=1e-7)
