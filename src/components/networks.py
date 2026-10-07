import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jss


def _fan_in_uniform[Layer: (eqx.nn.Linear, eqx.nn.Conv2d)](
    layer: Layer, key: jax.Array
) -> Layer:
    """Nature DQN init: weights and biases ~ U(+/- 1 / sqrt(fan_in))."""
    out_features, *fan_in_dims = layer.weight.shape
    bound = 1 / math.sqrt(math.prod(fan_in_dims))
    bias_shape = (out_features,) + (1,) * (len(fan_in_dims) - 1)
    wkey, bkey = jax.random.split(key)
    weight = jax.random.uniform(wkey, layer.weight.shape, minval=-bound, maxval=bound)
    bias = jax.random.uniform(bkey, bias_shape, minval=-bound, maxval=bound)
    return eqx.tree_at(lambda lay: (lay.weight, lay.bias), layer, (weight, bias))


def _conv(
    in_channels: int, out_channels: int, kernel_size: int, stride: int, key: jax.Array
) -> eqx.nn.Conv2d:
    conv = eqx.nn.Conv2d(in_channels, out_channels, kernel_size, stride, key=key)
    return _fan_in_uniform(conv, key)


def _linear(in_features: int, out_features: int, key: jax.Array) -> eqx.nn.Linear:
    return _fan_in_uniform(eqx.nn.Linear(in_features, out_features, key=key), key)


class QNetwork(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    layer3: eqx.nn.Linear

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layer1 = _linear(obs_dim, hidden_size, key=k1)
        self.layer2 = _linear(hidden_size, hidden_size, key=k2)
        self.layer3 = _linear(hidden_size, action_dim, key=k3)

    def __call__(self, x: jax.Array):
        x = jnp.ravel(x)
        x = jax.nn.relu(self.layer1(x))
        x = jax.nn.relu(self.layer2(x))
        return self.layer3(x)


class QNetworkLN(eqx.Module):
    """QNetwork with a no-affine LayerNorm after each hidden layer."""

    layer1: eqx.nn.Linear
    ln1: eqx.nn.LayerNorm
    layer2: eqx.nn.Linear
    ln2: eqx.nn.LayerNorm
    layer3: eqx.nn.Linear

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layer1 = _linear(obs_dim, hidden_size, key=k1)
        self.ln1 = eqx.nn.LayerNorm(hidden_size, use_weight=False, use_bias=False)
        self.layer2 = _linear(hidden_size, hidden_size, key=k2)
        self.ln2 = eqx.nn.LayerNorm(hidden_size, use_weight=False, use_bias=False)
        self.layer3 = _linear(hidden_size, action_dim, key=k3)

    def __call__(self, x: jax.Array):
        x = jnp.ravel(x)
        x = jax.nn.relu(self.ln1(self.layer1(x)))
        x = jax.nn.relu(self.ln2(self.layer2(x)))
        return self.layer3(x)


class QNetworkHL(eqx.Module):
    """QNetwork with histogram loss and no-affine LayerNorm after each hidden layer."""

    # Model layers
    layers: list

    # Histogram params
    action_dim: int = eqx.field(static=True)
    num_bins: int = eqx.field(static=True)
    histogram_support_min: float = eqx.field(static=True) # a
    histogram_support_max: float = eqx.field(static=True) # b
    histogram_sigma: float = eqx.field(static=True)
    histogram_bin_width: float = eqx.field(static=True)


    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int, key: jax.Array, num_bins: int, histogram_support_min: float, histogram_support_max: float, sigma_ratio: float):
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.histogram_support_min = histogram_support_min
        self.histogram_support_max = histogram_support_max
        self.histogram_bin_width = (histogram_support_max - histogram_support_min) / num_bins
        self.histogram_sigma = sigma_ratio * self.histogram_bin_width
        self.init_layers(obs_dim, action_dim, hidden_size, key)

    def init_layers(self, obs_dim: int, action_dim: int, hidden_size: int, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [
            _linear(obs_dim, hidden_size, key=k1),
            eqx.nn.Lambda(jax.nn.relu),
            _linear(hidden_size, hidden_size, key=k2),
            eqx.nn.Lambda(jax.nn.relu),
            _linear(hidden_size, action_dim * self.num_bins, key=k3),
        ]


    def __call__(self, x: jax.Array):
        x = jnp.ravel(x)
        for layer in self.layers:
            x = layer(x)

        # Raw logits from network
        logits = jnp.reshape(x, (self.action_dim, self.num_bins))
        return logits

    def histogram_bins(self):
        return jnp.linspace(self.histogram_support_min, self.histogram_support_max, self.num_bins + 1)

    def get_histogram_values(self, target: jax.Array) -> jax.Array:
        """
        Computes HL gauss histograms for a single target
        """

        # Code based on HL Gauss equations from page 5 of https://www.jmlr.org/papers/v27/24-0260.html

        # Compute Z
        left = jss.erf((self.histogram_support_max - target) / (jnp.sqrt(2) * self.histogram_sigma))
        right = jss.erf((self.histogram_support_min - target) / (jnp.sqrt(2) * self.histogram_sigma))
        z = 0.5 * (left - right)

        # Compute l (left side of bin)
        bins = self.histogram_bins()[:-1]

        # Compute probability for each bin
        c_left = jss.erf((bins + self.histogram_bin_width - target) / (jnp.sqrt(2) * self.histogram_sigma))
        c_right = jss.erf((bins - target) / (jnp.sqrt(2) * self.histogram_sigma))
        c = (1 / (2 * z)) * (c_left - c_right)

        return c

    def get_action_values(self, logits: jax.Array) -> jax.Array:
        bins = self.histogram_bins()
        bin_centres = (bins[:-1] + bins[1:]) / 2
        probabilities = jax.nn.softmax(logits, axis=-1)
        return probabilities @ bin_centres

class QNetworkHLLN(QNetworkHL):
    """QNetwork with histogram loss and no-affine LayerNorm after each hidden layer."""

    def init_layers(self, obs_dim: int, action_dim: int, hidden_size: int, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [
            _linear(obs_dim, hidden_size, key=k1),
            eqx.nn.LayerNorm(hidden_size, use_weight=False, use_bias=False),
            eqx.nn.Lambda(jax.nn.relu),
            _linear(hidden_size, hidden_size, key=k2),
            eqx.nn.LayerNorm(hidden_size, use_weight=False, use_bias=False),
            eqx.nn.Lambda(jax.nn.relu),
            _linear(hidden_size, action_dim * self.num_bins, key=k3),
        ]



def _nature_flat_dim(obs_shape: tuple[int, ...]):
    """Flattened size after the Nature-DQN conv stack, computed host-side."""
    h, w = obs_shape[0], obs_shape[1]
    for kernel, stride in ((8, 4), (4, 2), (3, 1)):
        h = (h - kernel) // stride + 1
        w = (w - kernel) // stride + 1
    return 64 * h * w


class NatureCNN(eqx.Module):
    """Nature-DQN conv torso + 512 head for channel-last image observations."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    head: eqx.nn.Linear
    out: eqx.nn.Linear

    def __init__(self, obs_shape: tuple[int, ...], action_dim: int, key: jax.Array):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        channels = obs_shape[-1]
        self.conv1 = _conv(channels, 32, kernel_size=8, stride=4, key=k1)
        self.conv2 = _conv(32, 64, kernel_size=4, stride=2, key=k2)
        self.conv3 = _conv(64, 64, kernel_size=3, stride=1, key=k3)
        self.head = _linear(_nature_flat_dim(obs_shape), 512, key=k4)
        self.out = _linear(512, action_dim, key=k5)

    def __call__(self, x: jax.Array):
        # (H,W,C) uint8 -> (C,H,W) float in [0,1]
        x = jnp.transpose(x, (2, 0, 1)).astype(jnp.float32) / 255.0
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        x = jax.nn.relu(self.head(jnp.ravel(x)))
        return self.out(x)


class NatureCNNLN(eqx.Module):
    """NatureCNN with a no-affine LayerNorm after the dense head, before its relu.

    The conv torso is left unnormalized; only the head follows QNetworkLN's
    "every hidden layer, not the output" convention.
    """

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    head: eqx.nn.Linear
    head_ln: eqx.nn.LayerNorm
    out: eqx.nn.Linear

    def __init__(self, obs_shape: tuple[int, ...], action_dim: int, key: jax.Array):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        channels = obs_shape[-1]
        self.conv1 = _conv(channels, 32, kernel_size=8, stride=4, key=k1)
        self.conv2 = _conv(32, 64, kernel_size=4, stride=2, key=k2)
        self.conv3 = _conv(64, 64, kernel_size=3, stride=1, key=k3)
        self.head = _linear(_nature_flat_dim(obs_shape), 512, key=k4)
        self.head_ln = eqx.nn.LayerNorm(512, use_weight=False, use_bias=False)
        self.out = _linear(512, action_dim, key=k5)

    def __call__(self, x: jax.Array):
        # (H,W,C) uint8 -> (C,H,W) float in [0,1]
        x = jnp.transpose(x, (2, 0, 1)).astype(jnp.float32) / 255.0
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        x = jax.nn.relu(self.head_ln(self.head(jnp.ravel(x))))
        return self.out(x)


def _minatar_flat_dim(obs_shape: tuple[int, ...]):
    """Flattened size after the 3x3 valid conv with 16 filters."""
    return (obs_shape[0] - 2) * (obs_shape[1] - 2) * 16


class MinAtarCNN(eqx.Module):
    """MinAtar DQN torso: one 3x3 conv with 16 filters, then a 128 unit head."""

    conv: eqx.nn.Conv2d
    head: eqx.nn.Linear
    out: eqx.nn.Linear

    def __init__(self, obs_shape: tuple[int, ...], action_dim: int, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.conv = _conv(obs_shape[-1], 16, kernel_size=3, stride=1, key=k1)
        self.head = _linear(_minatar_flat_dim(obs_shape), 128, key=k2)
        self.out = _linear(128, action_dim, key=k3)

    def __call__(self, x: jax.Array):
        # (H,W,C) bool -> (C,H,W) float
        x = jnp.transpose(x, (2, 0, 1)).astype(jnp.float32)
        x = jax.nn.relu(self.conv(x))
        x = jax.nn.relu(self.head(jnp.ravel(x)))
        return self.out(x)


class MinAtarCNNLN(eqx.Module):
    """MinAtarCNN with a no-affine LayerNorm after the dense head, before its relu.

    The conv layer is left unnormalized, following NatureCNNLN's convention.
    """

    conv: eqx.nn.Conv2d
    head: eqx.nn.Linear
    head_ln: eqx.nn.LayerNorm
    out: eqx.nn.Linear

    def __init__(self, obs_shape: tuple[int, ...], action_dim: int, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.conv = _conv(obs_shape[-1], 16, kernel_size=3, stride=1, key=k1)
        self.head = _linear(_minatar_flat_dim(obs_shape), 128, key=k2)
        self.head_ln = eqx.nn.LayerNorm(128, use_weight=False, use_bias=False)
        self.out = _linear(128, action_dim, key=k3)

    def __call__(self, x: jax.Array):
        # (H,W,C) bool -> (C,H,W) float
        x = jnp.transpose(x, (2, 0, 1)).astype(jnp.float32)
        x = jax.nn.relu(self.conv(x))
        x = jax.nn.relu(self.head_ln(self.head(jnp.ravel(x))))
        return self.out(x)
