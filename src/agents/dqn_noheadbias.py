from dataclasses import dataclass

import equinox as eqx
import jax

from agents.dqn import DQNAgent, DQNConfig
from components import NatureCNN, NatureCNNLN, QNetwork, QNetworkLN


@dataclass(frozen=True, kw_only=True)
class DQNNoHeadBiasConfig(DQNConfig):
    """DQN's hyperparameters; the Q-value head has no bias."""


def _without_bias(linear: eqx.nn.Linear):
    # The key is unused: the weight is replaced by the original one.
    unbiased = eqx.nn.Linear(
        linear.in_features,
        linear.out_features,
        use_bias=False,
        key=jax.random.key(0),
    )
    return eqx.tree_at(lambda layer: layer.weight, unbiased, linear.weight)


class DQNNoHeadBiasAgent(DQNAgent):
    def _build_q(
        self, key: jax.Array, obs_shape: tuple[int, ...], action_dim: int
    ) -> eqx.Module:
        q = super()._build_q(key, obs_shape, action_dim)
        if isinstance(q, QNetwork | QNetworkLN):
            return eqx.tree_at(lambda net: net.layer3, q, _without_bias(q.layer3))
        if isinstance(q, NatureCNN | NatureCNNLN):
            return eqx.tree_at(lambda net: net.out, q, _without_bias(q.out))
        raise TypeError(f"no known Q-value head on {type(q).__name__}")
