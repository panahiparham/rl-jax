from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True, kw_only=True)
class RandomConfig:
    TOTAL_TIMESTEPS: int = 100_000


class RandomState(NamedTuple):
    action_dim: jax.Array


class RandomAgent:
    def __init__(self, config: RandomConfig):
        self._config = config

    def init(self, key: jax.Array, observation_space, action_space):
        del key, observation_space
        return RandomState(action_dim=jnp.asarray(action_space.n, jnp.int32))

    def act(self, state: RandomState, key: jax.Array, obs: jax.Array):
        del obs
        return jax.random.randint(key, (), 0, state.action_dim, dtype=jnp.int32)

    def update(
        self,
        state: RandomState,
        key: jax.Array,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        termination: jax.Array,
        truncation: jax.Array,
        discount: jax.Array,
    ):
        del key, obs, action, reward, termination, truncation, discount
        return state
