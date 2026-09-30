from typing import NamedTuple

import jax
import jax.numpy as jnp

from components.buffers.contract import ObservationSpace


class SelectorState(NamedTuple):
    obs: jax.Array
    action: jax.Array
    ret: jax.Array
    weight: jax.Array
    horizon: jax.Array
    active: jax.Array
    closed: jax.Array


class NStepSelector:
    def __init__(self, *, n_step: int, gamma: float) -> None:
        self._n_step = n_step
        self._gamma = gamma

    def init(self, observation_space: ObservationSpace) -> SelectorState:
        return SelectorState(
            obs=jnp.zeros(observation_space.shape, observation_space.dtype),
            action=jnp.asarray(0, jnp.int32),
            ret=jnp.asarray(0.0, jnp.float32),
            weight=jnp.asarray(1.0, jnp.float32),
            horizon=jnp.asarray(0, jnp.int32),
            active=jnp.asarray(False),
            closed=jnp.asarray(False),
        )

    def size(self, state: SelectorState) -> jax.Array:
        return jnp.asarray(0, jnp.int32)
