from typing import NamedTuple

import jax
import jax.numpy as jnp

from components.buffers.contract import ObservationSpace, Pushed


class SubsampleState(NamedTuple):
    seen: jax.Array


class Subsample:
    def __init__(self, *, every: int) -> None:
        self._every = every

    def init(self, observation_space: ObservationSpace) -> SubsampleState:
        return SubsampleState(seen=jnp.asarray(0, jnp.int32))

    def push[T](
        self, state: SubsampleState, item: T, valid: jax.Array
    ) -> Pushed[SubsampleState, T]:
        keep = valid & (state.seen % self._every == 0)
        return Pushed(SubsampleState(state.seen + valid), item, keep)

    def size(self, state: SubsampleState) -> jax.Array:
        return jnp.asarray(0, jnp.int32)
