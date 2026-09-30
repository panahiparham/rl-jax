from collections.abc import Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from components.buffers.contract import (
    Buffer,
    ObservationSpace,
    Pushed,
    SampleableBuffer,
    TimeStep,
)

# Each buffer in a chain has its own state and item types.
type ChainedBuffer = Buffer[Any, Any, Any]


class ComposedState(NamedTuple):
    states: tuple[object, ...]


class ComposedBuffer:
    def __init__(self, buffers: Sequence[ChainedBuffer]) -> None:
        sampleable = [
            buffer for buffer in buffers if isinstance(buffer, SampleableBuffer)
        ]
        if not sampleable:
            raise ValueError("a composed buffer needs a sampleable buffer")
        fallback, *others = sampleable
        if isinstance(fallback, ComposedBuffer):
            raise TypeError("the first sampleable buffer cannot be composed")
        if fallback.batch_size <= sum(other.batch_size for other in others):
            raise ValueError(
                "the first sampleable buffer's batch size must exceed the "
                "other buffers' total"
            )
        self._buffers = tuple(buffers)
        self._fallback = fallback

    @property
    def batch_size(self) -> int:
        return self._fallback.batch_size

    def init(self, observation_space: ObservationSpace) -> ComposedState:
        return ComposedState(
            tuple(buffer.init(observation_space) for buffer in self._buffers)
        )

    def push(
        self, state: ComposedState, item: object, valid: jax.Array
    ) -> Pushed[ComposedState, object]:
        states = []
        for buffer, buffer_state in zip(self._buffers, state.states, strict=True):
            pushed = buffer.push(buffer_state, item, valid)
            states.append(pushed.state)
            item, valid = pushed.out, pushed.out_valid
        return Pushed(ComposedState(tuple(states)), item, valid)

    def add(
        self,
        state: ComposedState,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        termination: jax.Array,
        truncation: jax.Array,
        discount: jax.Array,
    ) -> ComposedState:
        step = TimeStep(obs, action, reward, termination, truncation, discount)
        return self.push(state, step, jnp.asarray(True)).state

    def size(self, state: ComposedState) -> jax.Array:
        sizes = [
            buffer.size(buffer_state)
            for buffer, buffer_state in zip(self._buffers, state.states, strict=True)
        ]
        return jnp.sum(jnp.stack(sizes))
