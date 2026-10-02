from collections.abc import Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from components.buffers.contract import (
    Batch,
    Buffer,
    ObservationSpace,
    Pushed,
    SampleableBuffer,
    TimeStep,
)

# Each buffer in a chain has its own state and item types.
type ChainedBuffer = Buffer[Any, Any, Any]


def _rows(batch: Batch, start: int, stop: int) -> Batch:
    return Batch(*(field[start:stop] for field in batch))


def _choose(available: jax.Array, own: Batch, substitute: Batch) -> Batch:
    return Batch(
        *(
            jnp.where(available, a, b)
            for a, b in zip(own, substitute, strict=True)
        )
    )


class ComposedState(NamedTuple):
    states: tuple[object, ...]


class ComposedBuffer:
    def __init__(self, buffers: Sequence[ChainedBuffer]) -> None:
        sampled = [
            (index, buffer)
            for index, buffer in enumerate(buffers)
            if isinstance(buffer, SampleableBuffer)
        ]
        if not sampled:
            raise ValueError("a composed buffer needs a sampleable buffer")
        (_, fallback), *others = sampled
        if isinstance(fallback, ComposedBuffer):
            raise TypeError("the first sampleable buffer cannot be composed")
        self._fallback_share = fallback.batch_size - sum(
            other.batch_size for _, other in others
        )
        if self._fallback_share <= 0:
            raise ValueError(
                "the first sampleable buffer's batch size must exceed the "
                "other buffers' total"
            )
        self._buffers = tuple(buffers)
        self._fallback = fallback
        self._sampled = tuple(sampled)

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

    def can_sample(self, state: ComposedState) -> jax.Array:
        return self._fallback.can_sample(state.states[self._sampled[0][0]])

    def sample(self, state: ComposedState, key: jax.Array) -> Batch:
        (fallback, _), *others = self._draw(state, key)
        # Rows past the fallback's own share stand in for any buffer that
        # cannot sample yet.
        offset = self._fallback_share
        parts = [_rows(fallback, 0, offset)]
        for batch, available in others:
            stop = offset + len(batch.action)
            parts.append(_choose(available, batch, _rows(fallback, offset, stop)))
            offset = stop
        return Batch(*(jnp.concatenate(fields) for fields in zip(*parts, strict=True)))

    def sample_components(
        self, state: ComposedState, key: jax.Array
    ) -> tuple[Batch, ...]:
        (fallback, _), *others = self._draw(state, key)
        offset = self._fallback_share
        fallback_mask = [fallback.mask[:offset]]
        components = []
        for batch, available in others:
            stop = offset + len(batch.action)
            fallback_mask.append(fallback.mask[offset:stop] & ~available)
            components.append(batch._replace(mask=batch.mask & available))
            offset = stop
        return (fallback._replace(mask=jnp.concatenate(fallback_mask)), *components)

    def _draw(
        self, state: ComposedState, key: jax.Array
    ) -> list[tuple[Batch, jax.Array]]:
        keys = jax.random.split(key, len(self._sampled))
        return [
            (
                buffer.sample(state.states[index], buffer_key),
                buffer.can_sample(state.states[index]),
            )
            for (index, buffer), buffer_key in zip(self._sampled, keys, strict=True)
        ]
