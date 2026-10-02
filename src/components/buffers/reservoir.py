from typing import NamedTuple

import jax
import jax.numpy as jnp

from components.buffers.contract import Batch, ObservationSpace, Pushed


class ReservoirState(NamedTuple):
    data: Batch
    seen: jax.Array
    key: jax.Array
    # Where the next item goes; ``capacity`` or more drops it.
    slot: jax.Array
    # The transition stored at ``slot``.
    outgoing: Batch


class ReservoirBuffer:
    def __init__(self, *, capacity: int, batch_size: int) -> None:
        self._capacity = capacity
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def init(self, observation_space: ObservationSpace) -> ReservoirState:
        obs_shape = (self._capacity, *observation_space.shape)
        data = Batch(
            obs=jnp.zeros(obs_shape, observation_space.dtype),
            action=jnp.zeros((self._capacity,), jnp.int32),
            ret=jnp.zeros((self._capacity,), jnp.float32),
            discount=jnp.zeros((self._capacity,), jnp.float32),
            boot_obs=jnp.zeros(obs_shape, observation_space.dtype),
            boot_action=jnp.zeros((self._capacity,), jnp.int32),
            mask=jnp.zeros((self._capacity,), jnp.bool_),
        )
        return ReservoirState(
            data=data,
            seen=jnp.asarray(0, jnp.int32),
            key=jax.random.key(0),
            slot=jnp.asarray(0, jnp.int32),
            outgoing=Batch(*(stored[0] for stored in data)),
        )

    def push(
        self, state: ReservoirState, item: Batch, valid: jax.Array
    ) -> Pushed[ReservoirState, Batch]:
        write = valid & (state.slot < self._capacity)
        slot = jnp.minimum(state.slot, self._capacity - 1)
        data = Batch(
            *(
                stored.at[slot].set(jnp.where(write, new, stored[slot]))
                for stored, new in zip(state.data, item, strict=True)
            )
        )
        seen = state.seen + valid

        # The first ``capacity`` items fill the buffer in order. Afterwards the
        # i-th item takes a uniform slot in [0, i] and is dropped past the end,
        # so every item seen so far is kept with equal probability.
        # Each draw is keyed by the count of items seen, so an invalid push
        # leaves the state untouched.
        draw_key = jax.random.fold_in(state.key, seen)
        drawn = jax.random.randint(draw_key, (), 0, seen + 1)
        next_slot = jnp.where(seen < self._capacity, seen, drawn)
        next_slot = jnp.where(valid, next_slot, state.slot)
        # Reading the next victim after this write keeps the storage updated in
        # place instead of copied.
        victim = jnp.minimum(next_slot, self._capacity - 1)
        pushed = ReservoirState(
            data=data,
            seen=seen,
            key=state.key,
            slot=next_slot,
            outgoing=Batch(*(stored[victim] for stored in data)),
        )
        out_valid = write & (state.seen >= self._capacity)
        return Pushed(pushed, state.outgoing, out_valid)

    def size(self, state: ReservoirState) -> jax.Array:
        return jnp.minimum(state.seen, self._capacity)

    def sample(self, state: ReservoirState, key: jax.Array) -> Batch:
        slots = jax.random.randint(key, (self._batch_size,), 0, self.size(state))
        return Batch(*(stored[slots] for stored in state.data))

    def can_sample(self, state: ReservoirState) -> jax.Array:
        return self.size(state) > 0
