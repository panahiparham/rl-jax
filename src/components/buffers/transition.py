from typing import NamedTuple

import jax
import jax.numpy as jnp

from components.buffers.contract import Batch, ObservationSpace, Pushed


class TransitionState(NamedTuple):
    data: Batch
    head: jax.Array
    size: jax.Array
    # The transition that the next write overwrites.
    outgoing: Batch


class TransitionBuffer:
    def __init__(self, *, capacity: int, batch_size: int) -> None:
        self._capacity = capacity
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def init(self, observation_space: ObservationSpace) -> TransitionState:
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
        return TransitionState(
            data=data,
            head=jnp.asarray(0, jnp.int32),
            size=jnp.asarray(0, jnp.int32),
            outgoing=Batch(*(stored[0] for stored in data)),
        )

    def push(
        self, state: TransitionState, item: Batch, valid: jax.Array
    ) -> Pushed[TransitionState, Batch]:
        slot = state.head
        data = Batch(
            *(
                stored.at[slot].set(jnp.where(valid, new, stored[slot]))
                for stored, new in zip(state.data, item, strict=True)
            )
        )
        head = jnp.where(valid, (slot + 1) % self._capacity, slot)
        pushed = TransitionState(
            data=data,
            head=head,
            size=jnp.where(
                valid, jnp.minimum(state.size + 1, self._capacity), state.size
            ),
            # Reading the next victim after this write keeps the ring updated
            # in place instead of copied.
            outgoing=Batch(*(stored[head] for stored in data)),
        )
        out_valid = valid & (state.size == self._capacity)
        return Pushed(pushed, state.outgoing, out_valid)

    def size(self, state: TransitionState) -> jax.Array:
        return state.size

    def sample(self, state: TransitionState, key: jax.Array) -> Batch:
        slots = jax.random.randint(key, (self._batch_size,), 0, state.size)
        return Batch(*(stored[slots] for stored in state.data))

    def can_sample(self, state: TransitionState) -> jax.Array:
        return state.size > 0
