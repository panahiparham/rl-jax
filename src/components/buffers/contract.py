from typing import NamedTuple, Protocol

import jax
from jax.typing import DTypeLike


class TimeStep(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    termination: jax.Array
    truncation: jax.Array
    discount: jax.Array


class Batch(NamedTuple):
    obs: jax.Array
    action: jax.Array
    ret: jax.Array
    discount: jax.Array
    boot_obs: jax.Array
    mask: jax.Array


class ObservationSpace(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> DTypeLike: ...


class Pushed[S, Out](NamedTuple):
    state: S
    out: Out
    out_valid: jax.Array


class Buffer[S, In, Out](Protocol):
    def init(self, observation_space: ObservationSpace, /) -> S: ...

    # An invalid item leaves the state unchanged and reports nothing aged out.
    def push(self, state: S, item: In, valid: jax.Array, /) -> Pushed[S, Out]: ...

    def size(self, state: S, /) -> jax.Array: ...


class SampleableBuffer[S, In, Out](Buffer[S, In, Out], Protocol):
    @property
    def batch_size(self) -> int: ...

    def sample(self, state: S, key: jax.Array, /) -> Batch: ...

    def can_sample(self, state: S, /) -> jax.Array: ...
