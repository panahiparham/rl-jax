from typing import NamedTuple

import jax
import jax.numpy as jnp

from components.buffers.contract import Batch, ObservationSpace, Pushed, TimeStep


class SelectorState(NamedTuple):
    obs: jax.Array
    action: jax.Array
    ret: jax.Array
    weight: jax.Array
    horizon: jax.Array
    active: jax.Array
    closed: jax.Array


def _select(
    pred: jax.Array, on_true: SelectorState, on_false: SelectorState
) -> SelectorState:
    return SelectorState(
        *(jnp.where(pred, a, b) for a, b in zip(on_true, on_false, strict=True))
    )


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

    def push(
        self, state: SelectorState, step: TimeStep, valid: jax.Array
    ) -> Pushed[SelectorState, Batch]:
        # A chunk leaves once the step after its last reward arrives, so that
        # step supplies the bootstrap and at most one chunk leaves per step.
        emit = state.active & (
            (state.horizon == self._n_step) | state.closed | step.truncation
        )
        out = Batch(
            obs=state.obs,
            action=state.action,
            ret=state.ret,
            discount=state.weight,
            boot_obs=step.obs,
            boot_action=step.action,
            mask=jnp.asarray(True),
        )

        start = emit | ~state.active
        weight = jnp.where(start, 1.0, state.weight)
        accumulated = SelectorState(
            obs=jnp.where(start, step.obs, state.obs),
            action=jnp.where(start, step.action, state.action),
            ret=jnp.where(start, 0.0, state.ret) + weight * step.reward,
            weight=weight * self._gamma * step.discount,
            horizon=jnp.where(start, 0, state.horizon) + 1,
            active=jnp.asarray(True),
            closed=step.termination | (step.discount == 0),
        )
        # The truncating step's successor belongs to the next episode.
        ended = accumulated._replace(active=jnp.asarray(False))
        advanced = _select(step.truncation, ended, accumulated)
        return Pushed(_select(valid, advanced, state), out, valid & emit)
