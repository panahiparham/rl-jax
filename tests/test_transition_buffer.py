from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from components import Batch
from components.buffers.transition import TransitionBuffer, TransitionState


@dataclass(frozen=True)
class _Space:
    shape: tuple[int, ...] = (2,)
    dtype: type = jnp.float32


def _transition(index: int) -> Batch:
    return Batch(
        obs=jnp.asarray([index, index], jnp.float32),
        action=jnp.asarray(index, jnp.int32),
        ret=jnp.asarray(index * 0.5, jnp.float32),
        discount=jnp.asarray(0.9, jnp.float32),
        boot_obs=jnp.asarray([index + 1, index + 1], jnp.float32),
        boot_action=jnp.asarray(index + 1, jnp.int32),
        mask=jnp.asarray(True),
    )


def _push_all(
    buffer: TransitionBuffer, indices: list[int], valid: list[bool] | None = None
) -> tuple[TransitionState, list[int]]:
    state = buffer.init(_Space())
    evicted = []
    for position, index in enumerate(indices):
        is_valid = True if valid is None else valid[position]
        pushed = buffer.push(state, _transition(index), jnp.asarray(is_valid))
        state = pushed.state
        if bool(pushed.out_valid):
            evicted.append(int(pushed.out.action))
    return state, evicted


def _sampled_actions(buffer: TransitionBuffer, state: TransitionState) -> set[int]:
    sample = jax.jit(jax.vmap(buffer.sample, in_axes=(None, 0)))
    batches = sample(state, jax.random.split(jax.random.key(0), 64))
    return set(np.asarray(batches.action).reshape(-1).tolist())


# Writes and evictions


def test_full_buffer_evicts_the_oldest_transition_per_write():
    """Once full, each write hands back the transition it overwrites."""
    buffer = TransitionBuffer(capacity=3, batch_size=2)

    state, evicted = _push_all(buffer, [0, 1, 2, 3, 4])

    assert evicted == [0, 1]
    assert int(buffer.size(state)) == 3


def test_invalid_push_leaves_the_buffer_unchanged():
    """An invalid transition is not stored and evicts nothing."""
    buffer = TransitionBuffer(capacity=3, batch_size=2)
    clean_state, clean_evicted = _push_all(buffer, [0, 1, 2, 3])

    mixed_state, mixed_evicted = _push_all(
        buffer, [0, 1, 99, 2, 98, 3], valid=[True, True, False, True, False, True]
    )

    assert mixed_evicted == clean_evicted
    for clean, mixed in zip(
        jax.tree.leaves(clean_state), jax.tree.leaves(mixed_state), strict=True
    ):
        assert np.array_equal(clean, mixed)


# Sampling


@pytest.mark.parametrize(
    ("indices", "expected"),
    [([0], {0}), ([0, 1], {0, 1}), ([0, 1, 2, 3, 4], {2, 3, 4})],
)
def test_samples_cover_exactly_the_stored_transitions(
    indices: list[int], expected: set[int]
):
    """Samples draw from every stored transition and nothing else."""
    buffer = TransitionBuffer(capacity=3, batch_size=4)
    state, _ = _push_all(buffer, indices)

    assert _sampled_actions(buffer, state) == expected


def test_sampled_rows_keep_their_fields_together():
    """Every field of a sampled row comes from the same stored transition."""
    buffer = TransitionBuffer(capacity=4, batch_size=8)
    state, _ = _push_all(buffer, [0, 1, 2, 3, 4, 5])

    batch = buffer.sample(state, jax.random.key(1))

    assert batch.obs.shape == (8, 2)
    action = np.asarray(batch.action)
    assert np.allclose(batch.obs, np.stack([action, action], axis=1))
    assert np.allclose(batch.ret, action * 0.5)
    assert np.array_equal(batch.boot_action, action + 1)


def test_can_sample_once_any_transition_is_stored():
    """Sampling is allowed from the first stored transition, as in the original."""
    buffer = TransitionBuffer(capacity=3, batch_size=4)

    assert not bool(buffer.can_sample(buffer.init(_Space())))
    assert bool(buffer.can_sample(_push_all(buffer, [0])[0]))
