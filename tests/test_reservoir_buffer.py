from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from components import Batch
from components.buffers.reservoir import ReservoirBuffer, ReservoirState


@dataclass(frozen=True)
class _Space:
    shape: tuple[int, ...] = (1,)
    dtype: type = jnp.float32


def _transition(index: int | jax.Array) -> Batch:
    index = jnp.asarray(index, jnp.int32)
    return Batch(
        obs=jnp.asarray([index], jnp.float32),
        action=index,
        ret=jnp.asarray(0.0, jnp.float32),
        discount=jnp.asarray(0.9, jnp.float32),
        boot_obs=jnp.asarray([index + 1], jnp.float32),
        boot_action=index + 1,
        mask=jnp.asarray(True),
    )


def _push_all(
    buffer: ReservoirBuffer, indices: list[int], valid: list[bool] | None = None
) -> tuple[ReservoirState, list[int]]:
    state = buffer.init(_Space())
    evicted = []
    for position, index in enumerate(indices):
        is_valid = True if valid is None else valid[position]
        pushed = buffer.push(state, _transition(index), jnp.asarray(is_valid))
        state = pushed.state
        if bool(pushed.out_valid):
            evicted.append(int(pushed.out.action))
    return state, evicted


def _stored(buffer: ReservoirBuffer, state: ReservoirState) -> list[int]:
    return np.asarray(state.data.action[: int(buffer.size(state))]).tolist()


# Filling and replacing


def test_first_items_fill_the_buffer_in_order():
    """Until it is full, every item is kept and nothing is evicted."""
    buffer = ReservoirBuffer(capacity=5, batch_size=2)

    state, evicted = _push_all(buffer, [0, 1, 2, 3, 4])

    assert _stored(buffer, state) == [0, 1, 2, 3, 4]
    assert evicted == []


def test_size_stays_at_capacity_once_full():
    """Later items replace stored ones or are dropped; the size never grows."""
    buffer = ReservoirBuffer(capacity=4, batch_size=2)

    state, _ = _push_all(buffer, list(range(50)))

    assert int(buffer.size(state)) == 4
    assert int(state.seen) == 50


def test_every_eviction_reports_the_replaced_item():
    """An evicted item was stored before the push and is gone after it."""
    buffer = ReservoirBuffer(capacity=3, batch_size=2)
    state = buffer.init(_Space())
    for index in range(30):
        before = set(_stored(buffer, state))
        pushed = buffer.push(state, _transition(index), jnp.asarray(True))
        state = pushed.state
        after = set(_stored(buffer, state))
        if bool(pushed.out_valid):
            assert int(pushed.out.action) in before - after
            assert index in after
        else:
            assert before <= after


def test_invalid_push_leaves_the_buffer_unchanged():
    """An invalid item is not stored, evicts nothing, and draws no slot."""
    buffer = ReservoirBuffer(capacity=3, batch_size=2)
    clean_state, clean_evicted = _push_all(buffer, list(range(12)))

    mixed = [0, 1, 2, 99, 3, 4, 98, 5, 6, 7, 8, 9, 10, 11]
    valid = [index < 90 for index in mixed]
    mixed_state, mixed_evicted = _push_all(buffer, mixed, valid)

    assert mixed_evicted == clean_evicted
    for clean, mixed_leaf in zip(
        jax.tree.leaves(clean_state._replace(key=None)),
        jax.tree.leaves(mixed_state._replace(key=None)),
        strict=True,
    ):
        assert np.array_equal(clean, mixed_leaf)


def test_items_are_kept_with_equal_probability():
    """After 20 items, a 4-slot reservoir holds each one with probability 0.2.

    Checked over 2,000 seeded reservoirs; the tolerance is about four
    standard errors.
    """
    capacity, num_items, num_keys = 4, 20, 2000
    buffer = ReservoirBuffer(capacity=capacity, batch_size=1)

    def fill(key: jax.Array) -> jax.Array:
        state = buffer.init(_Space())._replace(key=key)

        def one(state: ReservoirState, index: jax.Array):
            return buffer.push(state, _transition(index), jnp.asarray(True)).state, None

        state, _ = jax.lax.scan(one, state, jnp.arange(num_items))
        kept = jnp.zeros(num_items, jnp.bool_).at[state.data.action].set(True)
        return kept

    keys = jax.random.split(jax.random.key(0), num_keys)
    kept = np.asarray(jax.jit(jax.vmap(fill))(keys))

    assert kept.sum(axis=1).tolist() == [capacity] * num_keys
    assert np.allclose(kept.mean(axis=0), capacity / num_items, atol=0.04)


# Sampling


def test_samples_cover_exactly_the_stored_transitions():
    """Samples draw from every stored transition and nothing else."""
    buffer = ReservoirBuffer(capacity=3, batch_size=4)
    state, _ = _push_all(buffer, list(range(10)))

    sample = jax.jit(jax.vmap(buffer.sample, in_axes=(None, 0)))
    batches = sample(state, jax.random.split(jax.random.key(0), 64))

    sampled = set(np.asarray(batches.action).reshape(-1).tolist())
    assert sampled == set(_stored(buffer, state))


def test_can_sample_once_any_transition_is_stored():
    """Sampling is allowed from the first stored transition, as in the original."""
    buffer = ReservoirBuffer(capacity=3, batch_size=4)

    assert not bool(buffer.can_sample(buffer.init(_Space())))
    assert bool(buffer.can_sample(_push_all(buffer, [0])[0]))
