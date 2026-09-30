from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from components import BufferState, ReplayBuffer
from components.buffers.composed import ChainedBuffer, ComposedBuffer, ComposedState
from components.buffers.selector import NStepSelector
from components.buffers.transition import TransitionBuffer, TransitionState


@dataclass(frozen=True)
class _Space:
    shape: tuple[int, ...] = (1,)
    dtype: type = jnp.float32


def _recency(batch_size: int = 4) -> ReplayBuffer:
    return ReplayBuffer(capacity=4, batch_size=batch_size, n_step=1, gamma=0.9)


def _long_term(capacity: int = 3) -> TransitionBuffer:
    return TransitionBuffer(capacity=capacity, batch_size=1)


def _add_steps(buffer: ComposedBuffer, num_steps: int) -> ComposedState:
    state = buffer.init(_Space())
    for step in range(num_steps):
        state = buffer.add(
            state,
            jnp.asarray([step], jnp.float32),
            jnp.asarray(step, jnp.int32),
            jnp.asarray(1.0, jnp.float32),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(1.0, jnp.float32),
        )
    return state


# Construction


@pytest.mark.parametrize(
    ("buffers", "error"),
    [
        ([NStepSelector(n_step=2, gamma=0.9)], ValueError),
        ([_recency(batch_size=1), _long_term()], ValueError),
        ([ComposedBuffer([_recency()]), _long_term()], TypeError),
    ],
    ids=["no-sampleable-buffer", "fallback-too-small", "composed-fallback"],
)
def test_construction_rejects_layouts_without_a_usable_fallback(
    buffers: list[ChainedBuffer], error: type[Exception]
):
    """The first sampleable buffer must be plain and cover the other buffers."""
    with pytest.raises(error):
        ComposedBuffer(buffers)


def test_batch_size_is_the_fallback_batch_size():
    """The fallback's batch size is the total rows a composed sample returns."""
    buffer = ComposedBuffer(
        [_recency(batch_size=4), NStepSelector(n_step=2, gamma=0.9), _long_term()]
    )

    assert buffer.batch_size == 4


# Chaining


def test_aged_out_timesteps_reach_the_last_buffer_as_chunks():
    """Timesteps leave recency in order and arrive downstream as 2-step chunks.

    After twelve steps, timesteps 0 through 7 have left the 4-slot recency
    buffer, so the chunks starting at 0, 2, and 4 are complete; the chunk at
    6 still waits for its bootstrap.
    """
    buffer = ComposedBuffer(
        [_recency(), NStepSelector(n_step=2, gamma=0.9), _long_term(capacity=5)]
    )

    state = _add_steps(buffer, 12)

    recency_state, _, long_term_state = state.states
    assert isinstance(recency_state, BufferState)
    assert isinstance(long_term_state, TransitionState)
    stored = long_term_state.data
    size = int(long_term_state.size)
    assert int(recency_state.size) == 4
    assert np.asarray(stored.action[:size]).tolist() == [0, 2, 4]
    assert np.asarray(stored.boot_action[:size]).tolist() == [2, 4, 6]
    assert int(buffer.size(state)) == 4 + 3


def test_nested_composition_matches_the_flat_chain():
    """A composed buffer used as a stage stores what the flat chain stores."""
    flat = ComposedBuffer(
        [_recency(), NStepSelector(n_step=2, gamma=0.9), _long_term()]
    )
    nested = ComposedBuffer(
        [
            _recency(),
            ComposedBuffer([NStepSelector(n_step=2, gamma=0.9), _long_term()]),
        ]
    )

    flat_state = _add_steps(flat, 15)
    nested_state = _add_steps(nested, 15)

    for flat_leaf, nested_leaf in zip(
        jax.tree.leaves(flat_state), jax.tree.leaves(nested_state), strict=True
    ):
        assert np.array_equal(flat_leaf, nested_leaf)


# Mixture sampling


def _endpoint_like() -> ComposedBuffer:
    return ComposedBuffer(
        [
            ReplayBuffer(capacity=6, batch_size=4, n_step=1, gamma=0.9),
            NStepSelector(n_step=2, gamma=0.9),
            _long_term(capacity=5),
        ]
    )


# After 6 steps nothing has left recency; after 12 two chunks are stored.
_LONG_TERM_EMPTY, _LONG_TERM_FILLED = 6, 12


@pytest.mark.parametrize(
    ("num_steps", "expected_discounts"),
    [(_LONG_TERM_EMPTY, [0.9] * 4), (_LONG_TERM_FILLED, [0.9] * 3 + [0.81])],
    ids=["long-term-empty", "long-term-filled"],
)
def test_merged_sample_fills_unavailable_rows_from_the_fallback(
    num_steps: int, expected_discounts: list[float]
):
    """Long-term rows come from recency until the long-term buffer has data.

    Recency rows are one-step (discount 0.9); long-term rows are two-step
    chunks (discount 0.81).
    """
    buffer = _endpoint_like()
    state = _add_steps(buffer, num_steps)

    batch = jax.jit(buffer.sample)(state, jax.random.key(0))

    assert batch.obs.shape == (4, 1)
    assert np.allclose(batch.discount, expected_discounts)
    assert np.asarray(batch.mask).all()


@pytest.mark.parametrize(
    ("num_steps", "fallback_mask", "long_term_mask"),
    [
        (_LONG_TERM_EMPTY, [True] * 4, [False]),
        (_LONG_TERM_FILLED, [True] * 3 + [False], [True]),
    ],
    ids=["long-term-empty", "long-term-filled"],
)
def test_component_masks_keep_one_batch_of_rows_live(
    num_steps: int, fallback_mask: list[bool], long_term_mask: list[bool]
):
    """Per-component batches keep static shapes and exactly four live rows."""
    buffer = _endpoint_like()
    state = _add_steps(buffer, num_steps)

    recent, long_term = jax.jit(buffer.sample_components)(state, jax.random.key(0))

    assert recent.obs.shape == (4, 1)
    assert long_term.obs.shape == (1, 1)
    assert np.asarray(recent.mask).tolist() == fallback_mask
    assert np.asarray(long_term.mask).tolist() == long_term_mask


@pytest.mark.parametrize("num_steps", [_LONG_TERM_EMPTY, _LONG_TERM_FILLED])
def test_merged_rows_are_the_live_component_rows(num_steps: int):
    """With the same key, the merged batch holds exactly the live rows."""
    buffer = _endpoint_like()
    state = _add_steps(buffer, num_steps)
    key = jax.random.key(3)

    merged = buffer.sample(state, key)
    components = buffer.sample_components(state, key)

    live = [
        np.asarray(component.action)[np.asarray(component.mask)]
        for component in components
    ]
    assert np.array_equal(np.asarray(merged.action), np.concatenate(live))
