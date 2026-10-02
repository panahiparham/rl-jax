from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from components.buffers.composed import ComposedBuffer
from components.buffers.selector import NStepSelector
from components.buffers.subsample import Subsample
from components.buffers.transition import TransitionBuffer, TransitionState


@dataclass(frozen=True)
class _Space:
    shape: tuple[int, ...] = (1,)
    dtype: type = jnp.float32


def _passed(every: int, valid: list[bool]) -> list[int]:
    stage = Subsample(every=every)
    state = stage.init(_Space())
    passed = []
    for index, is_valid in enumerate(valid):
        pushed = stage.push(state, jnp.asarray(index), jnp.asarray(is_valid))
        state = pushed.state
        if bool(pushed.out_valid):
            passed.append(int(pushed.out))
    return passed


@pytest.mark.parametrize(
    ("every", "expected"), [(1, list(range(10))), (3, [0, 3, 6, 9]), (10, [0])]
)
def test_subsample_passes_every_kth_item_from_the_first(
    every: int, expected: list[int]
):
    """Items pass unchanged, every ``every``-th one counting from the first."""
    assert _passed(every, [True] * 10) == expected


def test_subsample_counts_only_valid_items():
    """Invalid items neither pass nor advance the count."""
    valid = [True, False, True, False, False, True, True, False, True]

    assert _passed(2, valid) == [0, 5, 8]


def test_subsample_passes_nothing_from_an_invalid_stream():
    """A stage that never sees a valid item never passes one on."""
    assert _passed(1, [False] * 5) == []


def test_subsample_thins_one_step_transitions_in_a_chain():
    """Behind a 1-step selector, every second transition reaches storage."""
    buffer = ComposedBuffer(
        [
            NStepSelector(n_step=1, gamma=0.9),
            Subsample(every=2),
            TransitionBuffer(capacity=10, batch_size=1),
        ]
    )
    state = buffer.init(_Space())
    for step in range(9):
        state = buffer.add(
            state,
            jnp.asarray([step], jnp.float32),
            jnp.asarray(step, jnp.int32),
            jnp.asarray(1.0, jnp.float32),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(1.0, jnp.float32),
        )

    stored = state.states[2]
    assert isinstance(stored, TransitionState)
    size = int(stored.size)
    # Step 8 has no successor yet, so transitions start at steps 0 to 7.
    assert np.asarray(stored.data.action[:size]).tolist() == [0, 2, 4, 6]
    assert np.asarray(stored.data.boot_action[:size]).tolist() == [1, 3, 5, 7]
