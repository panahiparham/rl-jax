from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from components import TimeStep
from components.buffers.selector import NStepSelector


@dataclass(frozen=True)
class _Space:
    shape: tuple[int, ...] = (1,)
    dtype: type = jnp.float32


class _Step(NamedTuple):
    reward: float
    termination: bool = False
    truncation: bool = False
    discount: float = 1.0


class _Chunk(NamedTuple):
    emitted_at: int
    start: int
    ret: float
    discount: float
    boot: int


def _ends_segment(step: _Step) -> bool:
    return step.termination or step.truncation or step.discount == 0.0


def _reference_chunks(steps: list[_Step], n_step: int, gamma: float) -> list[_Chunk]:
    chunks = []
    start = 0
    while start < len(steps):
        end = start
        while end < len(steps) and not _ends_segment(steps[end]):
            end += 1
        truncated = end < len(steps) and steps[end].truncation
        last = end - 1 if truncated or end == len(steps) else end
        for chunk_start in range(start, last + 1, n_step):
            boot = min(chunk_start + n_step, last + 1)
            if boot >= len(steps):
                break
            ret, weight = 0.0, 1.0
            for step in steps[chunk_start:boot]:
                ret += weight * step.reward
                weight *= gamma * step.discount
            chunks.append(_Chunk(boot, chunk_start, ret, weight, boot))
        start = end + 1
    return chunks


def _run_selector(
    steps: list[_Step], n_step: int, gamma: float, valid: list[bool] | None = None
) -> list[_Chunk]:
    selector = NStepSelector(n_step=n_step, gamma=gamma)
    push = jax.jit(selector.push)
    state = selector.init(_Space())
    chunks = []
    for index, step in enumerate(steps):
        pushed = push(
            state,
            TimeStep(
                obs=jnp.asarray([index], jnp.float32),
                action=jnp.asarray(index, jnp.int32),
                reward=jnp.asarray(step.reward, jnp.float32),
                termination=jnp.asarray(step.termination),
                truncation=jnp.asarray(step.truncation),
                discount=jnp.asarray(step.discount, jnp.float32),
            ),
            jnp.asarray(True if valid is None else valid[index]),
        )
        state = pushed.state
        if not bool(pushed.out_valid):
            continue
        out = pushed.out
        assert int(out.obs[0]) == int(out.action)
        assert int(out.boot_obs[0]) == int(out.boot_action)
        assert bool(out.mask)
        chunks.append(
            _Chunk(
                index,
                int(out.action),
                float(out.ret),
                float(out.discount),
                int(out.boot_action),
            )
        )
    return chunks


def _assert_chunks_match(actual: list[_Chunk], expected: list[_Chunk]):
    assert [(c.emitted_at, c.start, c.boot) for c in actual] == [
        (c.emitted_at, c.start, c.boot) for c in expected
    ]
    assert np.allclose([c.ret for c in actual], [c.ret for c in expected])
    assert np.allclose([c.discount for c in actual], [c.discount for c in expected])


# Chaining within an episode


def test_chunks_within_an_episode_chain_start_to_bootstrap():
    """Each chunk bootstraps from the next chunk's start, n_step steps later."""
    steps = [_Step(reward=float(i)) for i in range(10)]

    chunks = _run_selector(steps, n_step=3, gamma=0.5)

    assert [(c.start, c.boot) for c in chunks] == [(0, 3), (3, 6), (6, 9)]
    assert np.allclose([c.ret for c in chunks], [1.0, 6.25, 11.5])
    assert np.allclose([c.discount for c in chunks], [0.125] * 3)
    _assert_chunks_match(chunks, _reference_chunks(steps, n_step=3, gamma=0.5))


def test_one_step_chunks_emit_every_transition():
    """With n_step of one, every step becomes its own transition."""
    steps = [_Step(reward=1.0) for _ in range(4)]

    chunks = _run_selector(steps, n_step=1, gamma=0.9)

    assert [(c.start, c.boot) for c in chunks] == [(0, 1), (1, 2), (2, 3)]
    assert np.allclose([c.discount for c in chunks], [0.9] * 3)


# Episode and life boundaries


@pytest.mark.parametrize("terminates", [True, False])
def test_zero_discount_closes_the_chunk_and_restarts_the_phase(terminates: bool):
    """A termination or a life loss ends the chunk with no bootstrap.

    Both show up as a zero step discount; the next chunk starts on the
    following step instead of continuing the old phase.
    """
    steps = [_Step(1.0) for _ in range(10)]
    steps[4] = _Step(1.0, termination=terminates, discount=0.0)

    chunks = _run_selector(steps, n_step=3, gamma=1.0)

    _assert_chunks_match(
        chunks,
        [
            _Chunk(3, 0, 3.0, 1.0, 3),
            _Chunk(5, 3, 2.0, 0.0, 5),
            _Chunk(8, 5, 3.0, 1.0, 8),
        ],
    )


def test_truncation_ends_the_chunk_on_the_truncating_step():
    """A truncated chunk bootstraps from the truncating step's own observation.

    The truncating step's reward is dropped because its successor belongs to
    the next episode.
    """
    steps = [_Step(1.0) for _ in range(10)]
    steps[4] = _Step(1.0, truncation=True)

    chunks = _run_selector(steps, n_step=3, gamma=1.0)

    _assert_chunks_match(
        chunks,
        [
            _Chunk(3, 0, 3.0, 1.0, 3),
            _Chunk(4, 3, 1.0, 1.0, 4),
            _Chunk(8, 5, 3.0, 1.0, 8),
        ],
    )


def test_truncation_at_a_chunk_start_emits_nothing_for_that_step():
    """A chunk that would start on the truncating step is never emitted."""
    steps = [_Step(1.0) for _ in range(8)]
    steps[3] = _Step(1.0, truncation=True)

    chunks = _run_selector(steps, n_step=3, gamma=1.0)

    _assert_chunks_match(
        chunks, [_Chunk(3, 0, 3.0, 1.0, 3), _Chunk(7, 4, 3.0, 1.0, 7)]
    )


def test_invalid_steps_are_ignored():
    """Steps marked invalid neither advance nor emit a chunk."""
    steps = [_Step(1.0) for _ in range(9)]
    valid = [True] * 4 + [False] * 2 + [True] * 3

    chunks = _run_selector(steps, n_step=3, gamma=1.0, valid=valid)

    _assert_chunks_match(
        chunks, [_Chunk(3, 0, 3.0, 1.0, 3), _Chunk(8, 3, 3.0, 1.0, 8)]
    )


# Agreement with the reference


_STEP_KINDS = {
    "plain": {},
    "termination": {"termination": True, "discount": 0.0},
    "life_loss": {"discount": 0.0},
    "truncation": {"truncation": True},
    "truncated_life_loss": {"truncation": True, "discount": 0.0},
}


@st.composite
def _step_streams(draw: st.DrawFn):
    n_step = draw(st.integers(1, 4))
    gamma = draw(st.sampled_from([1.0, 0.9]))
    kinds = st.sampled_from(sorted(_STEP_KINDS))
    stream = draw(
        st.lists(st.tuples(st.integers(-3, 3), kinds), min_size=1, max_size=25)
    )
    steps = [_Step(float(reward), **_STEP_KINDS[kind]) for reward, kind in stream]
    return n_step, gamma, steps


@settings(max_examples=20, deadline=None, derandomize=True)
@given(case=_step_streams())
def test_selector_matches_the_reference_on_random_streams(
    case: tuple[int, float, list[_Step]],
):
    """Emitted chunks match the segment-based reference for any boundary mix."""
    n_step, gamma, steps = case

    chunks = _run_selector(steps, n_step=n_step, gamma=gamma)

    _assert_chunks_match(chunks, _reference_chunks(steps, n_step, gamma))
