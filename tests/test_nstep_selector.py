from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

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
