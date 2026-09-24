"""Run with: uv run --frozen --group bench pytest microbenchmarks."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from components import BufferState, ReplayBuffer


@dataclass(frozen=True)
class _Space:
    shape: tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True)
class _FrameSpace(_Space):
    frame_channels: int


_CONFIGS = [
    pytest.param((4,), np.float32, None, 10_000, id="vector"),
    pytest.param((84, 84, 4), np.uint8, 1, 10_000, id="atari-frames"),
    pytest.param((84, 84, 4), np.uint8, None, 10_000, id="atari-whole"),
    pytest.param((210, 160, 12), np.uint8, 3, 2_000, id="rgb-frames"),
]
_BATCH_SIZE = 32
_N_STEP = 3
_ADD_STEPS = 128


def _state_bytes(state: BufferState) -> int:
    return sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(state))


def _make_benchmark(
    shape: tuple[int, ...],
    dtype: type[np.generic],
    frame_channels: int | None,
    capacity: int,
):
    space = (
        _Space(shape, np.dtype(dtype))
        if frame_channels is None
        else _FrameSpace(shape, np.dtype(dtype), frame_channels)
    )
    buffer = ReplayBuffer(
        capacity=capacity, batch_size=_BATCH_SIZE, n_step=_N_STEP, gamma=0.99
    )
    state = buffer.init(space)
    observation = jnp.zeros(shape, dtype)
    return buffer, state, observation


def _compile_add_scan(
    buffer: ReplayBuffer, state: BufferState, observation: jax.Array, steps: int
):
    def add_scan(initial_state: BufferState, obs: jax.Array):
        def add_one(carry: BufferState, _: None):
            state = buffer.add(
                carry,
                obs,
                jnp.int32(0),
                jnp.float32(1),
                jnp.bool_(False),
                jnp.bool_(False),
            )
            return state, None

        return jax.lax.scan(add_one, initial_state, None, length=steps)[0]

    return jax.jit(add_scan, donate_argnums=0).lower(state, observation).compile()


@pytest.mark.parametrize(
    ("shape", "dtype", "frame_channels", "capacity"), _CONFIGS
)
def test_jitted_add_scan(
    benchmark: BenchmarkFixture,
    shape: tuple[int, ...],
    dtype: type[np.generic],
    frame_channels: int | None,
    capacity: int,
):
    """Benchmark a donated add scan after compilation and warm-up."""
    buffer, state, observation = _make_benchmark(
        shape, dtype, frame_channels, capacity
    )
    add_scan = _compile_add_scan(buffer, state, observation, _ADD_STEPS)
    state = jax.block_until_ready(add_scan(state, observation))

    state_bytes = _state_bytes(state)
    benchmark.extra_info["buffer_state_bytes"] = state_bytes
    benchmark.extra_info["add_calls_per_scan"] = _ADD_STEPS
    analysis = add_scan.memory_analysis()
    if analysis is None:
        benchmark.extra_info["add_scan_temp_bytes"] = "unavailable"
    else:
        temp_bytes = int(analysis.temp_size_in_bytes)
        benchmark.extra_info["add_scan_temp_bytes"] = temp_bytes
        assert temp_bytes < state_bytes // 4

    state_holder = [state]

    def run_add_scan():
        state_holder[0] = add_scan(state_holder[0], observation)
        jax.block_until_ready(state_holder[0])

    benchmark(run_add_scan)


@pytest.mark.parametrize(
    ("shape", "dtype", "frame_channels", "capacity"), _CONFIGS
)
def test_jitted_sample(
    benchmark: BenchmarkFixture,
    shape: tuple[int, ...],
    dtype: type[np.generic],
    frame_channels: int | None,
    capacity: int,
):
    """Benchmark warmed sampling across observation layouts."""
    buffer, state, observation = _make_benchmark(
        shape, dtype, frame_channels, capacity
    )
    fill_scan = _compile_add_scan(buffer, state, observation, _ADD_STEPS)
    state = jax.block_until_ready(fill_scan(state, observation))
    key = jax.random.key(0)
    sample = jax.jit(buffer.sample).lower(state, key).compile()
    jax.block_until_ready(sample(state, key))
    benchmark.extra_info["buffer_state_bytes"] = _state_bytes(state)

    benchmark(lambda: jax.block_until_ready(sample(state, key)))
