"""Run with: uv run --frozen --group bench pytest microbenchmarks."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from components import BufferState, ReplayBuffer, TimeStep
from components.buffers.composed import ComposedBuffer
from components.buffers.reservoir import ReservoirBuffer
from components.buffers.selector import NStepSelector
from components.buffers.transition import TransitionBuffer


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


def _state_bytes(state: object) -> int:
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


def _compile_scan[S](
    step: Callable[[S, jax.Array], S], state: S, observation: jax.Array, steps: int
):
    def scan(initial_state: S, obs: jax.Array):
        def one(carry: S, _: None):
            return step(carry, obs), None

        return jax.lax.scan(one, initial_state, None, length=steps)[0]

    return jax.jit(scan, donate_argnums=0).lower(state, observation).compile()


def _compile_add_scan(
    buffer: ReplayBuffer | ComposedBuffer,
    state: object,
    observation: jax.Array,
    steps: int,
):
    def add(carry, obs: jax.Array):
        return buffer.add(
            carry,
            obs,
            jnp.int32(0),
            jnp.float32(1),
            jnp.bool_(False),
            jnp.bool_(False),
            jnp.float32(1),
        )

    return _compile_scan(add, state, observation, steps)


def _benchmark_scan(
    benchmark: BenchmarkFixture,
    scan: jax.stages.Compiled,
    state: object,
    observation: jax.Array,
):
    state = jax.block_until_ready(scan(state, observation))

    state_bytes = _state_bytes(state)
    benchmark.extra_info["buffer_state_bytes"] = state_bytes
    benchmark.extra_info["calls_per_scan"] = _ADD_STEPS
    analysis = scan.memory_analysis()
    if analysis is None:
        benchmark.extra_info["scan_temp_bytes"] = "unavailable"
    else:
        temp_bytes = int(analysis.temp_size_in_bytes)
        benchmark.extra_info["scan_temp_bytes"] = temp_bytes
        assert temp_bytes < state_bytes // 4

    state_holder = [state]

    def run_scan():
        state_holder[0] = scan(state_holder[0], observation)
        jax.block_until_ready(state_holder[0])

    benchmark(run_scan)


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
    _benchmark_scan(benchmark, add_scan, state, observation)


@pytest.mark.parametrize(
    ("shape", "dtype", "frame_channels", "capacity"), _CONFIGS
)
def test_jitted_push_scan(
    benchmark: BenchmarkFixture,
    shape: tuple[int, ...],
    dtype: type[np.generic],
    frame_channels: int | None,
    capacity: int,
):
    """Benchmark a donated push scan, which also emits the outgoing timestep."""
    buffer, state, observation = _make_benchmark(
        shape, dtype, frame_channels, capacity
    )

    def push(carry: BufferState, obs: jax.Array):
        step = TimeStep(
            obs,
            jnp.int32(0),
            jnp.float32(1),
            jnp.bool_(False),
            jnp.bool_(False),
            jnp.float32(1),
        )
        return buffer.push(carry, step, jnp.bool_(True)).state

    push_scan = _compile_scan(push, state, observation, _ADD_STEPS)
    _benchmark_scan(benchmark, push_scan, state, observation)


def test_jitted_endpoint_add_scan(benchmark: BenchmarkFixture):
    """Benchmark adds through recency, an n-step selector, and long-term storage."""
    buffer = ComposedBuffer(
        [
            ReplayBuffer(
                capacity=10_000, batch_size=_BATCH_SIZE, n_step=1, gamma=0.99
            ),
            NStepSelector(n_step=10, gamma=0.99),
            TransitionBuffer(capacity=2_000, batch_size=4),
        ]
    )
    state = buffer.init(_FrameSpace((84, 84, 4), np.dtype(np.uint8), 1))
    observation = jnp.zeros((84, 84, 4), jnp.uint8)
    add_scan = _compile_add_scan(buffer, state, observation, _ADD_STEPS)
    _benchmark_scan(benchmark, add_scan, state, observation)


def test_jitted_reservoir_add_scan(benchmark: BenchmarkFixture):
    """Benchmark adds through recency, a 1-step selector, and a reservoir."""
    buffer = ComposedBuffer(
        [
            ReplayBuffer(
                capacity=10_000, batch_size=_BATCH_SIZE, n_step=1, gamma=0.99
            ),
            NStepSelector(n_step=1, gamma=0.99),
            ReservoirBuffer(capacity=2_000, batch_size=4),
        ]
    )
    state = buffer.init(_FrameSpace((84, 84, 4), np.dtype(np.uint8), 1))
    observation = jnp.zeros((84, 84, 4), jnp.uint8)
    add_scan = _compile_add_scan(buffer, state, observation, _ADD_STEPS)
    _benchmark_scan(benchmark, add_scan, state, observation)


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
