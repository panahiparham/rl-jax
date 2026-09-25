from __future__ import annotations

from functools import cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from components import ReplayBuffer, TimeStep, n_step_return
from components.buffer import sample_windows, stack_frames
from environments import ENVIRONMENTS


class _ReferenceSample(NamedTuple):
    ret: float
    discount: float
    horizon: int
    boot_id: int
    mask: bool


class _Space:
    shape = (1,)
    dtype = jnp.float32


class _FrameStackSpace:
    dtype = jnp.int32

    def __init__(self, shape: tuple[int, ...], frame_channels: int):
        self.shape = shape
        self.frame_channels = frame_channels


@st.composite
def _reference_cases(draw: st.DrawFn):
    capacity = draw(st.sampled_from([4, 5, 6]))
    n_step = draw(st.integers(1, min(3, capacity - 1)))
    num_adds = draw(st.integers(max(2, n_step + 1), capacity + 3))
    flags = draw(
        st.lists(
            st.tuples(st.booleans(), st.booleans()),
            min_size=num_adds,
            max_size=num_adds,
        )
    )
    rewards = draw(
        st.lists(
            st.integers(-3, 3),
            min_size=num_adds,
            max_size=num_adds,
        )
    )
    return capacity, n_step, flags, rewards


@st.composite
def _window_cases(draw: st.DrawFn):
    lookback = draw(st.integers(0, 3))
    lookahead = draw(st.integers(1, 3))
    capacity = draw(st.integers(lookback + lookahead + 1, 8))
    head = draw(st.integers(0, capacity - 1))
    if capacity == lookahead + 1 or draw(st.booleans()):
        size = capacity
    else:
        size = draw(st.integers(lookahead + 1, capacity - 1))
    return capacity, lookback, lookahead, head, size


@cache
def _compiled_sampler(capacity: int, n_step: int, batch_size: int):
    buffer = ReplayBuffer(
        capacity=capacity, batch_size=batch_size, n_step=n_step, gamma=0.9
    )
    return buffer, jax.jit(jax.vmap(buffer.sample, in_axes=(None, 0)))


@cache
def _compiled_frame_sampler(
    capacity: int,
    n_step: int,
    height: int,
    width: int,
    stack_size: int,
    frame_channels: int,
):
    batch_size = 2
    buffer = ReplayBuffer(
        capacity=capacity, batch_size=batch_size, n_step=n_step, gamma=0.9
    )
    space = _FrameStackSpace(
        (height, width, stack_size * frame_channels), frame_channels
    )
    return buffer, space, jax.jit(jax.vmap(buffer.sample, in_axes=(None, 0)))


_jit_sample_windows = jax.jit(
    sample_windows,
    static_argnames=("capacity", "lookback", "lookahead", "batch_size"),
)


def _reference_sample(
    transitions: list[tuple[int, int, int, bool, bool]], start: int, n_step: int
) -> _ReferenceSample:
    window = transitions[start : start + n_step]
    horizon = next((i + 1 for i, t in enumerate(window) if t[3] or t[4]), n_step)
    last = window[horizon - 1]
    return _ReferenceSample(
        sum(0.9**i * t[2] for i, t in enumerate(window[:horizon])),
        0.0 if last[3] else 0.9**horizon,
        horizon,
        transitions[start + horizon][0],
        not last[4],
    )


def _synthetic_frame_log(
    episode_lengths: list[int],
    truncations: list[bool],
    height: int,
    width: int,
    stack_size: int,
    frame_channels: int,
):
    transitions = []
    observations = []
    pixel = (
        np.arange(height)[:, None, None] * 30
        + np.arange(width)[None, :, None] * 3
        + np.arange(frame_channels)[None, None, :]
    )
    step = 0
    for episode_id, (length, is_truncation) in enumerate(
        zip(episode_lengths, truncations, strict=True)
    ):
        frames = []
        for episode_step in range(length):
            frame = (step * 10_000 + episode_id * 1_000 + pixel).astype(np.int32)
            frames.append(frame)
            padded = [np.zeros_like(frame)] * (stack_size - len(frames))
            observations.append(
                np.concatenate((*padded, *frames[-stack_size:]), axis=-1)
            )
            terminated = episode_step == length - 1 and not is_truncation
            truncated = episode_step == length - 1 and is_truncation
            reward = (step * 3 + episode_id) % 7 - 3
            transitions.append((step, step, reward, terminated, truncated))
            step += 1
    return transitions, observations


@st.composite
def _frame_stack_cases(draw: st.DrawFn):
    height = draw(st.integers(1, 3))
    width = draw(st.integers(1, 3))
    stack_size = draw(st.sampled_from([1, 2, 4]))
    frame_channels = draw(st.sampled_from([1, 3]))
    n_step = draw(st.integers(1, 3))
    capacity = draw(st.integers(max(stack_size + n_step, 4), 8))
    episode_lengths = draw(st.lists(st.integers(1, 5), min_size=2, max_size=4))
    assume(sum(episode_lengths) >= max(2, n_step + 1))
    truncations = draw(
        st.lists(
            st.booleans(),
            min_size=len(episode_lengths),
            max_size=len(episode_lengths),
        )
    )
    return (
        height,
        width,
        stack_size,
        frame_channels,
        n_step,
        capacity,
        episode_lengths,
        truncations,
    )


def _window(reward, termination, truncation, obs):
    return TimeStep(
        obs=jnp.array([obs]),
        action=jnp.zeros((1, len(reward)), jnp.int32),
        reward=jnp.array([reward]),
        termination=jnp.array([termination]),
        truncation=jnp.array([truncation]),
        discount=1.0 - jnp.array([termination], jnp.float32),
    )


def _fill(buffer, rewards, terminations, truncations):
    state = buffer.init(_Space)
    for i, (r, term, trunc) in enumerate(
        zip(rewards, terminations, truncations, strict=True)
    ):
        state = buffer.add(
            state,
            jnp.array([float(i)]),
            jnp.asarray(i % 2, jnp.int32),
            jnp.asarray(r, jnp.float32),
            jnp.asarray(term),
            jnp.asarray(trunc),
            jnp.float32(not term),
        )
    return state


def test_n_step_return_runs_to_the_end_of_an_unbroken_window():
    batch = _window(
        [1.0, 2.0, 3.0, 0.0],
        [False] * 4,
        [False] * 4,
        [[10.0], [20.0], [30.0], [40.0]],
    )
    ret, discount, horizon, mask = n_step_return(batch, 0.9, 3)
    np.testing.assert_allclose(np.asarray(ret), [5.23], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(discount), [0.729], rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(horizon), [3])
    np.testing.assert_array_equal(np.asarray(mask), [True])


@settings(max_examples=12, deadline=None, derandomize=True)
@given(case=_reference_cases())
@example(
    case=(
        4,
        2,
        [(False, False)] * 3 + [(False, True), (True, False)] + [(False, False)] * 2,
        list(range(1, 8)),
    )
)
def test_sampled_batches_match_a_reference_buffer(
    case: tuple[int, int, list[tuple[bool, bool]], list[int]],
):
    """Sampled rows match each legal reference window after ring wrap.

    The sample keys must reach every retained start.
    """
    capacity, n_step, boundaries, rewards = case
    batch_size = 2
    buffer, sample = _compiled_sampler(capacity, n_step, batch_size)
    state = buffer.init(_Space)
    transitions = [
        (i, i + 10, reward, *flags)
        for i, (flags, reward) in enumerate(zip(boundaries, rewards, strict=True))
    ]
    for i, action, reward, terminated, truncated in transitions:
        state = buffer.add(
            state,
            jnp.asarray([i], jnp.float32),
            jnp.asarray(action, jnp.int32),
            jnp.asarray(reward, jnp.float32),
            jnp.asarray(terminated),
            jnp.asarray(truncated),
            jnp.float32(not terminated),
        )

    assert bool(buffer.can_sample(state))
    retained = transitions[-capacity:]
    legal_starts = {t[0] for t in retained[: len(retained) - n_step]}
    batch = jax.tree.map(
        lambda value: np.asarray(value).reshape(-1),
        sample(state, jax.random.split(jax.random.key(0), 32)),
    )
    sampled_starts = batch.obs.astype(int)
    assert set(sampled_starts) == set(legal_starts)

    for row, start_id in enumerate(sampled_starts):
        retained_start = int(start_id - retained[0][0])
        expected = _reference_sample(retained, retained_start, n_step)
        assert batch.action[row] == retained[retained_start][1]
        assert np.isclose(batch.ret[row], expected.ret, rtol=1e-6)
        assert np.isclose(batch.discount[row], expected.discount, rtol=1e-6)
        assert int(batch.boot_obs[row]) == expected.boot_id
        assert bool(batch.mask[row]) is expected.mask


@settings(max_examples=8, deadline=None, derandomize=True)
@given(case=_frame_stack_cases())
@example(case=(2, 3, 4, 1, 1, 8, [1, 2, 2], [False, True, False]))
@example(case=(2, 3, 4, 3, 1, 7, [1, 2, 4, 1], [False, True, False, True]))
def test_frame_stacked_batches_match_a_synthetic_environment(
    case: tuple[int, int, int, int, int, int, list[int], list[bool]],
):
    """Rebuild sampled and retained observations from synthetic episode frames."""
    (
        height,
        width,
        stack_size,
        frame_channels,
        n_step,
        capacity,
        episode_lengths,
        truncations,
    ) = case
    transitions, observations = _synthetic_frame_log(
        episode_lengths,
        truncations,
        height,
        width,
        stack_size,
        frame_channels,
    )
    buffer, space, sample = _compiled_frame_sampler(
        capacity, n_step, height, width, stack_size, frame_channels
    )
    state = buffer.init(space)
    for step, action, reward, terminated, truncated in transitions:
        state = buffer.add(
            state,
            jnp.asarray(observations[step]),
            jnp.asarray(action, jnp.int32),
            jnp.asarray(reward, jnp.float32),
            jnp.asarray(terminated),
            jnp.asarray(truncated),
            jnp.float32(not terminated),
        )

    assert bool(buffer.can_sample(state))
    retained = transitions[-capacity:]
    batches = jax.tree.map(
        np.asarray, sample(state, jax.random.split(jax.random.key(0), 16))
    )
    batches = jax.tree.map(
        lambda x: x.reshape((-1, *x.shape[2:])), batches
    )
    for row, start_id in enumerate(batches.action.astype(int)):
        retained_start = next(
            i for i, transition in enumerate(retained) if transition[0] == start_id
        )
        assert retained_start < len(retained) - n_step
        expected = _reference_sample(retained, retained_start, n_step)
        assert np.isclose(batches.ret[row], expected.ret, rtol=1e-6)
        assert np.isclose(batches.discount[row], expected.discount, rtol=1e-6)
        assert bool(batches.mask[row]) is expected.mask
        np.testing.assert_array_equal(batches.obs[row], observations[start_id])
        np.testing.assert_array_equal(
            batches.boot_obs[row], observations[start_id + expected.horizon]
        )

    stored = buffer.stored_transitions(state)
    skip = stack_size - 1 if len(transitions) >= capacity else 0
    expected_ids = [transition[0] for transition in retained[skip:]]
    np.testing.assert_array_equal(
        np.asarray(stored.obs), np.stack([observations[i] for i in expected_ids])
    )


@settings(max_examples=8, deadline=None, derandomize=True)
@given(case=_window_cases())
@example(case=(6, 1, 2, 0, 6))
@example(case=(7, 2, 2, 3, 7))
@example(case=(6, 2, 2, 4, 4))
@example(case=(4, 1, 2, 2, 4))
def test_sample_windows_cover_legal_starts(case: tuple[int, int, int, int, int]):
    """Check sampled windows are consecutive and cover legal starts."""
    capacity, lookback, lookahead, head, size = case
    batch_size = 128
    indices = np.asarray(
        _jit_sample_windows(
            jax.random.key(0),
            jnp.asarray(head, jnp.int32),
            jnp.asarray(size, jnp.int32),
            capacity,
            lookback,
            lookahead,
            batch_size,
        )
    )

    assert indices.shape == (batch_size, lookback + lookahead + 1)
    assert indices.dtype == np.int32
    oldest = (head - size) % capacity
    positions = (indices - oldest) % capacity
    start_positions = positions[:, lookback]
    before_start = positions[:, :lookback]
    before_start = np.where(
        before_start > start_positions[:, None],
        before_start - capacity,
        before_start,
    )
    ordered_positions = np.concatenate(
        (before_start, positions[:, lookback:]), axis=1
    )

    assert np.all(np.diff(ordered_positions, axis=1) == 1)
    assert np.all(positions[:, lookback:] < size)
    if size == capacity:
        assert np.all(ordered_positions >= 0)

    expected_starts = size - (lookback if size == capacity else 0) - lookahead
    assert len(set(start_positions.tolist())) == expected_starts


def test_stack_frames_preserves_rgb_layout_and_start_at_zero():
    """Preserve RGB layout and leave a start at position zero unchanged."""
    frames = np.arange(4 * 2 * 3 * 3, dtype=np.uint8).reshape(4, 2, 3, 3)
    expected = np.concatenate([frames[i] for i in range(4)], axis=-1)

    for first in (np.zeros(4, dtype=bool), np.array([True, False, False, False])):
        stacked = stack_frames(jnp.asarray(frames), jnp.asarray(first))
        np.testing.assert_array_equal(np.asarray(stacked), expected)


@pytest.mark.parametrize(
    ("first", "last_start"),
    [([False, False, True, False], 2), ([False, True, False, True], 3)],
)
def test_stack_frames_zeroes_frames_before_the_last_start(
    first: list[bool], last_start: int
):
    """Zero frames preceding the latest episode start."""
    frames = np.arange(4 * 2 * 3 * 3, dtype=np.uint8).reshape(4, 2, 3, 3)
    expected_frames = [
        np.zeros_like(frame) if i < last_start else frame
        for i, frame in enumerate(frames)
    ]
    expected = np.concatenate(expected_frames, axis=-1)

    stacked = stack_frames(jnp.asarray(frames), jnp.asarray(first))

    np.testing.assert_array_equal(np.asarray(stacked), expected)


@pytest.mark.parametrize("frame_shape", [(2, 3, 3), (5,)])
def test_stack_frames_with_one_frame_returns_that_frame(
    frame_shape: tuple[int, ...],
):
    """Remove a singleton frame axis for images and vectors."""
    frame = np.arange(np.prod(frame_shape), dtype=np.uint8).reshape(frame_shape)
    result = stack_frames(jnp.asarray(frame[None]), jnp.asarray([True]))

    np.testing.assert_array_equal(np.asarray(result), frame)


def test_stack_frames_handles_leading_batch_dims_independently():
    """Pad each batch row using its own latest episode start."""
    frames = np.arange(2 * 4 * 2 * 2 * 3, dtype=np.uint8).reshape(2, 4, 2, 2, 3)
    first = np.array(
        [[False, False, True, False], [False, True, False, True]], dtype=bool
    )
    zero = np.zeros_like(frames[0, 0])
    expected = np.stack(
        [
            np.concatenate((zero, zero, frames[0, 2], frames[0, 3]), axis=-1),
            np.concatenate((zero, zero, zero, frames[1, 3]), axis=-1),
        ]
    )

    stacked = stack_frames(jnp.asarray(frames), jnp.asarray(first))

    np.testing.assert_array_equal(np.asarray(stacked), expected)


def test_termination_cuts_the_window_and_zeroes_the_discount():
    batch = _window(
        [1.0, 2.0, 3.0, 0.0],
        [False, True, False, False],
        [False] * 4,
        [[10.0], [20.0], [30.0], [40.0]],
    )
    ret, discount, horizon, mask = n_step_return(batch, 0.9, 3)
    np.testing.assert_allclose(np.asarray(ret), [2.8], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(discount), [0.0])
    np.testing.assert_array_equal(np.asarray(horizon), [2])
    # a terminal window still trains: the discount alone removes the bootstrap
    np.testing.assert_array_equal(np.asarray(mask), [True])


def test_zero_step_discount_ends_the_return_without_cutting_the_window():
    """A non-terminal zero discount, such as a lost life, drops every later
    reward and the bootstrap but leaves the window and its mask intact."""
    batch = _window(
        [1.0, 2.0, 3.0, 0.0],
        [False] * 4,
        [False] * 4,
        [[10.0], [20.0], [30.0], [40.0]],
    )._replace(discount=jnp.array([[1.0, 0.0, 1.0, 1.0]]))
    ret, discount, horizon, mask = n_step_return(batch, 0.9, 3)
    np.testing.assert_allclose(np.asarray(ret), [2.8], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(discount), [0.0])
    np.testing.assert_array_equal(np.asarray(horizon), [3])
    np.testing.assert_array_equal(np.asarray(mask), [True])


def test_step_discounts_compound_with_gamma():
    """Each reward and the bootstrap are scaled by the step discounts before
    them, on top of ``gamma``."""
    batch = _window(
        [1.0, 2.0, 3.0, 0.0],
        [False] * 4,
        [False] * 4,
        [[10.0], [20.0], [30.0], [40.0]],
    )._replace(discount=jnp.array([[0.5, 1.0, 1.0, 1.0]]))
    ret, discount, _horizon, _mask = n_step_return(batch, 0.9, 3)
    np.testing.assert_allclose(np.asarray(ret), [3.115], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(discount), [0.3645], rtol=1e-6)


def test_truncation_cuts_the_window_and_clears_the_mask():
    batch = _window(
        [1.0, 2.0, 3.0, 0.0],
        [False] * 4,
        [False, True, False, False],
        [[10.0], [20.0], [30.0], [40.0]],
    )
    ret, discount, horizon, mask = n_step_return(batch, 0.9, 3)
    np.testing.assert_allclose(np.asarray(ret), [2.8], rtol=1e-6)
    # the bootstrap observation is the next episode's first, so it is dropped
    np.testing.assert_allclose(np.asarray(discount), [0.81], rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(horizon), [2])
    np.testing.assert_array_equal(np.asarray(mask), [False])


def test_sample_bootstraps_from_the_successor_observation():
    buffer = ReplayBuffer(capacity=16, batch_size=8, n_step=1, gamma=0.9)
    state = _fill(buffer, [1.0] * 10, [False] * 10, [False] * 10)
    batch = jax.jit(buffer.sample)(state, jax.random.key(0))

    obs = np.asarray(batch.obs).reshape(-1)
    np.testing.assert_array_equal(np.asarray(batch.boot_obs).reshape(-1), obs + 1)
    np.testing.assert_allclose(np.asarray(batch.ret), np.ones(8))
    np.testing.assert_allclose(np.asarray(batch.discount), 0.9 * np.ones(8))
    assert np.asarray(batch.mask).all()


def test_sample_masks_every_window_when_every_step_truncates():
    buffer = ReplayBuffer(capacity=16, batch_size=8, n_step=1, gamma=0.9)
    state = _fill(buffer, [1.0] * 10, [False] * 10, [True] * 10)
    batch = jax.jit(buffer.sample)(state, jax.random.key(0))
    assert not np.asarray(batch.mask).any()


def test_can_sample_is_false_before_the_buffer_fills():
    buffer = ReplayBuffer(capacity=16, batch_size=8, n_step=1, gamma=0.9)
    assert not bool(
        buffer.can_sample(_fill(buffer, [1.0] * 3, [False] * 3, [False] * 3))
    )
    assert bool(buffer.can_sample(_fill(buffer, [1.0] * 9, [False] * 9, [False] * 9)))


def test_replay_buffer_requires_capacity_for_one_full_window():
    """Reject capacities too small for an n-step window."""
    buffer = ReplayBuffer(capacity=3, batch_size=1, n_step=3, gamma=0.9)
    with pytest.raises(ValueError, match="capacity must be at least"):
        buffer.init(_Space)


@pytest.mark.parametrize(
    "env_name", ["catch", "cartpole", "mountaincar", "acrobot", "pinball"]
)
def test_buffer_fills_and_samples_from_a_real_env_under_jit(env_name: str):
    """End to end: a jitted rollout of a vector-observation env fills the
    buffer, and the episode cutoff shows up as truncation-masked windows in
    the sample."""
    spec = ENVIRONMENTS[env_name]
    env = spec.build(spec.config_cls(EPISODE_CUTOFF=5))
    num_actions = env.action_space().n
    buffer = ReplayBuffer(capacity=64, batch_size=64, n_step=2, gamma=0.9)

    def rollout(key):
        env_key, scan_key = jax.random.split(key)
        env_state, obs = env.init(env_key)

        def step(carry, _):
            key, env_state, obs, buffer_state = carry
            act_key, env_key, key = jax.random.split(key, 3)
            action = jax.random.randint(
                act_key, (), 0, num_actions, dtype=jnp.int32
            )
            env_state, reward, term, trunc, discount, next_obs = env.step(
                env_state, env_key, action
            )
            buffer_state = buffer.add(
                buffer_state, obs, action, reward, term, trunc, discount
            )
            return (key, env_state, next_obs, buffer_state), trunc

        carry, truncs = jax.lax.scan(
            step,
            (scan_key, env_state, obs, buffer.init(env.observation_space())),
            None,
            length=40,
        )
        return buffer.sample(carry[3], scan_key), truncs

    batch, truncs = jax.block_until_ready(jax.jit(rollout)(jax.random.key(0)))
    assert np.asarray(truncs).sum() == 8  # a cutoff every 5 steps
    assert np.isfinite(np.asarray(batch.ret)).all()
    # gamma ** n for a horizon of 1 or 2, or zero where the episode ended
    discount = np.asarray(batch.discount)[:, None]
    assert np.isclose(discount, [0.0, 0.9, 0.81]).any(axis=1).all()
    assert not np.asarray(batch.mask).all()  # truncated windows dropped


def test_stored_transitions_before_wrap_returns_exact_additions():
    buffer = ReplayBuffer(capacity=4, batch_size=2, n_step=1, gamma=0.9)
    state = _fill(buffer, [1.0, 1.0, 1.0], [False] * 3, [False] * 3)

    result = buffer.stored_transitions(state)

    np.testing.assert_array_equal(np.asarray(result.obs).reshape(-1), [0.0, 1.0, 2.0])


def test_stored_transitions_after_wrap_returns_full_capacity_oldest_first():
    buffer = ReplayBuffer(capacity=4, batch_size=2, n_step=1, gamma=0.9)
    state = _fill(buffer, [1.0] * 6, [False] * 6, [False] * 6)

    result = buffer.stored_transitions(state)

    np.testing.assert_array_equal(
        np.asarray(result.obs).reshape(-1), [2.0, 3.0, 4.0, 5.0]
    )
    assert int(state.size) == 4
