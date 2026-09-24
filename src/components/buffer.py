from typing import NamedTuple

import jax
import jax.numpy as jnp


class TimeStep(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    termination: jax.Array
    truncation: jax.Array


class Batch(NamedTuple):
    obs: jax.Array
    action: jax.Array
    ret: jax.Array
    discount: jax.Array
    boot_obs: jax.Array
    mask: jax.Array


class BufferState(NamedTuple):
    # data.obs stores each step's newest frame, not the stacked observation.
    data: TimeStep
    first: jax.Array
    head: jax.Array
    size: jax.Array
    prev_done: jax.Array


def build_buffer(name: str, **kwargs):
    if name == "uniform":
        return ReplayBuffer(**kwargs)
    else:
        raise ValueError(f"unknown buffer {name!r}")


def sample_windows(
    key: jax.Array,
    head: jax.Array,
    size: jax.Array,
    capacity: int,
    lookback: int,
    lookahead: int,
    batch_size: int,
) -> jax.Array:
    """Sample uniformly from valid starts in the retained chronological range."""
    # Before full, episode-start padding covers lookback before the oldest slot.
    skip = jnp.where(size == capacity, lookback, 0)
    oldest = (head - size) % capacity
    count = size - skip - lookahead
    starts = (
        oldest + skip + jax.random.randint(key, (batch_size,), 0, count)
    ) % capacity
    offsets = jnp.arange(-lookback, lookahead + 1, dtype=jnp.int32)
    return ((starts[:, None] + offsets[None, :]) % capacity).astype(jnp.int32)


def stack_frames(frames: jax.Array, first: jax.Array) -> jax.Array:
    """Mask frames from prior episodes and concatenate channels chronologically."""
    k_axis = first.ndim - 1
    k_size = frames.shape[k_axis]
    positions = jnp.arange(k_size)
    last_start = jnp.max(jnp.where(first, positions, -1), axis=-1)
    valid = positions >= last_start[..., None]
    valid = valid.reshape(first.shape + (1,) * (frames.ndim - first.ndim))
    frames = jnp.where(valid, frames, 0)
    frames = jnp.moveaxis(frames, k_axis, -2)
    return frames.reshape(frames.shape[:-2] + (k_size * frames.shape[-1],))


def n_step_return(
    batch: TimeStep, gamma: float, n_step: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Accumulate n-step returns and report each realised horizon.

    Returns ``(ret, discount, horizon, mask)`` for windows cut at their first
    episode boundary. ``horizon`` is the realised transition count as int32;
    ``discount`` is ``gamma ** horizon``, or zero when the cut is a termination.
    ``mask`` is ``False`` when the cut is a truncation, whose bootstrap
    observation belongs to the next episode and cannot be trained on.
    """
    done = (batch.termination | batch.truncation)[:, :-1]
    alive = jnp.cumprod(1.0 - done.astype(jnp.float32), axis=1)
    w = jnp.concatenate([jnp.ones_like(alive[:, :1]), alive[:, :-1]], axis=1)
    ret = jnp.sum(w * gamma ** jnp.arange(n_step) * batch.reward[:, :n_step], axis=1)
    n = jnp.sum(w, axis=1).astype(jnp.int32)
    cut_idx = (n - 1)[:, None]
    term_cut = jnp.take_along_axis(
        batch.termination[:, :n_step], cut_idx, axis=1
    ).squeeze(1)
    trunc_cut = jnp.take_along_axis(
        batch.truncation[:, :n_step], cut_idx, axis=1
    ).squeeze(1)
    discount = gamma**n * (1.0 - term_cut.astype(jnp.float32))
    return ret, discount, n, ~trunc_cut


class ReplayBuffer:
    def __init__(self, *, capacity: int, batch_size: int, n_step: int, gamma: float):
        self._capacity = capacity
        self._batch_size = batch_size
        self._n_step = n_step
        self._gamma = gamma

    def init(self, observation_space) -> BufferState:
        obs_shape = observation_space.shape
        frame_channels = getattr(observation_space, "frame_channels", None)
        if frame_channels is None:
            self._frame_channels = obs_shape[-1]
            frame_shape = obs_shape
            self._stack_size = 1
        else:
            self._frame_channels = frame_channels
            frame_shape = obs_shape[:-1] + (frame_channels,)
            self._stack_size = obs_shape[-1] // frame_channels
        if self._capacity < self._stack_size + self._n_step:
            raise ValueError("capacity must be at least stack_size + n_step")
        return BufferState(
            data=TimeStep(
                obs=jnp.zeros(
                    (self._capacity, *frame_shape), observation_space.dtype
                ),
                action=jnp.zeros((self._capacity,), jnp.int32),
                reward=jnp.zeros((self._capacity,), jnp.float32),
                termination=jnp.zeros((self._capacity,), jnp.bool_),
                truncation=jnp.zeros((self._capacity,), jnp.bool_),
            ),
            first=jnp.zeros((self._capacity,), jnp.bool_),
            head=jnp.asarray(0, jnp.int32),
            size=jnp.asarray(0, jnp.int32),
            prev_done=jnp.asarray(True),
        )

    def add(
        self,
        state: BufferState,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        termination: jax.Array,
        truncation: jax.Array,
    ) -> BufferState:
        data = TimeStep(
            obs[..., -self._frame_channels:], action, reward, termination, truncation
        )
        return BufferState(
            data=jax.tree.map(
                lambda buf, value: buf.at[state.head].set(value), state.data, data
            ),
            first=state.first.at[state.head].set(state.prev_done),
            head=(state.head + 1) % self._capacity,
            size=jnp.minimum(state.size + 1, self._capacity),
            prev_done=termination | truncation,
        )

    def sample(self, state: BufferState, key: jax.Array) -> Batch:
        k = self._stack_size
        indices = sample_windows(
            key,
            state.head,
            state.size,
            self._capacity,
            k - 1,
            self._n_step,
            self._batch_size,
        )
        transition_slots = indices[:, k - 1 :]
        transitions = jax.tree.map(lambda x: x[transition_slots], state.data)
        ret, discount, horizon, mask = n_step_return(
            transitions, self._gamma, self._n_step
        )
        obs_slots = indices[:, :k]
        obs = stack_frames(state.data.obs[obs_slots], state.first[obs_slots])
        boot_slots = jnp.take_along_axis(
            indices, horizon[:, None] + jnp.arange(k)[None, :], axis=1
        )
        boot_obs = stack_frames(
            state.data.obs[boot_slots], state.first[boot_slots]
        )
        return Batch(
            obs=obs,
            action=transitions.action[:, 0],
            ret=ret,
            discount=discount,
            boot_obs=boot_obs,
            mask=mask,
        )

    def can_sample(self, state: BufferState) -> jax.Array:
        return state.size >= max(self._batch_size, self._n_step + 1)

    def stored_transitions(self, state: BufferState) -> TimeStep:
        """Return all stored transitions in add order, oldest first."""
        size = int(state.size)
        skip = self._stack_size - 1 if size == self._capacity else 0
        oldest = (int(state.head) - size) % self._capacity
        indices = (oldest + jnp.arange(skip, size)) % self._capacity
        frame_indices = (
            indices[:, None] + jnp.arange(1 - self._stack_size, 1)[None, :]
        ) % self._capacity
        frames = state.data.obs[frame_indices]
        first = state.first[frame_indices]
        obs = stack_frames(frames, first)
        return TimeStep(
            obs=obs,
            action=state.data.action[indices],
            reward=state.data.reward[indices],
            termination=state.data.termination[indices],
            truncation=state.data.truncation[indices],
        )
