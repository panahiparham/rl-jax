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
    data: TimeStep
    head: jax.Array
    size: jax.Array


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
        if capacity < n_step + 1:
            raise ValueError("capacity must be at least n_step + 1")
        self._capacity = capacity
        self._batch_size = batch_size
        self._n_step = n_step
        self._gamma = gamma

    def init(self, observation_space) -> BufferState:
        return BufferState(
            data=TimeStep(
                obs=jnp.zeros(
                    (self._capacity, *observation_space.shape), observation_space.dtype
                ),
                action=jnp.zeros((self._capacity,), jnp.int32),
                reward=jnp.zeros((self._capacity,), jnp.float32),
                termination=jnp.zeros((self._capacity,), jnp.bool_),
                truncation=jnp.zeros((self._capacity,), jnp.bool_),
            ),
            head=jnp.asarray(0, jnp.int32),
            size=jnp.asarray(0, jnp.int32),
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
        data = TimeStep(obs, action, reward, termination, truncation)
        return BufferState(
            data=jax.tree.map(
                lambda buf, value: buf.at[state.head].set(value), state.data, data
            ),
            head=(state.head + 1) % self._capacity,
            size=jnp.minimum(state.size + 1, self._capacity),
        )

    def sample(self, state: BufferState, key: jax.Array) -> Batch:
        indices = sample_windows(
            key,
            state.head,
            state.size,
            self._capacity,
            0,
            self._n_step,
            self._batch_size,
        )
        window = jax.tree.map(lambda x: x[indices], state.data)
        ret, discount, horizon, mask = n_step_return(
            window, self._gamma, self._n_step
        )
        obs_idx = horizon.reshape((-1,) + (1,) * (window.obs.ndim - 1))
        boot_obs = jnp.take_along_axis(window.obs, obs_idx, axis=1).squeeze(1)
        return Batch(
            obs=window.obs[:, 0],
            action=window.action[:, 0],
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
        oldest = (int(state.head) - size) % self._capacity
        indices = (oldest + jnp.arange(size)) % self._capacity
        return jax.tree.map(lambda x: x[indices], state.data)
