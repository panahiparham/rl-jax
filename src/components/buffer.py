from typing import Any, NamedTuple

import flashbax as fbx
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


def stored_transitions(state: Any) -> TimeStep:
    """Return all stored transitions in add order, oldest first.

    Once the buffer wraps, this includes the entire capacity.
    """
    n = int(state.current_index)
    stored = jax.tree.map(lambda x: x[0], state.experience)
    if bool(state.is_full):
        return jax.tree.map(lambda x: jnp.roll(x, -n, axis=0), stored)
    return jax.tree.map(lambda x: x[:n], stored)


class ReplayBuffer:
    def __init__(self, *, capacity: int, batch_size: int, n_step: int, gamma: float):
        self._n_step = n_step
        self._gamma = gamma
        self._buffer = fbx.make_trajectory_buffer(
            add_batch_size=1,
            sample_batch_size=batch_size,
            sample_sequence_length=n_step + 1,
            period=1,
            min_length_time_axis=max(batch_size, n_step + 1),
            max_length_time_axis=capacity,
        )

    def init(self, observation_space):
        return self._buffer.init(
            TimeStep(
                obs=jnp.zeros(observation_space.shape, observation_space.dtype),
                action=jnp.asarray(0, jnp.int32),
                reward=jnp.asarray(0.0, jnp.float32),
                termination=jnp.asarray(False),
                truncation=jnp.asarray(False),
            )
        )

    def add(
        self,
        state: Any,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        termination: jax.Array,
        truncation: jax.Array,
    ):
        timestep = TimeStep(
            obs=obs,
            action=action,
            reward=reward,
            termination=termination,
            truncation=truncation,
        )
        return self._buffer.add(
            state, jax.tree.map(lambda x: x[None, None, ...], timestep)
        )

    def sample(self, state: Any, key: jax.Array):
        window = self._buffer.sample(state, key).experience
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

    def can_sample(self, state: Any) -> jax.Array:
        return self._buffer.can_sample(state)
