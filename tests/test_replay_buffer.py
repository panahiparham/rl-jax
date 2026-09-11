from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from components import ReplayBuffer, TimeStep, n_step_return, stored_transitions
from environments import ENVIRONMENTS
from environments.catch import CatchConfig


class _Space:
    shape = (1,)
    dtype = jnp.float32


def _window(reward, termination, truncation, obs):
    return TimeStep(
        obs=jnp.array([obs]),
        action=jnp.zeros((1, len(reward)), jnp.int32),
        reward=jnp.array([reward]),
        termination=jnp.array([termination]),
        truncation=jnp.array([truncation]),
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
        )
    return state


def test_n_step_return_runs_to_the_end_of_an_unbroken_window():
    batch = _window(
        [1.0, 2.0, 3.0, 0.0],
        [False] * 4,
        [False] * 4,
        [[10.0], [20.0], [30.0], [40.0]],
    )
    ret, discount, boot_obs, mask = n_step_return(batch, 0.9, 3)
    np.testing.assert_allclose(np.asarray(ret), [5.23], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(discount), [0.729], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(boot_obs), [[40.0]])
    np.testing.assert_array_equal(np.asarray(mask), [True])


def test_termination_cuts_the_window_and_zeroes_the_discount():
    batch = _window(
        [1.0, 2.0, 3.0, 0.0],
        [False, True, False, False],
        [False] * 4,
        [[10.0], [20.0], [30.0], [40.0]],
    )
    ret, discount, boot_obs, mask = n_step_return(batch, 0.9, 3)
    np.testing.assert_allclose(np.asarray(ret), [2.8], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(discount), [0.0])
    np.testing.assert_allclose(np.asarray(boot_obs), [[30.0]])
    # a terminal window still trains: the discount alone removes the bootstrap
    np.testing.assert_array_equal(np.asarray(mask), [True])


def test_truncation_cuts_the_window_and_clears_the_mask():
    batch = _window(
        [1.0, 2.0, 3.0, 0.0],
        [False] * 4,
        [False, True, False, False],
        [[10.0], [20.0], [30.0], [40.0]],
    )
    ret, discount, boot_obs, mask = n_step_return(batch, 0.9, 3)
    np.testing.assert_allclose(np.asarray(ret), [2.8], rtol=1e-6)
    # the bootstrap observation is the next episode's first, so it is dropped
    np.testing.assert_allclose(np.asarray(discount), [0.81], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(boot_obs), [[30.0]])
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


def test_buffer_fills_and_samples_from_a_real_env_under_jit():
    """End to end: a jitted rollout of catch fills the buffer, and the
    episode cutoff shows up as truncation-masked windows in the sample."""
    env = ENVIRONMENTS["catch"].build(CatchConfig(EPISODE_CUTOFF=5))
    buffer = ReplayBuffer(capacity=64, batch_size=64, n_step=2, gamma=0.9)

    def rollout(key):
        env_key, scan_key = jax.random.split(key)
        env_state, obs = env.init(env_key)

        def step(carry, _):
            key, env_state, obs, buffer_state = carry
            act_key, env_key, key = jax.random.split(key, 3)
            action = jax.random.randint(act_key, (), 0, 3, dtype=jnp.int32)
            env_state, reward, term, trunc, next_obs = env.step(
                env_state, env_key, action
            )
            buffer_state = buffer.add(buffer_state, obs, action, reward, term, trunc)
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

    result = stored_transitions(state)

    np.testing.assert_array_equal(np.asarray(result.obs).reshape(-1), [0.0, 1.0, 2.0])


def test_stored_transitions_after_wrap_returns_full_capacity_oldest_first():
    buffer = ReplayBuffer(capacity=4, batch_size=2, n_step=1, gamma=0.9)
    state = _fill(buffer, [1.0] * 6, [False] * 6, [False] * 6)

    result = stored_transitions(state)

    np.testing.assert_array_equal(
        np.asarray(result.obs).reshape(-1), [2.0, 3.0, 4.0, 5.0]
    )
    assert bool(state.is_full)
