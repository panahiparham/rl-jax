"""Unit tests for the AutoresetImmediate wrapper.

Uses a small local fake env with a distinguishable reset sentinel to check
the immediate-autoreset contract in isolation: the boundary step keeps its
own reward and flag but hands back the next episode's first observation, and
no step is dead. Also checks the wrapper composes under jax.jit and jax.vmap,
and does an end-to-end smoke check against the real gymnax adapter.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from environments.autoreset import AutoresetImmediate
from environments.classic_control import GymnaxEnv


class _Box:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _Discrete:
    def __init__(self, n):
        self.n = int(n)


class _State(NamedTuple):
    counter: jax.Array


class _BoundaryEnv:
    """DISABLED-mode env that ends every ``period`` steps.

    ``obs`` is the in-episode step counter (1-based) as a float, distinct
    from the reset sentinel ``RESET_OBS``. Never resets itself - matches
    every real env in this repo, and is what AutoresetImmediate wraps.
    """

    RESET_OBS = -1.0

    def __init__(self, period=3, n=2):
        self._period = int(period)
        self._n = int(n)

    def observation_space(self, params=None):
        return _Box((1,), jnp.float32)

    def action_space(self, params=None):
        return _Discrete(self._n)

    def reset(self, key, params=None):
        obs = jnp.asarray([self.RESET_OBS], jnp.float32)
        return obs, _State(jnp.asarray(0, jnp.int32))

    def step(self, key, state, action, params=None):
        del key, action
        nc = state.counter + 1
        done = (nc % self._period) == 0
        terminal_obs = nc.astype(jnp.float32).reshape((1,))
        reward = jnp.asarray(1.0, jnp.float32)
        return terminal_obs, _State(nc), reward, done, jnp.asarray(False), {}


def _rollout_immediate(env, n_steps, actions=None):
    state, _obs = env.init(jax.random.key(0))
    rows = []
    for i in range(n_steps):
        action = actions[i] if actions is not None else jnp.int32(0)
        state, r, term, trunc, _discount, next_obs = env.step(
            state, jax.random.key(100 + i), action
        )
        rows.append((float(next_obs[0]), float(r), bool(term), bool(trunc)))
    return rows


def test_immediate_boundary_returns_the_fresh_obs():
    """The boundary step keeps its own reward and flag but hands back the new
    episode's initial observation, so the true final obs is not visible."""
    env = AutoresetImmediate(_BoundaryEnv(period=3))
    rows = _rollout_immediate(env, 7)
    obs, reward, term, trunc = zip(*rows, strict=True)
    assert list(zip(term, trunc, strict=True)) == [
        (False, False),
        (False, False),
        (True, False),
        (False, False),
        (False, False),
        (True, False),
        (False, False),
    ]
    assert obs[2] == _BoundaryEnv.RESET_OBS  # not the terminal counter, 3.0
    assert reward[2] == 1.0  # the boundary reward survives


def test_immediate_has_no_dead_step():
    env = AutoresetImmediate(_BoundaryEnv(period=3))
    obs, reward, _term, _trunc = zip(*_rollout_immediate(env, 7), strict=True)
    assert all(r == 1.0 for r in reward)
    # counter restarts right after the boundary rather than running on to 4
    assert obs[3] == 1.0
    assert obs[:3] == (1.0, 2.0, _BoundaryEnv.RESET_OBS)


def test_immediate_state_resets_on_the_boundary_only():
    env = AutoresetImmediate(_BoundaryEnv(period=3))
    state, _obs = env.init(jax.random.key(0))
    counters = []
    for i in range(4):
        state, *_ = env.step(state, jax.random.key(i), jnp.int32(0))
        counters.append(int(state.counter))
    assert counters == [1, 2, 0, 1]


def test_immediate_under_jit():
    env = AutoresetImmediate(_BoundaryEnv(period=3))

    @jax.jit
    def rollout(state, keys):
        def one(st, k):
            st, r, term, trunc, _discount, next_obs = env.step(st, k, jnp.int32(0))
            return st, (next_obs[0], r, term, trunc)

        return jax.lax.scan(one, state, keys)

    state0, _obs = env.init(jax.random.key(0))
    _, (obs, r, term, _trunc) = rollout(state0, jax.random.split(jax.random.key(1), 7))
    assert term.tolist() == [False, False, True, False, False, True, False]
    assert obs.tolist()[2] == _BoundaryEnv.RESET_OBS
    assert r.tolist() == [1.0] * 7


@pytest.fixture(scope="module")
def vmapped_rollout():
    env = AutoresetImmediate(_BoundaryEnv(period=3))

    def rollout(key):
        state0, _obs = env.init(key)

        def one(st, k):
            st, r, term, trunc, _discount, next_obs = env.step(st, k, jnp.int32(0))
            return st, (next_obs[0], r, term, trunc)

        return jax.lax.scan(one, state0, jax.random.split(key, 7))[1]

    seeds = jax.vmap(jax.random.key)(jnp.arange(4))
    return jax.vmap(rollout)(seeds)


def test_immediate_under_vmap_shape(vmapped_rollout):
    obs, _r, _term, _trunc = vmapped_rollout
    assert obs.shape == (4, 7)


@pytest.mark.parametrize("row", range(4))
def test_immediate_under_vmap(vmapped_rollout, row):
    obs, r, term, _trunc = vmapped_rollout
    assert term.tolist()[row] == [False, False, True, False, False, True, False]
    assert obs.tolist()[row][2] == _BoundaryEnv.RESET_OBS
    assert r.tolist()[row] == [1.0] * 7


def test_immediate_real_gymnax_env_smoke():
    """Sanity check against a real env: the cutoff truncates in place."""
    inner, _params = GymnaxEnv.make("CartPole-v1", 3)
    env = AutoresetImmediate(inner)
    rows = _rollout_immediate(env, 4)
    _obs, _reward, term, trunc = zip(*rows, strict=True)
    assert list(zip(term, trunc, strict=True)) == [
        (False, False),
        (False, False),
        (False, True),
        (False, False),
    ]
