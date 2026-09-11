"""Termination / truncation audit: episode-boundary handling across every agent
and environment, under immediate autoreset.

The RL contract this suite pins down:

* **Environments** report ``termination`` (a real MDP terminal - no bootstrap)
  and ``truncation`` (a time-limit cutoff) as *separate* flags, never merged,
  and never both true.
* **Environments autoreset in place.** Every non-Atari env here (pinball,
  catch, gymnax classic control) is wrapped in
  ``environments.autoreset.AutoresetImmediate``; Atari consumes ale's dead step inside
  its own adapter (see ``test_atari.py``). The step that ends an episode
  reports its own reward and flag but hands back the *next* episode's first
  observation, so no step is dead and episode boundaries fall exactly
  ``cutoff`` steps apart.
* **Agents do not reset anything, and no longer track a dead step.** They act
  on the observation the loop hands them and store what ``update`` receives.
* **Replay agents** store ``(obs, action, reward, termination, truncation)``
  and rely on the stream being unbroken: ``next_obs`` at index ``i`` is always
  ``obs`` at index ``i + 1``, at a boundary as much as anywhere else.
* **A truncated transition must not train.** Its stored successor belongs to
  the next episode, so ``dqn``/``ddqn`` exclude it from the TD loss.
* **Analysis** (``plotting.episode_returns``) segments episodes on the merged
  ``done`` flag, the only form the stored metrics keep.

Fake envs (single float obs, distinguishable reset sentinel) make the stored
transitions inspectable without ale-py/heavy deps; wrapped in
``AutoresetImmediate`` exactly like every other DISABLED-mode env in the repo,
so they exercise the same composition real code uses. See
``test_autoreset.py`` for the wrapper's own jit/vmap unit tests, and
``test_atari.py`` for the ale-py path.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.dqn import DQNAgent, DQNConfig
from agents.random_buffered import RandomBufferAgent, RandomBufferConfig
from components import stored_transitions
from environments.autoreset import AutoresetImmediate
from environments.catch import CatchConfig
from environments.catch import build as build_catch
from environments.classic_control import CartpoleConfig, GymnaxEnv, build_cartpole
from environments.pinball import PinballConfig
from environments.pinball import build as build_pinball
from main import interaction

# --- fake envs (DISABLED mode; single float obs, sentinel reset) ------------


class _Box:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _Discrete:
    def __init__(self, n):
        self.n = int(n)


class _State(NamedTuple):
    counter: jax.Array


class FakeEnv:
    """Single-obs env that ends every ``period`` steps.

    ``obs`` is the (1-based) in-episode step counter as a float, so it is distinct
    from the reset sentinel ``RESET_OBS`` - this lets a test tell the true terminal
    observation apart from the fresh-episode observation in the replay buffer.

    ``mode`` picks whether an episode-end is reported as ``terminated`` or
    ``truncated``. DISABLED-mode: it never resets itself - the caller does. Every
    use below wraps it in ``AutoresetImmediate``, exactly like every real env.
    """

    RESET_OBS = -1.0

    def __init__(self, period=3, mode="terminated", n=2):
        if mode not in ("terminated", "truncated"):
            raise ValueError(f"unknown mode {mode!r}")
        self._period = int(period)
        self._mode = mode
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
        terminated = done & (self._mode == "terminated")
        truncated = done & (self._mode == "truncated")
        return terminal_obs, _State(nc), reward, terminated, truncated, {}


# --- helpers: run an agent, read its replay buffer --------------------------


def _run_agent(agent, env, *, total, buffer_size=64, **overrides):
    if agent == "random_buffered":
        built = RandomBufferAgent(
            RandomBufferConfig(
                TOTAL_TIMESTEPS=total,
                BUFFER_SIZE=buffer_size,
                BATCH_SIZE=2,
                **overrides,
            )
        )
    elif agent == "dqn":
        built = DQNAgent(
            DQNConfig(
                TOTAL_TIMESTEPS=total,
                BUFFER_SIZE=buffer_size,
                BATCH_SIZE=2,
                # no training: pure buffer
                LEARNING_STARTS=total,
                HIDDEN_SIZE=8,
                **overrides,
            )
        )
    else:
        raise ValueError(agent)
    run = jax.jit(lambda key: interaction(key, built, env, total))
    _metrics, final_carry = jax.block_until_ready(run(jax.random.key(0)))
    return final_carry[1]


def _buffer(agent_state):
    """Flat per-transition arrays read from the trajectory buffer, in add
    order: ``{"obs", "next_obs", "reward", "termination", "truncation"}``
    (``next_obs`` is ``obs`` shifted by one index, so it is one shorter;
    every other field has length #adds)."""
    ts = stored_transitions(agent_state.buffer_state)
    obs = np.asarray(ts.obs).reshape(-1)
    return {
        "obs": obs,
        "next_obs": obs[1:],
        "reward": np.asarray(ts.reward).reshape(-1),
        "termination": np.asarray(ts.termination).reshape(-1).astype(bool),
        "truncation": np.asarray(ts.truncation).reshape(-1).astype(bool),
    }


BUFFER_AGENTS = ["random_buffered", "dqn"]


# ===========================================================================
# 1. Environment flag contract (real envs, unwrapped level)
# ===========================================================================


def test_pinball_split_and_truncation_at_cutoff():
    """Pinball reports terminated/truncated separately; a short cutoff truncates
    (terminated stays False) and the two flags are never simultaneously true."""
    env = build_pinball(PinballConfig(SETTING="empty", EPISODE_CUTOFF=5))
    st, _obs = env.init(jax.random.key(0))
    rows = []
    for i in range(5):
        st, r, term, trunc, _obs = env.step(st, jax.random.key(i), jnp.int32(0))
        rows.append((bool(term), bool(trunc), float(r)))
    assert all(not (t and tr) for t, tr, _ in rows)  # never both at once
    assert all(r == -1.0 for *_, r in rows)  # pinball reward is -1/step
    # cutoff at step 5
    assert [tr for _, tr, _ in rows] == [False, False, False, False, True]
    # truncation, not termination
    assert not rows[-1][0]


def test_gymnax_truncation_is_time_limit_only():
    """A gymnax episode ended purely by the step cutoff is truncated, not
    terminated (the wrapper splits gymnax's merged ``done``)."""
    env, params = GymnaxEnv.make("CartPole-v1", 3)
    _obs, st = env.reset(jax.random.key(0))
    rows = []
    for i in range(3):
        _obs, st, _r, term, trunc, _info = env.step(
            jax.random.key(100 + i), st, jnp.int32(i % 2), params
        )
        rows.append((bool(term), bool(trunc)))
    assert rows[:-1] == [(False, False), (False, False)]  # in-episode
    assert rows[-1] == (False, True)  # cutoff -> truncated only


def test_gymnax_real_terminal_is_terminated_not_truncated():
    """A gymnax episode that reaches a real MDP terminal before the cutoff is
    terminated, not truncated; the two are never both true."""
    env, params = GymnaxEnv.make("CartPole-v1", 500)
    _obs, st = env.reset(jax.random.key(0))
    ended = None
    for i in range(500):
        _obs, st, _r, term, trunc, _info = env.step(
            jax.random.key(i), st, jnp.int32(0), params
        )
        assert not (bool(term) and bool(trunc))  # invariant every step
        if bool(term) or bool(trunc):
            ended = (bool(term), bool(trunc))
            break
    # pole fell well before cutoff 500
    assert ended == (True, False)


# ===========================================================================
# 2. Shared env contract: every env autoresets in place, no dead step
# ===========================================================================


def _pinball_env():
    return build_pinball(PinballConfig(SETTING="empty", EPISODE_CUTOFF=5)), 5, 0


def _catch_env():
    return build_catch(CatchConfig(EPISODE_CUTOFF=5)), 5, 1


def _cartpole_env():
    return build_cartpole(CartpoleConfig(EPISODE_CUTOFF=3)), 3, 0


def _fake_terminated_env():
    return AutoresetImmediate(FakeEnv(period=3, mode="terminated")), 3, 0


def _fake_truncated_env():
    return AutoresetImmediate(FakeEnv(period=3, mode="truncated")), 3, 0


ENV_FACTORIES = {
    "pinball": _pinball_env,
    "catch": _catch_env,
    "cartpole": _cartpole_env,
    "fake_terminated": _fake_terminated_env,
    "fake_truncated": _fake_truncated_env,
}


@pytest.mark.parametrize("env_name", list(ENV_FACTORIES))
def test_env_contract_boundaries_are_exactly_a_cutoff_apart(env_name):
    """Shared audit, every env: with a cutoff of ``k``, boundaries land on every
    ``k``-th step and never carry both flags at once.

    This is the property that separates immediate autoreset from NEXT_STEP: no
    step is spent replaying a boundary, so the spacing is ``k`` rather than
    ``k + 1``. Each env here is driven with an action that cannot end an episode
    early, so the cutoff is the only thing that ends one."""
    env, cutoff, action = ENV_FACTORIES[env_name]()

    state, _obs = env.init(jax.random.key(0))
    flags = []
    for i in range(3 * cutoff):
        state, _r, term, trunc, _obs = env.step(
            state, jax.random.key(1000 + i), jnp.int32(action)
        )
        assert not (bool(term) and bool(trunc))
        flags.append(bool(term) or bool(trunc))

    ends = np.flatnonzero(flags)
    np.testing.assert_array_equal(ends, np.arange(1, 4) * cutoff - 1)


def test_fake_env_boundary_hands_back_the_fresh_observation():
    """The boundary step keeps its own reward and flag but returns the next
    episode's first observation - the true final obs is not observable."""
    env = AutoresetImmediate(FakeEnv(period=3, mode="terminated"))
    state, _obs = env.init(jax.random.key(0))
    rows = []
    for i in range(4):
        state, r, term, _trunc, obs = env.step(state, jax.random.key(i), jnp.int32(0))
        rows.append((float(obs[0]), float(r), bool(term)))
    assert rows == [
        (1.0, 1.0, False),
        (2.0, 1.0, False),
        (FakeEnv.RESET_OBS, 1.0, True),  # not the terminal counter 3.0
        (1.0, 1.0, False),  # the new episode, already running
    ]


def test_gymnax_split_survives_wrapping_past_the_boundary():
    """The terminated/truncated split (test 1, above) still holds once
    AutoresetImmediate composes on top: the cutoff step truncates and the step
    after it is an ordinary step of the next episode."""
    env = build_cartpole(CartpoleConfig(EPISODE_CUTOFF=3))
    state, _obs = env.init(jax.random.key(0))
    rows = []
    for i in range(4):
        state, _r, term, trunc, _obs = env.step(state, jax.random.key(i), jnp.int32(0))
        rows.append((bool(term), bool(trunc)))
    assert rows[:2] == [(False, False), (False, False)]
    assert rows[2] == (False, True)  # cutoff -> truncated only
    assert rows[3] == (False, False)  # an ordinary step, not a dead one


# ===========================================================================
# 3. Replay-buffer boundary handling: the env resets itself, in place
# ===========================================================================


@pytest.mark.parametrize("agent", BUFFER_AGENTS)
@pytest.mark.parametrize("mode", ["terminated", "truncated"])
def test_buffer_boundary_successor_is_the_next_episodes_first_obs(agent, mode):
    """At a boundary the stored successor (the next entry's ``obs``) is the new
    episode's first observation, and the flag is on the transition that earned
    it. The entry after a boundary starts the new episode for real."""
    env = AutoresetImmediate(FakeEnv(period=3, mode=mode))
    buf = _buffer(_run_agent(agent, env, total=9))

    flag = buf["termination" if mode == "terminated" else "truncation"]
    other = buf["truncation" if mode == "terminated" else "termination"]
    ends = np.flatnonzero(flag)

    assert list(ends) == [2, 5, 8]  # every cutoff step, nothing between
    assert not other.any()
    # the successor of a boundary transition is the fresh reset obs
    np.testing.assert_array_equal(buf["next_obs"][ends[:-1]], [FakeEnv.RESET_OBS] * 2)
    # ... which is exactly the next entry's own obs: the stream is unbroken
    np.testing.assert_array_equal(buf["obs"][ends[:-1] + 1], [FakeEnv.RESET_OBS] * 2)
    assert buf["obs"][0] == FakeEnv.RESET_OBS


@pytest.mark.parametrize("agent", BUFFER_AGENTS)
def test_no_stored_transition_is_dead(agent):
    """Immediate autoreset spends no step replaying a boundary, so every stored
    transition carries a real action and this fake's reward of 1."""
    env = AutoresetImmediate(FakeEnv(period=3, mode="truncated"))
    buf = _buffer(_run_agent(agent, env, total=10))
    assert (buf["reward"] == 1.0).all()
    np.testing.assert_array_equal(
        np.flatnonzero(buf["termination"] | buf["truncation"]), [2, 5, 8]
    )


# ===========================================================================
# 4. DQN update rule: termination masks the bootstrap, truncation drops
# ===========================================================================


def _dqn_q_leaves(env, *, seed=0, **overrides):
    """Run DQN (with training on) on ``env`` and return its online-Q array leaves."""
    hypers = {
        "TOTAL_TIMESTEPS": 80,
        "BUFFER_SIZE": 256,
        "BATCH_SIZE": 8,
        "LEARNING_STARTS": 8,
        "TRAIN_FREQUENCY": 1,
        "TARGET_NETWORK_FREQUENCY": 10,
        "HIDDEN_SIZE": 16,
        # all-random actions -> identical data
        "EPSILON_START": 1.0,
        "EPSILON_END": 1.0,
    }
    agent = DQNAgent(DQNConfig(**{**hypers, **overrides}))
    run = jax.jit(lambda key: interaction(key, agent, env, 80))
    _metrics, final_carry = run(jax.random.key(seed))
    return jax.tree.leaves(eqx.filter(final_carry[1].q, eqx.is_array))


def test_dqn_treats_termination_and_truncation_differently():
    """Two runs with *identical* dynamics, rewards and observations that differ
    only in whether the episode-end is labelled terminated or truncated must
    learn different Q-functions: a terminated boundary trains with its bootstrap
    masked off, while a truncated one is dropped from the loss entirely. If the
    code merged the two flags into ``done`` the runs would be identical."""
    term_env = AutoresetImmediate(FakeEnv(period=4, mode="terminated"))
    trunc_env = AutoresetImmediate(FakeEnv(period=4, mode="truncated"))

    # sanity: the two runs really do differ only in the boundary flag
    tb = _buffer(_run_agent("random_buffered", term_env, total=40))
    ub = _buffer(_run_agent("random_buffered", trunc_env, total=40))
    np.testing.assert_array_equal(tb["obs"], ub["obs"])
    np.testing.assert_array_equal(tb["next_obs"], ub["next_obs"])
    assert tb["termination"].any() and not tb["truncation"].any()
    assert ub["truncation"].any() and not ub["termination"].any()

    term_leaves = _dqn_q_leaves(term_env)
    trunc_leaves = _dqn_q_leaves(trunc_env)
    # at least one weight differs -> the update rule distinguishes the two flags
    assert any(
        not np.allclose(a, b) for a, b in zip(term_leaves, trunc_leaves, strict=True)
    )


def test_dqn_is_deterministic_given_a_seed():
    """A control: two runs of the same env with the same seed are bit-identical,
    guarding the comparison above against dependence on anything but the flag."""
    a = _dqn_q_leaves(AutoresetImmediate(FakeEnv(period=4, mode="truncated")), seed=1)
    b = _dqn_q_leaves(AutoresetImmediate(FakeEnv(period=4, mode="truncated")), seed=1)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_an_entirely_truncated_stream_does_not_train():
    """Every transition of a ``period=1`` truncated env is dropped from the TD
    loss, so the whole batch is masked out. The loss is then 0 rather than a
    0/0 NaN, no gradient reaches the network, and the online Q-function comes
    out exactly as it went in."""
    env = AutoresetImmediate(FakeEnv(period=1, mode="truncated"))
    trained = _dqn_q_leaves(env)
    # LEARNING_STARTS above the step budget means no training at all
    untrained = _dqn_q_leaves(env, LEARNING_STARTS=80)

    for a, b in zip(trained, untrained, strict=True):
        assert np.isfinite(a).all()
        np.testing.assert_array_equal(a, b)


def test_an_entirely_terminated_stream_does_train():
    """A control for the test above: the same env labelling its boundary
    terminated instead keeps every transition, so training does move the
    network. Otherwise the equality above would hold for the wrong reason."""
    env = AutoresetImmediate(FakeEnv(period=1, mode="terminated"))
    trained = _dqn_q_leaves(env)
    untrained = _dqn_q_leaves(env, LEARNING_STARTS=80)

    assert any(not np.allclose(a, b) for a, b in zip(trained, untrained, strict=True))
