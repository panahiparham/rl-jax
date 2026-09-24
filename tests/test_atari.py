"""Tests for the Atari env wrapper and DQN-on-image-obs.

Logic tests drive ``AtariEnvLike`` against a fake vector env reproducing ale-py's
``.xla()`` FFI contract (opaque ``(8,)`` handle, NEXT_STEP autoreset), and run the
``nature_cnn`` DQN on a fake image env - all without ale-py, so they run anywhere.
The real ale-py tests self-skip when the XLA build is not installed.

Note: on macOS-CPU, ale-py's XLA FFI can intermittently *segfault* when the
episode-boundary reset consume runs under the DQN graph (it is solid on
Linux-CUDA, where Atari training actually happens).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.dqn import DQNAgent, DQNConfig
from agents.random_buffered import RandomBufferAgent, RandomBufferConfig
from components import ReplayBuffer
from environments import ENVIRONMENTS
from environments.atari import AtariConfig, AtariEnv, AtariEnvLike
from environments.autoreset import AutoresetImmediate
from main import interaction


class _Box:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _Discrete:
    def __init__(self, n):
        self.n = int(n)


_START_LIVES = 5


class _FakeVectorEnv:
    """Mimics ``ale_py.AtariVectorEnv`` + ``.xla()`` (jittable, no ale-py).

    Terminates every ``period`` steps and, when ``life_period`` is set, loses a
    life every ``life_period`` steps. Obs frames carry the step counter so a reset
    (value 0) is distinguishable. Models ale's NEXT_STEP autoreset via a pending
    flag in the handle: a terminal step sets it; the next ``step_fn`` returns a fresh
    obs.
    """

    def __init__(
        self,
        num_envs: int = 1,
        period: int = 3,
        frames: int = 4,
        h: int = 84,
        w: int = 84,
        n: int = 6,
        colours: int = 0,
        life_period: int = 0,
    ) -> None:
        self.num_envs = num_envs
        self._life_period = life_period
        self._period, self._frames, self._h, self._w = period, frames, h, w
        self._colours = colours
        obs_shape = (frames, h, w, colours) if colours else (frames, h, w)
        self.single_observation_space = _Box(obs_shape, jnp.uint8)
        self.single_action_space = _Discrete(n)

    def xla(self):
        frames, h, w, period = self._frames, self._h, self._w, self._period
        colours, life_period = self._colours, self._life_period
        obs_shape = (1, frames, h, w, colours) if colours else (1, frames, h, w)

        def obs(frame_values):
            per_frame = frame_values.reshape((1, frames) + (1,) * (len(obs_shape) - 2))
            return jnp.broadcast_to(per_frame, obs_shape)

        init = jnp.zeros((8,), jnp.uint8)  # count, reset flag, rolling frames

        def lives(count):
            lost = count.astype(jnp.int32) // life_period if life_period else 0
            return (jnp.int32(_START_LIVES) - lost).reshape((1,))

        def reset_fn(handle, seed):
            fresh = jnp.zeros((8,), jnp.uint8)
            info = {"lives": lives(fresh[0])}
            return fresh, (obs(jnp.zeros((frames,), jnp.uint8)), info)

        def step_fn(handle, actions):
            is_reset = handle[1] > 0
            count = handle[0] + jnp.uint8(1)
            term = (count % period == 0) & ~is_reset
            new_handle = (
                handle.at[0]
                .set(jnp.where(is_reset, jnp.uint8(0), count))
                .at[1]
                .set(jnp.where(term, jnp.uint8(1), jnp.uint8(0)))
            )
            new_frame = jnp.where(is_reset, jnp.uint8(0), count)
            frame_values = jnp.concatenate(
                (handle[3 : 2 + frames], new_frame[None])
            )
            frame_values = jnp.where(
                is_reset, jnp.zeros_like(frame_values), frame_values
            )
            new_handle = new_handle.at[2 : 2 + frames].set(frame_values)
            return new_handle, (
                obs(frame_values),
                jnp.where(is_reset, 0.0, 1.0).reshape((1,)).astype(jnp.float32),
                term.reshape((1,)),
                jnp.zeros((1,), bool),
                {"lives": lives(new_handle[0])},
            )

        return init, reset_fn, step_fn


def test_spaces_and_dtype():
    env = AtariEnvLike(_FakeVectorEnv(n=6))
    assert env.observation_space().shape == (84, 84, 4)
    assert env.observation_space().dtype == jnp.uint8
    assert env.observation_space().frame_channels == 1
    assert env.action_space().n == 6
    # no env in this repo auto-resets; the agent does
    assert not hasattr(env, "auto_resets")


def test_reset_and_step_shapes_and_transpose():
    env = AtariEnvLike(_FakeVectorEnv())
    obs, state = env.reset(jax.random.key(0))
    # frames -> channels
    assert obs.shape == (84, 84, 4) and obs.dtype == jnp.uint8
    obs2, _state2, reward, terminated, truncated, info = env.step(
        jax.random.key(1), state, jnp.int32(0)
    )
    assert obs2.shape == (84, 84, 4)
    assert reward.shape == () and terminated.shape == () and truncated.shape == ()
    # separate flags, not merged
    assert not bool(terminated) and not bool(truncated)
    assert info == {}


def test_no_in_step_reset_returns_true_boundary_obs():
    """The wrapper does NOT absorb ale's autoreset: on a terminal step it returns the
    *true*
    terminal observation (counter 3), not the fresh episode's (0). This is what makes a
    boundary transition store the state the agent actually reached (issue #1)."""
    env = AtariEnvLike(_FakeVectorEnv(period=3))
    _, state = env.reset(jax.random.key(0))
    seen = []
    for _ in range(3):
        obs, state, _r, terminated, _tr, _i = env.step(
            jax.random.key(1), state, jnp.int32(0)
        )
        seen.append((int(obs[0, 0, -1]), bool(terminated)))
    # obs 3 = true terminal, not reset(0)
    assert seen == [(1, False), (2, False), (3, True)]


def test_immediate_autoreset_returns_the_fresh_obs_on_the_boundary():
    """AtariEnv consumes ale's dead step in place: the terminal step reports its
    own reward and flag but hands back the new episode's first observation."""
    env = AtariEnv(AtariEnvLike(_FakeVectorEnv(period=3)))
    state, _obs = env.init(jax.random.key(0))
    seen = []
    for i in range(7):
        state, reward, term, trunc, _discount, obs = env.step(
            state, jax.random.key(i), jnp.int32(0)
        )
        seen.append((int(obs[0, 0, -1]), float(reward), bool(term), bool(trunc)))
    assert seen == [
        (1, 1.0, False, False),
        (2, 1.0, False, False),
        (0, 1.0, True, False),
        (1, 1.0, False, False),
        (2, 1.0, False, False),
        (0, 1.0, True, False),
        (1, 1.0, False, False),
    ]


def test_no_dead_step_reaches_the_caller():
    """Every step advances the emulator - none reports the dead step's reward=0."""
    env = AtariEnv(AtariEnvLike(_FakeVectorEnv(period=3)))
    state, _obs = env.init(jax.random.key(0))
    rewards = []
    for i in range(6):
        state, reward, _te, _tr, _discount, _obs = env.step(
            state, jax.random.key(i), jnp.int32(0)
        )
        rewards.append(float(reward))
    assert rewards == [1.0] * 6


def test_non_boundary_step_does_not_double_step_the_emulator():
    """The extra step_fn call is paid only on a boundary, so a plain step
    advances ale's counter exactly once."""
    env = AtariEnv(AtariEnvLike(_FakeVectorEnv(period=10)))
    state, _obs = env.init(jax.random.key(0))
    for i in range(3):
        state, _r, _te, _tr, _discount, obs = env.step(
            state, jax.random.key(i), jnp.int32(0)
        )
    assert int(obs[0, 0, -1]) == 3


def test_immediate_autoreset_under_jit():
    env = AtariEnv(AtariEnvLike(_FakeVectorEnv(period=3)))

    @jax.jit
    def rollout(state, keys):
        def one(st, k):
            st, r, term, trunc, _discount, obs = env.step(st, k, jnp.int32(0))
            return st, (obs[0, 0, -1], r, term, trunc)

        return jax.lax.scan(one, state, keys)

    state0, _obs = env.init(jax.random.key(0))
    _, (obs, r, term, _trunc) = rollout(state0, jax.random.split(jax.random.key(1), 7))
    assert term.tolist() == [False, False, True, False, False, True, False]
    assert obs.tolist() == [1, 2, 0, 1, 2, 0, 1]
    assert r.tolist() == [1.0] * 7


def test_rgb_spaces_fold_frames_and_colours():
    env = AtariEnvLike(_FakeVectorEnv(frames=4, h=210, w=160, colours=3))
    assert env.observation_space().shape == (210, 160, 12)
    assert env.observation_space().dtype == jnp.uint8
    assert env.observation_space().frame_channels == 3


def test_rgb_reset_and_step_shapes():
    env = AtariEnvLike(_FakeVectorEnv(frames=4, h=210, w=160, colours=3))
    obs, state = env.reset(jax.random.key(0))
    assert obs.shape == (210, 160, 12) and obs.dtype == jnp.uint8
    obs2, _state2, _r, _te, _tr, _i = env.step(jax.random.key(1), state, jnp.int32(0))
    assert obs2.shape == (210, 160, 12)


def test_num_envs_gt_one_rejected():
    with pytest.raises(ValueError):
        AtariEnvLike(_FakeVectorEnv(num_envs=2))


@pytest.mark.parametrize("agent_name", ["random_buffered", "dqn"])
def test_boundary_transition_successor_is_the_fresh_frame(agent_name):
    """A transition at an episode boundary stores the new episode's first frame
    as its successor, and no transition is dead - the whole chain from ale's
    native NEXT_STEP through AtariEnv's in-place reset to the buffer.

    This inverts the old issue #1 expectation on purpose: under immediate
    autoreset the true pre-boundary frame is deliberately not observable, and a
    truncated transition is excluded from training instead."""
    cutoff = 4
    env = AtariEnv(AtariEnvLike(_FakeVectorEnv(period=cutoff)))
    if agent_name == "random_buffered":
        agent = RandomBufferAgent(
            RandomBufferConfig(TOTAL_TIMESTEPS=11, BUFFER_SIZE=16, BATCH_SIZE=2)
        )
    else:
        agent = DQNAgent(
            DQNConfig(
                TOTAL_TIMESTEPS=11,
                BUFFER_SIZE=16,
                BATCH_SIZE=2,
                # no training: pure buffer inspection
                LEARNING_STARTS=11,
                NETWORK_PRESET="nature_cnn",
            )
        )
    run = jax.jit(lambda key: interaction(key, agent, env, 11))
    _metrics, final_carry = jax.block_until_ready(run(jax.random.key(0)))

    bs = final_carry[1].buffer_state
    transitions = agent._buffer.stored_transitions(bs)
    term = np.asarray(transitions.termination).astype(bool)
    trunc = np.asarray(transitions.truncation).astype(bool)
    # every pixel of a fake frame carries the step counter, so one pixel
    # identifies the frame
    obs = np.asarray(transitions.obs)[:, 0, 0, -1]

    ends = np.flatnonzero(term)
    # exactly every cutoff-th step: no step is spent replaying the boundary
    assert list(ends) == [cutoff - 1, 2 * cutoff - 1]
    assert not trunc.any()  # this fake's ending is always a termination
    # the entry after a boundary is the new episode's first frame, not the
    # boundary frame carried forward
    np.testing.assert_array_equal(obs[ends + 1], [0, 0])
    # the boundary entry's own obs is the last frame the agent acted on
    np.testing.assert_array_equal(obs[ends], [cutoff - 1, cutoff - 1])


# --- DQN on image obs in jit (fake env; reliable, no ale-py) -----------------


class _FakeImageEnv:
    """A DISABLED-mode env with (84,84,4) uint8 obs, to exercise the
    ``nature_cnn`` DQN (uint8 replay) under ``jax.jit``. Wrapped in
    AutoresetImmediate by the test, exactly like the real image envs."""

    def __init__(
        self, h: int = 84, w: int = 84, c: int = 4, n: int = 6, period: int = 7
    ) -> None:
        self._shape, self._n, self._period = (h, w, c), n, period

    def observation_space(self, params=None):
        return _Box(self._shape, jnp.uint8)

    def action_space(self, params=None):
        return _Discrete(self._n)

    def reset(self, key, params=None):
        return jnp.zeros(self._shape, jnp.uint8), jnp.int32(0)

    def step(self, key, state, action, params=None):
        t = state + 1
        term = t % self._period == 0
        # true obs, incl. terminal
        obs = jnp.full(self._shape, (t % 256).astype(jnp.uint8), jnp.uint8)
        return obs, t, jnp.float32(1.0), term, jnp.asarray(False), {}


def test_dqn_nature_cnn_jit_smoke():
    env = AutoresetImmediate(_FakeImageEnv())
    agent = DQNAgent(
        DQNConfig(
            TOTAL_TIMESTEPS=120,
            BUFFER_SIZE=200,
            BATCH_SIZE=8,
            LEARNING_STARTS=10,
            TRAIN_FREQUENCY=2,
            TARGET_NETWORK_FREQUENCY=20,
            EPSILON_FRACTION=0.5,
            NETWORK_PRESET="nature_cnn",
        )
    )
    run = jax.jit(lambda key: interaction(key, agent, env, 120))
    metrics, final_carry = jax.block_until_ready(run(jax.random.key(0)))
    assert metrics["reward"].shape == (120,)
    assert set(metrics) == {"reward", "done"}
    assert int(np.asarray(metrics["done"]).sum()) >= 1
    assert np.isfinite(np.asarray(final_carry[1].q.out.weight)).all()


# --- real ale-py XLA tests (skip if not installed) --------------------------


def _ale_xla_available() -> bool:
    try:
        import ale_py._ale_py as c

        return hasattr(c, "VectorXLAReset")
    except ImportError:
        return False


ale_only = pytest.mark.skipif(
    not _ale_xla_available(), reason="ale-py XLA build not installed"
)


@ale_only
@pytest.mark.parametrize("grayscale", [True, False])
def test_atari_frame_stacks_restart_zero_padded_at_every_boundary(grayscale):
    """Frame replay relies on ale restarting each stack zero-padded after every
    episode end and otherwise rolling by one frame - through a lost life too,
    so no step is hidden there - so the buffer reproduces every observation."""
    cutoff = 60
    config = AtariConfig(
        GAME="breakout",
        GRAYSCALE=grayscale,
        FRAMESKIP=4,
        MAX_FRAMES_PER_EPISODE=cutoff * 4,
        ZERO_DISCOUNT_ON_LIFE_LOSS=True,
    )
    env = ENVIRONMENTS["atari"].build(config)
    channels = env.observation_space().frame_channels
    buffer = ReplayBuffer(capacity=256, batch_size=2, n_step=1, gamma=0.99)
    buffer_state = buffer.init(env.observation_space())
    step = jax.jit(env.step)
    env_state, obs = env.init(jax.random.key(0))
    assert not np.asarray(obs)[..., :-channels].any()

    logged, boundaries, life_losses = [], 0, 0
    for t in range(150):
        action = jnp.int32(t % env.action_space().n)
        env_state, reward, term, trunc, discount, next_obs = step(
            env_state, jax.random.key(t), action
        )
        buffer_state = buffer.add(
            buffer_state, obs, action, reward, term, trunc, discount
        )
        logged.append(np.asarray(obs))
        older = np.asarray(next_obs)[..., :-channels]
        if bool(term | trunc):
            boundaries += 1
            assert not older.any()
        else:
            life_losses += int(discount == 0)
            np.testing.assert_array_equal(older, np.asarray(obs)[..., channels:])
        obs = next_obs

    assert boundaries >= 2
    assert life_losses >= 2  # breakout loses lives quickly under these actions
    np.testing.assert_array_equal(
        np.asarray(buffer.stored_transitions(buffer_state).obs), np.stack(logged)
    )


@ale_only
def test_atari_real_immediate_autoreset_cutoff():
    """End-to-end against real ale: with the exact cutoff (no +1 fudge),
    truncation fires at exactly ``cutoff`` agent steps into every
    episode, and no step is spent replaying the boundary - AtariEnv consumes
    ale's dead step itself, so boundaries land exactly ``cutoff`` apart."""
    cutoff = 12
    config = AtariConfig(GAME="pong", FRAMESKIP=4, MAX_FRAMES_PER_EPISODE=cutoff * 4)
    env = ENVIRONMENTS["atari"].build(config)
    state, _obs = env.init(jax.random.key(0))

    rows = []
    for n in range(1, 2 * cutoff + 3):
        action = jnp.int32(n % 6)
        state, r, term, trunc, _discount, _obs = env.step(
            state, jax.random.key(n), action
        )
        rows.append((float(r), bool(term), bool(trunc)))

    trunc_steps = [i + 1 for i, (_r, t, tr) in enumerate(rows) if tr]
    assert trunc_steps[0] == cutoff
    assert not any(t for _r, t, _tr in rows)
    # no dead step between episodes
    assert trunc_steps[1] == trunc_steps[0] + cutoff


@ale_only
def test_atari_real_boundary_successor_is_a_fresh_frame():
    """Drive ``random_buffered`` over real ale and check the buffer around a
    truncation: the entry after a boundary is the new episode's first frame,
    and boundaries are exactly ``cutoff`` apart with nothing dead between."""
    cutoff = 20
    config = AtariConfig(GAME="pong", FRAMESKIP=4, MAX_FRAMES_PER_EPISODE=cutoff * 4)
    env = ENVIRONMENTS["atari"].build(config)
    agent = RandomBufferAgent(
        RandomBufferConfig(TOTAL_TIMESTEPS=45, BUFFER_SIZE=64, BATCH_SIZE=2)
    )
    run = jax.jit(lambda key: interaction(key, agent, env, 45))
    _metrics, final_carry = jax.block_until_ready(run(jax.random.key(0)))

    bs = final_carry[1].buffer_state
    transitions = agent._buffer.stored_transitions(bs)
    trunc = np.asarray(transitions.truncation).astype(bool)
    obs = np.asarray(transitions.obs)

    ends = np.flatnonzero(trunc)
    assert list(ends) == [cutoff - 1, 2 * cutoff - 1]
    for e in ends:
        # the entry after a boundary starts from a fresh reset frame
        assert not np.array_equal(obs[e], obs[e + 1])


@ale_only
def test_atari_env_real_jit_scan():
    config = AtariConfig(GAME="pong", MAX_FRAMES_PER_EPISODE=4000)
    env = ENVIRONMENTS["atari"].build(config)
    state, obs = env.init(jax.random.key(0))
    assert obs.shape == (84, 84, 4) and obs.dtype == jnp.uint8

    @jax.jit
    def rollout(state, keys):
        def one(st, k):
            st, reward, _tm, _tr, _discount, _o = env.step(st, k, jnp.int32(0))
            return st, reward

        return jax.lax.scan(one, state, keys)

    _, rewards = rollout(state, jax.random.split(jax.random.key(1), 40))
    jax.block_until_ready(rewards)
    assert rewards.shape == (40,)
