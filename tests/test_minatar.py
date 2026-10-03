"""Tests for the MinAtar environment, driven through the registry with the real
gymnax games."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnax.environments.minatar.breakout import MinBreakout

from agents.random import RandomConfig
from environments import ENVIRONMENTS
from environments.minatar import MinAtarConfig
from main import ExperimentConfig, process_shard

CHANNELS = {"asterix": 4, "breakout": 4, "freeway": 7, "space_invaders": 6}
MINIMAL_ACTIONS = {"asterix": 5, "breakout": 3, "freeway": 3, "space_invaders": 4}
GAMES = list(CHANNELS)


def build(**fields):
    return ENVIRONMENTS["minatar"].build(MinAtarConfig(**fields))


def rollout(env, policy, steps):
    state, _obs = env.init(jax.random.key(0))

    def step(state, key):
        state, _r, term, trunc, _discount, _obs = env.step(state, key, policy(state))
        return state, (term, trunc)

    keys = jax.random.split(jax.random.key(1), steps)
    _state, (terminated, truncated) = jax.lax.scan(step, state, keys)
    return np.asarray(terminated), np.asarray(truncated)


def noop(_state):
    return jnp.int32(0)


def track_ball(state):
    game = state.game
    return jnp.where(
        game.ball_x < game.pos, 1, jnp.where(game.ball_x > game.pos, 2, 0)
    )


class TestSpaces:
    @pytest.mark.parametrize("game", GAMES)
    def test_observation_is_a_bool_grid_with_the_games_channels(self, game):
        """The observation space and the observations agree: a 10x10 grid of
        bools with one channel per game object."""
        env = build(GAME=game)
        space = env.observation_space()
        _state, obs = env.init(jax.random.key(0))

        assert space.shape == (10, 10, CHANNELS[game])
        assert jnp.dtype(space.dtype) == jnp.bool_
        assert obs.shape == space.shape
        assert obs.dtype == jnp.bool_

    @pytest.mark.parametrize("game", GAMES)
    def test_minimal_action_set_is_the_default(self, game):
        """Each game exposes only the actions it uses unless asked otherwise."""
        assert build(GAME=game).action_space().n == MINIMAL_ACTIONS[game]

    @pytest.mark.parametrize("game", GAMES)
    def test_full_action_set_has_six_actions(self, game):
        """Turning the minimal set off exposes all six MinAtar actions."""
        env = build(GAME=game, USE_MINIMAL_ACTION_SET=False)
        assert env.action_space().n == 6


class TestConfig:
    def test_unknown_game_is_rejected(self):
        """Seaquest is not supported, so asking for it fails with the list of
        supported games."""
        with pytest.raises(ValueError, match="seaquest.*asterix"):
            build(GAME="seaquest")

    def test_registered_as_vmappable(self):
        """The games are pure jax, so runs can be vmapped over seeds."""
        assert ENVIRONMENTS["minatar"].vmappable


class TestTermination:
    def test_freeway_ends_as_a_termination_at_its_own_time_limit(self):
        """Freeway's 2500 steps are part of the game, as in the original, so the
        episode terminates there and is never reported as truncated."""
        terminated, truncated = rollout(build(GAME="freeway"), noop, 2600)

        assert np.flatnonzero(terminated).tolist() == [2499]
        assert not truncated.any()

    def test_a_game_without_a_limit_runs_past_gymnaxs_default_cap(self):
        """Without sticky actions a paddle that tracks the ball keeps Breakout
        alive for 1500 steps, so nothing ends the episode at gymnax's default
        limit of 1000."""
        env = build(GAME="breakout", STICKY_ACTION_PROB=0.0)
        terminated, truncated = rollout(env, track_ball, 1500)

        assert not terminated.any()
        assert not truncated.any()


class TestInteraction:
    @pytest.mark.parametrize("game", GAMES)
    def test_random_agent_runs_vmapped_under_jit(self, game):
        """A random agent steps every game through the full interaction loop,
        vmapped over seeds."""
        config = ExperimentConfig(
            AGENT="random",
            ENV="minatar",
            AGENT_HYPERS=RandomConfig(TOTAL_TIMESTEPS=50),
            ENV_HYPERS=MinAtarConfig(GAME=game),
        )
        runs = process_shard([config] * 3, [0, 1, 2])

        assert [run["reward"].shape for run in runs] == [(50,)] * 3
        assert [run["done"].shape for run in runs] == [(50,)] * 3


class TestStickyActions:
    LEFT, RIGHT = 1, 2

    def test_zero_probability_leaves_the_game_unchanged(self):
        """Without sticky actions the observations match raw gymnax step for
        step, so the wrapper adds nothing else."""
        env = build(GAME="breakout", STICKY_ACTION_PROB=0.0)
        raw = MinBreakout()
        params = raw.default_params
        state, obs = env.init(jax.random.key(0))
        raw_obs, raw_state = raw.reset_env(jax.random.key(0), params)
        assert np.array_equal(obs, raw_obs)

        for n, action in enumerate([self.LEFT, self.LEFT, self.RIGHT, 0, 0]):
            key = jax.random.key(n)
            state, _r, _term, _trunc, _discount, obs = env.step(state, key, action)
            raw_obs, raw_state, *_ = raw.step_env(key, raw_state, action, params)
            assert np.array_equal(obs, raw_obs)

    def test_certain_stickiness_repeats_the_previous_action(self):
        """At probability one the chosen action is ignored and the previous one
        runs instead."""
        env = build(GAME="breakout", STICKY_ACTION_PROB=1.0)
        state, _obs = env.init(jax.random.key(0))
        start = int(state.game.pos)

        state = state._replace(last_action=jnp.int32(self.LEFT))
        state, *_ = env.step(state, jax.random.key(1), self.RIGHT)

        assert int(state.game.pos) == start - 1
        assert int(state.last_action) == self.LEFT

    def test_fresh_episode_starts_with_no_previous_action(self):
        """Nothing is repeated on the first step: the previous action is the
        no-op, so a certain repeat leaves the paddle where it started."""
        env = build(GAME="breakout", STICKY_ACTION_PROB=1.0)
        state, _obs = env.init(jax.random.key(0))
        start = int(state.game.pos)

        state, *_ = env.step(state, jax.random.key(1), self.LEFT)

        assert int(state.game.pos) == start


class TestEpisodeCutoff:
    def test_cutoff_ends_episodes_as_truncations(self):
        """A cutoff is a time limit, not part of the game, so every 10th step
        is a truncation and nothing is a termination."""
        env = build(GAME="freeway", EPISODE_CUTOFF=10)
        terminated, truncated = rollout(env, noop, 25)

        assert np.flatnonzero(truncated).tolist() == [9, 19]
        assert not terminated.any()

    def test_cutoff_applies_to_games_without_a_limit(self):
        """A game that never ends on its own is still cut at the cutoff."""
        env = build(GAME="breakout", STICKY_ACTION_PROB=0.0, EPISODE_CUTOFF=100)
        terminated, truncated = rollout(env, track_ball, 150)

        assert np.flatnonzero(truncated).tolist() == [99]
        assert not terminated.any()

    def test_game_over_before_the_cutoff_is_a_termination(self):
        """Losing the game ends the episode as a termination even when a cutoff
        is set."""
        env = build(GAME="breakout", EPISODE_CUTOFF=1000)
        terminated, truncated = rollout(env, noop, 100)

        assert terminated.any()
        assert not truncated.any()

    def test_cutoff_above_freeways_limit_leaves_its_termination(self):
        """Freeway's own limit comes first, so it still ends as a termination at
        2500 steps."""
        env = build(GAME="freeway", EPISODE_CUTOFF=3000)
        terminated, truncated = rollout(env, noop, 2600)

        assert np.flatnonzero(terminated).tolist() == [2499]
        assert not truncated.any()
