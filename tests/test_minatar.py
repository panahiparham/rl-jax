"""Tests for the MinAtar environment, driven through the registry with the real
gymnax games."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from environments import ENVIRONMENTS
from environments.minatar import MinAtarConfig

CHANNELS = {"asterix": 4, "breakout": 4, "freeway": 7, "space_invaders": 6}
MINIMAL_ACTIONS = {"asterix": 5, "breakout": 3, "freeway": 3, "space_invaders": 4}
GAMES = list(CHANNELS)


def build(**fields):
    return ENVIRONMENTS["minatar"].build(MinAtarConfig(**fields))


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
