from dataclasses import dataclass
from typing import NamedTuple

import gymnax
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from environments.autoreset import AutoresetImmediate

_GAMES = {
    "asterix": "Asterix-MinAtar",
    "breakout": "Breakout-MinAtar",
    "freeway": "Freeway-MinAtar",
    "space_invaders": "SpaceInvaders-MinAtar",
}
# Only Freeway has a time limit of its own; gymnax would cap the others at 1000.
_GAME_TIME_LIMITS = {"freeway": 2500}
_UNBOUNDED = 2**31 - 1


@dataclass(frozen=True)
class MinAtarConfig:
    GAME: str = "breakout"
    USE_MINIMAL_ACTION_SET: bool = True
    STICKY_ACTION_PROB: float = 0.1


class MinAtarState(NamedTuple):
    game: environment.EnvState
    last_action: jax.Array


class MinAtarEnv:
    def __init__(self, env: environment.Environment, sticky_action_prob: float):
        self._env = env
        self._sticky_action_prob = sticky_action_prob

    def observation_space(self, params: environment.EnvParams):
        shape = self._env.observation_space(params).shape
        return spaces.Box(0, 1, shape, dtype=jnp.dtype(jnp.bool_))

    def action_space(self, params: environment.EnvParams):
        return self._env.action_space(params)

    def reset(self, key: jax.Array, params: environment.EnvParams):
        obs, game = self._env.reset_env(key, params)
        return obs.astype(bool), MinAtarState(game, jnp.int32(0))

    def step(
        self,
        key: jax.Array,
        state: MinAtarState,
        action: jax.Array,
        params: environment.EnvParams,
    ):
        sticky_key, game_key = jax.random.split(key)
        repeat = jax.random.uniform(sticky_key) < self._sticky_action_prob
        action = jnp.where(repeat, state.last_action, action)

        obs, game, reward, done, _info = self._env.step_env(
            game_key, state.game, action, params
        )
        state = MinAtarState(game, action)
        return obs.astype(bool), state, reward, done, jnp.zeros_like(done), {}


def build(config: MinAtarConfig):
    if config.GAME not in _GAMES:
        raise ValueError(
            f"unknown MinAtar game {config.GAME!r}; supported: {sorted(_GAMES)}"
        )
    env, params = gymnax.make(
        _GAMES[config.GAME], use_minimal_action_set=config.USE_MINIMAL_ACTION_SET
    )
    params = params.replace(
        max_steps_in_episode=_GAME_TIME_LIMITS.get(config.GAME, _UNBOUNDED)
    )
    env = MinAtarEnv(env, config.STICKY_ACTION_PROB)
    return AutoresetImmediate(env, params)
