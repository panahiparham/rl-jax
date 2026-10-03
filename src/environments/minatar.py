from dataclasses import asdict, dataclass
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
_NON_PARAM_FIELDS = {
    "GAME",
    "USE_MINIMAL_ACTION_SET",
    "STICKY_ACTION_PROB",
    "EPISODE_CUTOFF",
}


@dataclass(frozen=True)
class MinAtarConfig:
    GAME: str = "breakout"
    USE_MINIMAL_ACTION_SET: bool = True
    STICKY_ACTION_PROB: float = 0.1
    EPISODE_CUTOFF: int | None = None
    # gymnax EnvParams of the games; None keeps the game's default
    RAMPING: bool | None = None
    RAMP_INTERVAL: int | None = None
    INIT_SPAWN_SPEED: int | None = None
    INIT_MOVE_INTERVAL: int | None = None
    SHOT_COOL_DOWN: int | None = None
    ENEMY_MOVE_INTERVAL: int | None = None
    ENEMY_SHOT_INTERVAL: int | None = None
    PLAYER_SPEED: int | None = None


class MinAtarState(NamedTuple):
    game: environment.EnvState
    last_action: jax.Array


class MinAtarEnv:
    def __init__(
        self,
        env: environment.Environment,
        sticky_action_prob: float,
        truncate_at: int,
    ):
        self._env = env
        self._sticky_action_prob = sticky_action_prob
        self._truncate_at = truncate_at

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
        truncated = game.time >= self._truncate_at
        terminated = done & ~truncated
        return obs.astype(bool), state, reward, terminated, truncated, {}


def build(config: MinAtarConfig):
    if config.GAME not in _GAMES:
        raise ValueError(
            f"unknown MinAtar game {config.GAME!r}; supported: {sorted(_GAMES)}"
        )
    env, params = gymnax.make(
        _GAMES[config.GAME], use_minimal_action_set=config.USE_MINIMAL_ACTION_SET
    )
    game_limit = _GAME_TIME_LIMITS.get(config.GAME, _UNBOUNDED)
    cutoff = config.EPISODE_CUTOFF
    truncate_at = cutoff if cutoff is not None and cutoff < game_limit else _UNBOUNDED
    overrides = {
        name.lower(): value
        for name, value in asdict(config).items()
        if name not in _NON_PARAM_FIELDS and value is not None
    }
    params = params.replace(
        max_steps_in_episode=min(truncate_at, game_limit), **overrides
    )
    env = MinAtarEnv(env, config.STICKY_ACTION_PROB, truncate_at)
    return AutoresetImmediate(env, params)
