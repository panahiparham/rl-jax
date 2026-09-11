from dataclasses import dataclass
from typing import Any

import gymnax
import jax
import jax.numpy as jnp

from environments.autoreset import AutoresetImmediate


class GymnaxEnv:
    def __init__(self, env: Any, env_params: Any):
        self._env = env
        self.env_params = env_params

    @classmethod
    def make(cls, name: str, episode_cutoff: int):
        env, env_params = gymnax.make(name)
        env_params = env_params.replace(max_steps_in_episode=int(episode_cutoff))
        return cls(env, env_params), env_params

    def observation_space(self, params: Any | None = None):
        return self._env.observation_space(
            self.env_params if params is None else params
        )

    def action_space(self, params: Any | None = None):
        return self._env.action_space(self.env_params if params is None else params)

    def reset(self, key: jax.Array, params: Any | None = None):
        return self._env.reset(key, self.env_params if params is None else params)

    def step(
        self,
        key: jax.Array,
        state: Any,
        action: jax.Array,
        params: Any | None = None,
    ):
        params = self.env_params if params is None else params
        obs, next_state, reward, done, _info = self._env.step_env(
            key, state, action, params
        )
        truncated = next_state.time >= params.max_steps_in_episode
        terminated = jnp.logical_and(done, jnp.logical_not(truncated))
        return obs, next_state, reward, terminated, truncated, {}


@dataclass(frozen=True)
class MountainCarConfig:
    EPISODE_CUTOFF: int = 1000


@dataclass(frozen=True)
class CartpoleConfig:
    EPISODE_CUTOFF: int = 500


@dataclass(frozen=True)
class AcrobotConfig:
    EPISODE_CUTOFF: int = 500


def build_mountaincar(config: MountainCarConfig):
    env, env_params = GymnaxEnv.make("MountainCar-v0", config.EPISODE_CUTOFF)
    return AutoresetImmediate(env, env_params)


def build_cartpole(config: CartpoleConfig):
    env, env_params = GymnaxEnv.make("CartPole-v1", config.EPISODE_CUTOFF)
    return AutoresetImmediate(env, env_params)


def build_acrobot(config: AcrobotConfig):
    env, env_params = GymnaxEnv.make("Acrobot-v1", config.EPISODE_CUTOFF)
    return AutoresetImmediate(env, env_params)
