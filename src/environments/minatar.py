import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces


class MinAtarEnv:
    def __init__(self, env: environment.Environment):
        self._env = env

    def observation_space(self, params: environment.EnvParams):
        shape = self._env.observation_space(params).shape
        return spaces.Box(0, 1, shape, dtype=jnp.dtype(jnp.bool_))

    def action_space(self, params: environment.EnvParams):
        return self._env.action_space(params)

    def reset(self, key: jax.Array, params: environment.EnvParams):
        obs, state = self._env.reset_env(key, params)
        return obs.astype(bool), state

    def step(
        self,
        key: jax.Array,
        state: environment.EnvState,
        action: jax.Array,
        params: environment.EnvParams,
    ):
        obs, state, reward, done, _info = self._env.step_env(
            key, state, action, params
        )
        return obs.astype(bool), state, reward, done, jnp.zeros_like(done), {}
