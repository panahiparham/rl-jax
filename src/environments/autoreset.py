from typing import Any

import jax
import jax.numpy as jnp


class AutoresetImmediate:
    def __init__(self, env: Any, params: object | None = None):
        self._env = env
        self._params = params

    def observation_space(self):
        return self._env.observation_space(self._params)

    def action_space(self):
        return self._env.action_space(self._params)

    def init(self, key: jax.Array):
        obs, state = self._env.reset(key, self._params)
        return state, obs

    def step(self, state: Any, key: jax.Array, action: jax.Array):
        step_key, reset_key = jax.random.split(key)
        obs, stepped, reward, termination, truncation, _info = self._env.step(
            step_key, state, action, self._params
        )
        obs_reset, state_reset = self._env.reset(reset_key, self._params)

        done = termination | truncation
        next_obs = jnp.where(done, obs_reset, obs)
        next_state = jax.tree.map(
            lambda a, b: jnp.where(done, a, b), state_reset, stepped
        )
        return next_state, reward, termination, truncation, next_obs
