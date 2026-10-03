import jax
from gymnax.environments import environment, spaces


class MinAtarEnv:
    def __init__(self, env: environment.Environment):
        self._env = env

    def observation_space(self, params: environment.EnvParams):
        shape = self._env.observation_space(params).shape
        return spaces.Box(0, 1, shape, dtype=bool)

    def action_space(self, params: environment.EnvParams):
        return self._env.action_space(params)

    def reset(self, key: jax.Array, params: environment.EnvParams):
        obs, state = self._env.reset_env(key, params)
        return obs.astype(bool), state
