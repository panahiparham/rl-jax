# Observations are (210, 160, 12): native-resolution frames with the 4-frame
# stack and RGB channels folded into a single channel-last axis.

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from environments.atari import AtariEnv, AtariEnvLike, require_ale_xla

SCREEN_HEIGHT = 210
SCREEN_WIDTH = 160


@dataclass(frozen=True)
class RealAtariConfig:
    GAME: str = "pong"
    FRAMESKIP: int = 4
    STICKY_ACTIONS: float = 0.25
    EPISODE_CUTOFF: int = 27000


class RealAtariEnvLike(AtariEnvLike):
    def __init__(self, vector_env):
        if vector_env.num_envs != 1:
            raise ValueError(
                "RealAtariEnvLike supports a single environment, got "
                f"num_envs={vector_env.num_envs}."
            )
        self._num_envs = 1
        self._init_handle, self._reset_fn, self._step_fn = vector_env.xla()
        frames, height, width, colours = vector_env.single_observation_space.shape
        self._obs_shape = (height, width, frames * colours)
        self._n_actions = int(vector_env.single_action_space.n)

    def _to_hwc(self, obs: jax.Array):
        stacked = jnp.transpose(obs[0], (1, 2, 0, 3))
        return stacked.reshape(self._obs_shape)


def build(config: RealAtariConfig):
    import ale_py  # lazy: the base project installs without the atari extra

    require_ale_xla(config.GAME)

    kwargs = {
        "game": config.GAME,
        "num_envs": 1,
        "frameskip": int(config.FRAMESKIP),
        "repeat_action_probability": float(config.STICKY_ACTIONS),
        "img_height": SCREEN_HEIGHT,
        "img_width": SCREEN_WIDTH,
        "grayscale": False,
        "reward_clipping": False,
        "full_action_space": True,
        "noop_max": 0,
    }
    if config.EPISODE_CUTOFF and config.EPISODE_CUTOFF > 0:
        kwargs["max_num_frames_per_episode"] = int(config.EPISODE_CUTOFF) * int(
            config.FRAMESKIP
        )

    return AtariEnv(RealAtariEnvLike(ale_py.AtariVectorEnv(**kwargs)))
