from dataclasses import dataclass

from pinball_jax import Pinball, PinballParams

from environments.autoreset import AutoresetImmediate


@dataclass(frozen=True)
class PinballConfig:
    SETTING: str = "easy"  # bundled config: box/empty/easy/medium/hard
    EPISODE_CUTOFF: int = 1000


def build(config: PinballConfig):
    env = Pinball(config.SETTING)
    env_params = PinballParams(max_steps_in_episode=config.EPISODE_CUTOFF)
    return AutoresetImmediate(env, env_params)
