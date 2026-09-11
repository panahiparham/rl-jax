from dataclasses import dataclass

from catch_jax import Catch, CatchParams

from environments.autoreset import AutoresetImmediate


@dataclass(frozen=True)
class CatchConfig:
    ROWS: int = 10
    COLUMNS: int = 5
    SPAWN_PROBABILITY: float = 0.1  # chance a new ball spawns each step
    EPISODE_CUTOFF: int = 1_000_000_000  # effectively unbounded


def build(config: CatchConfig):
    env = Catch(rows=config.ROWS, columns=config.COLUMNS)
    env_params = CatchParams(
        spawn_probability=config.SPAWN_PROBABILITY,
        max_steps_in_episode=config.EPISODE_CUTOFF,
    )
    return AutoresetImmediate(env, env_params)
