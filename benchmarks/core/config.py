"""
Define: the weekly benchmark suite - DQN, DDQN and a Random baseline at each
environment's tuned learning rate.

Mirrors experiments/tuned across every environment that experiment covers
except Pinball's medium setting, at 10 seeds rather than 100 - enough to catch
a regression without a 100-seed bill every week. The seeds are the first ten of
tuned's range(10, 110), so these numbers sit alongside that experiment's own
and never reuse the seeds experiments/tuning picked the learning rates on.

Decoupled from experiments/ on purpose: this is what the weekly Vulcan run
recomputes from scratch every time (see weekly.py and slurm.wipe()), so its
hypers must stay stable across weeks rather than drift with whatever
experiments/ happens to be exploring. Hence copied recipes rather than an
import of tuned's config.
"""

from __future__ import annotations

from pathlib import Path

from experiment.design import Component, Experiment
from report import METRIC_REWARD, Environment, Series

from agents.ddqn import DDQNConfig
from agents.dqn import DQNConfig
from agents.random import RandomConfig
from environments.catch import CatchConfig
from environments.classic_control import (
    AcrobotConfig,
    CartpoleConfig,
    MountainCarConfig,
)
from environments.pinball import PinballConfig
from main import ExperimentConfig

# Best LR per (env, agent), read off experiments/tuning/analysis.ipynb - the
# same table as experiments/tuned/config.py's _BEST_LR, minus pinball_medium.
_BEST_LR = {
    "cartpole": {"dqn": 0.0009765625, "ddqn": 0.0009765625},
    "acrobot": {"dqn": 0.0009765625, "ddqn": 0.000244140625},
    "mountaincar": {"dqn": 0.0009765625, "ddqn": 0.0009765625},
    "catch": {"dqn": 0.00390625, "ddqn": 0.00390625},
    "pinball_box": {"dqn": 0.00390625, "ddqn": 0.00390625},
    "pinball_easy": {"dqn": 0.00390625, "ddqn": 0.00390625},
    "pinball_empty": {"dqn": 0.00390625, "ddqn": 0.00390625},
}

_SEEDS = list(range(10, 20))


def _components(
    env_key: str,
    env: str,
    learner: dict,
    env_hypers,
) -> list[Component]:
    best_lr = _BEST_LR[env_key]
    return [
        Component(
            name=f"dqn_{env_key}",
            config=ExperimentConfig(
                AGENT="dqn", ENV=env,
                AGENT_HYPERS=DQNConfig(**learner, LR=best_lr["dqn"]),
                ENV_HYPERS=env_hypers,
            ),
            seeds=_SEEDS,
            shard_size=None,  # one shard of every seed - what a GPU wants
        ),
        Component(
            name=f"ddqn_{env_key}",
            config=ExperimentConfig(
                AGENT="ddqn", ENV=env,
                AGENT_HYPERS=DDQNConfig(**learner, LR=best_lr["ddqn"]),
                ENV_HYPERS=env_hypers,
            ),
            seeds=_SEEDS,
            shard_size=None,
        ),
        Component(
            name=f"random_{env_key}",
            config=ExperimentConfig(
                AGENT="random", ENV=env,
                AGENT_HYPERS=RandomConfig(TOTAL_TIMESTEPS=learner["TOTAL_TIMESTEPS"]),
                ENV_HYPERS=env_hypers,
            ),
            seeds=_SEEDS,
            shard_size=None,
        ),
    ]


# The classic_control recipe (experiments/tuning/config.py's per-env learners,
# identical across Cartpole, Acrobot and MountainCar apart from the LR above).
_CLASSIC_CONTROL_LEARNER = {
    "TOTAL_TIMESTEPS": 100_000,
    "BUFFER_SIZE": 10_000,
    "BATCH_SIZE": 64,
    "LEARNING_STARTS": 1_000,
    "TRAIN_FREQUENCY": 1,
    "TARGET_NETWORK_FREQUENCY": 128,
    "GAMMA": 0.99,
    "EPSILON_START": 0.1,
    "EPSILON_END": 0.1,
    "NETWORK_PRESET": "mlp",
}

# The catch-jax README's recommended DQN hypers (experiments/tuning/config.py's
# _CATCH_LEARNER).
_CATCH_LEARNER = {
    "TOTAL_TIMESTEPS": 50_000,
    "BUFFER_SIZE": 100_000,
    "BATCH_SIZE": 32,
    "LEARNING_STARTS": 1_000,
    "TRAIN_FREQUENCY": 4,
    "TARGET_NETWORK_FREQUENCY": 128,
    "GAMMA": 0.9,
    "EPSILON_START": 0.01,
    "EPSILON_END": 0.01,
    "HIDDEN_SIZE": 32,
    "NETWORK_PRESET": "mlp",
}
_CATCH_ENV_HYPERS = CatchConfig(
    ROWS=10, COLUMNS=5, SPAWN_PROBABILITY=0.1,
    EPISODE_CUTOFF=1_000_000_000,  # continuing task - never truncate within a run
)

# The pinball recipe (experiments/tuning/config.py's _PINBALL_LEARNER).
_PINBALL_LEARNER = {
    "TOTAL_TIMESTEPS": 100_000,
    "BUFFER_SIZE": 10_000,
    "BATCH_SIZE": 32,
    "LEARNING_STARTS": 1_000,
    "TARGET_NETWORK_FREQUENCY": 100,
    "EPSILON_START": 0.1,
    "EPSILON_END": 0.1,
    "HIDDEN_SIZE": 32,
    "GAMMA": 0.99,
    "TRAIN_FREQUENCY": 1,
}

PLOTS_DIR = Path(__file__).resolve().parent / "plots"

_SERIES = (("Random", "tab:red", "random"), ("DQN", "tab:blue", "dqn"),
           ("DDQN", "tab:green", "ddqn"))


def _environment(env_key: str, title: str, **kwargs) -> Environment:
    return Environment(
        key=env_key,
        title=title,
        series=tuple(
            Series(label=label, color=color, component=f"{agent}_{env_key}")
            for label, color, agent in _SERIES
        ),
        **kwargs,
    )


# Titles, y-ranges and the continuing-task metric, as experiments/tuned's
# analysis.ipynb sets them - minus pinball_medium, which this suite skips.
ENVIRONMENTS = [
    _environment("cartpole", "Cartpole", ylim=(0, 500)),
    _environment("acrobot", "Acrobot", ylim=(-500, 0)),
    _environment("mountaincar", "MountainCar", ylim=(-1000, 0)),
    _environment("catch", "Catch", metric=METRIC_REWARD),
    _environment("pinball_box", "Pinball (Box)", ylim=(-1000, 0)),
    _environment("pinball_easy", "Pinball (Easy)", ylim=(-1000, 0)),
    _environment("pinball_empty", "Pinball (Empty)", ylim=(-1000, 0)),
]

EXPERIMENT = Experiment(
    name="bench_core",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        *_components(
            "cartpole", "cartpole", _CLASSIC_CONTROL_LEARNER,
            CartpoleConfig(EPISODE_CUTOFF=500),
        ),
        *_components(
            "acrobot", "acrobot", _CLASSIC_CONTROL_LEARNER,
            AcrobotConfig(EPISODE_CUTOFF=500),
        ),
        *_components(
            "mountaincar", "mountaincar", _CLASSIC_CONTROL_LEARNER,
            MountainCarConfig(EPISODE_CUTOFF=1_000),
        ),
        *_components(
            "catch", "catch", _CATCH_LEARNER, _CATCH_ENV_HYPERS,
        ),
        *_components(
            "pinball_box", "pinball", _PINBALL_LEARNER,
            PinballConfig(SETTING="box", EPISODE_CUTOFF=1_000),
        ),
        *_components(
            "pinball_easy", "pinball", _PINBALL_LEARNER,
            PinballConfig(SETTING="easy", EPISODE_CUTOFF=1_000),
        ),
        *_components(
            "pinball_empty", "pinball", _PINBALL_LEARNER,
            PinballConfig(SETTING="empty", EPISODE_CUTOFF=1_000),
        ),
    ],
)
