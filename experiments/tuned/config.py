"""
Define: DQN, DDQN, and Random Agent at each environment's tuned learning rate,
across every environment covered by ``experiments/tuning`` - Acrobot, Cartpole,
Catch, MountainCar, and Pinball's box/easy/empty/medium settings.

Each (agent, env) pair runs at the best LR found by ``experiments/tuning``'s
sensitivity sweep (see ``_BEST_LR`` below), 100 seeds each, against a 100-seed
Random baseline. Seeds are offset by 10 - ``range(10, 110)`` - so none of the
seeds used to pick that LR are reused here.
"""

from __future__ import annotations

from pathlib import Path

from experiment.design import Component, Experiment

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

# Best LR per (env, agent), read off experiments/tuning/analysis.ipynb.
_BEST_LR = {
    "cartpole": {"dqn": 0.0009765625, "ddqn": 0.0009765625},
    "acrobot": {"dqn": 0.0009765625, "ddqn": 0.000244140625},
    "mountaincar": {"dqn": 0.0009765625, "ddqn": 0.0009765625},
    "catch": {"dqn": 0.00390625, "ddqn": 0.00390625},
    "pinball_box": {"dqn": 0.00390625, "ddqn": 0.00390625},
    "pinball_easy": {"dqn": 0.00390625, "ddqn": 0.00390625},
    "pinball_empty": {"dqn": 0.00390625, "ddqn": 0.00390625},
    "pinball_medium": {"dqn": 0.00390625, "ddqn": 0.00390625},
}

# 100 seeds, offset by 10 so tuning's seeds (0-9) are never reused.
_SEEDS = list(range(10, 110))


def _components(
    env_key: str,
    env: str,
    learner: dict,
    env_hypers,
) -> list[Component]:
    """The dqn/ddqn/random trio for one environment, at its tuned LR.

    Args:
        env_key: Suffix identifying this environment (and setting, for Pinball) in
            component names, e.g. ``"cartpole"`` or ``"pinball_box"``.
        env: The registry ``ENV`` name, e.g. ``"cartpole"`` or ``"pinball"``.
        learner: DQN/DDQN hypers shared by both agents, minus ``LR``.
        env_hypers: The environment's own hypers (e.g. ``CartpoleConfig``).

    Returns:
        Three named components: ``dqn_<env_key>``, ``ddqn_<env_key>``,
        ``random_<env_key>``.
    """
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
            shard_size=10,
        ),
        Component(
            name=f"ddqn_{env_key}",
            config=ExperimentConfig(
                AGENT="ddqn", ENV=env,
                AGENT_HYPERS=DDQNConfig(**learner, LR=best_lr["ddqn"]),
                ENV_HYPERS=env_hypers,
            ),
            seeds=_SEEDS,
            shard_size=10,
        ),
        Component(
            name=f"random_{env_key}",
            config=ExperimentConfig(
                AGENT="random", ENV=env,
                AGENT_HYPERS=RandomConfig(TOTAL_TIMESTEPS=learner["TOTAL_TIMESTEPS"]),
                ENV_HYPERS=env_hypers,
            ),
            seeds=_SEEDS,
            shard_size=None,  # random is cheap: one shard, all 100 seeds at once
        ),
    ]


# Every other hyper follows the classic_control Cartpole benchmark recipe (see
# experiments/tuning/config.py's _CARTPOLE_LEARNER).
_CARTPOLE_LEARNER = {
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

# Every other hyper follows the classic_control Acrobot benchmark recipe (see
# experiments/tuning/config.py's _ACROBOT_LEARNER).
_ACROBOT_LEARNER = {
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

# Every other hyper follows the classic_control MountainCar benchmark recipe (see
# experiments/tuning/config.py's _MOUNTAINCAR_LEARNER).
_MOUNTAINCAR_LEARNER = {
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

# Every other hyper follows the catch-jax README's recommended DQN hypers (see
# experiments/tuning/config.py's _CATCH_LEARNER).
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
    EPISODE_CUTOFF=1_000_000_000,  # continuing task - never truncate within this run
)

# Every other hyper follows the pinball benchmark recipe (see
# experiments/tuning/config.py's _PINBALL_LEARNER).
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

# Medium needed more training time than the other Pinball settings (see
# experiments/tuning/config.py's _PINBALL_MEDIUM_LEARNER).
_PINBALL_MEDIUM_LEARNER = {**_PINBALL_LEARNER, "TOTAL_TIMESTEPS": 500_000}

EXPERIMENT = Experiment(
    name="tuned",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        *_components(
            "cartpole", "cartpole", _CARTPOLE_LEARNER,
            CartpoleConfig(EPISODE_CUTOFF=500),
        ),
        *_components(
            "acrobot", "acrobot", _ACROBOT_LEARNER,
            AcrobotConfig(EPISODE_CUTOFF=500),
        ),
        *_components(
            "mountaincar", "mountaincar", _MOUNTAINCAR_LEARNER,
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
        *_components(
            "pinball_medium", "pinball", _PINBALL_MEDIUM_LEARNER,
            PinballConfig(SETTING="medium", EPISODE_CUTOFF=1_000),
        ),
    ],
)
