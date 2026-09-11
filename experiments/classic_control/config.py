"""
Define: DQN on Classic Control environments (MountainCar, Cartpole, Acrobot).
Default Hypers, 100 seeds.
"""

from __future__ import annotations

from pathlib import Path

from experiment.design import Component, Experiment

from agents.dqn import DQNConfig
from environments.classic_control import (
    AcrobotConfig,
    CartpoleConfig,
    MountainCarConfig,
)
from main import ExperimentConfig


def _dqn_hypers(total_timesteps: int = 100_000) -> DQNConfig:
    return DQNConfig(
        TOTAL_TIMESTEPS=total_timesteps,
        LR=0.001,
        BUFFER_SIZE=10_000,
        BATCH_SIZE=64,
        LEARNING_STARTS=1_000,
        TRAIN_FREQUENCY=1,
        TARGET_NETWORK_FREQUENCY=128,
        GAMMA=0.99,
        EPSILON_START=0.1,
        EPSILON_END=0.1,
        NETWORK_PRESET="mlp",
    )


EXPERIMENT = Experiment(
    name="classic_control",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name="dqn_mountaincar",
            config=ExperimentConfig(
                AGENT="dqn",
                ENV="mountaincar",
                AGENT_HYPERS=_dqn_hypers(),
                ENV_HYPERS=MountainCarConfig(EPISODE_CUTOFF=1_000),
            ),
            seeds=list(range(100)),
            shard_size=None,  # one vmap of every seed - what a GPU wants
        ),
        Component(
            name="dqn_cartpole",
            config=ExperimentConfig(
                AGENT="dqn",
                ENV="cartpole",
                AGENT_HYPERS=_dqn_hypers(),
                ENV_HYPERS=CartpoleConfig(EPISODE_CUTOFF=500),
            ),
            seeds=list(range(100)),
            shard_size=None,
        ),
        Component(
            name="dqn_acrobot",
            config=ExperimentConfig(
                AGENT="dqn",
                ENV="acrobot",
                AGENT_HYPERS=_dqn_hypers(),
                ENV_HYPERS=AcrobotConfig(EPISODE_CUTOFF=500),
            ),
            seeds=list(range(100)),
            shard_size=None,
        ),
    ],
)
