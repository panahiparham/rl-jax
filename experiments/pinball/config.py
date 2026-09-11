"""
Define: DQN, DDQN, and Random Agent on Pinball(Easy) Environment.
Default Hypers, 30 seeds.
"""

from __future__ import annotations

from pathlib import Path

from experiment.design import Component, Experiment

from agents.ddqn import DDQNConfig
from agents.dqn import DQNConfig
from agents.random import RandomConfig
from environments.pinball import PinballConfig
from main import ExperimentConfig

# Shared learner hypers so dqn_easy / ddqn_easy are a fair comparison.
_PINBALL_LEARNER = {
    "TOTAL_TIMESTEPS": 100_000,
    "LR": 0.002,
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

EXPERIMENT = Experiment(
    name="pinball",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name="random_easy",
            config=ExperimentConfig(
                AGENT="random",
                ENV="pinball",
                AGENT_HYPERS=RandomConfig(TOTAL_TIMESTEPS=100_000),
                ENV_HYPERS=PinballConfig(SETTING="easy", EPISODE_CUTOFF=1_000),
            ),
            seeds=list(range(30)),
            shard_size=None,  # random is cheap: one shard, all 30 seeds at once
        ),
        Component(
            name="dqn_easy",
            config=ExperimentConfig(
                AGENT="dqn",
                ENV="pinball",
                AGENT_HYPERS=DQNConfig(**_PINBALL_LEARNER),
                ENV_HYPERS=PinballConfig(SETTING="easy", EPISODE_CUTOFF=1_000),
            ),
            seeds=list(range(30)),
            shard_size=5,  # 30 seeds / 5 = 6 shards
        ),
        Component(
            name="ddqn_easy",
            config=ExperimentConfig(
                AGENT="ddqn",
                ENV="pinball",
                AGENT_HYPERS=DDQNConfig(**_PINBALL_LEARNER),
                ENV_HYPERS=PinballConfig(SETTING="easy", EPISODE_CUTOFF=1_000),
            ),
            seeds=list(range(30)),
            shard_size=5,
        ),
    ],
)
