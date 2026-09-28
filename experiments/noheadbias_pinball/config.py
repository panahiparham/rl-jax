"""
Define: DQN vs DQN without a Q-value head bias on Pinball(Easy), with a
1,000-transition buffer.

Both agents use DQN's tuned Pinball(Easy) hypers from ``experiments/tuned``
except ``BUFFER_SIZE``, 30 seeds each. Seeds start at 10, as in
``experiments/tuned``, so none of the seeds used to pick the LR are reused.
"""

from __future__ import annotations

from pathlib import Path

from experiment.design import Component, Experiment

from agents.dqn import DQNConfig
from agents.dqn_noheadbias import DQNNoHeadBiasConfig
from environments.pinball import PinballConfig
from main import ExperimentConfig

_PINBALL_LEARNER = {
    "TOTAL_TIMESTEPS": 100_000,
    "LR": 0.00390625,
    "BUFFER_SIZE": 1_000,
    "BATCH_SIZE": 32,
    "LEARNING_STARTS": 1_000,
    "TARGET_NETWORK_FREQUENCY": 100,
    "EPSILON_START": 0.1,
    "EPSILON_END": 0.1,
    "HIDDEN_SIZE": 32,
    "GAMMA": 0.99,
    "TRAIN_FREQUENCY": 1,
}
_ENV_HYPERS = PinballConfig(SETTING="easy", EPISODE_CUTOFF=1_000)
_SEEDS = list(range(10, 40))

EXPERIMENT = Experiment(
    name="noheadbias_pinball",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name="dqn_easy",
            config=ExperimentConfig(
                AGENT="dqn",
                ENV="pinball",
                AGENT_HYPERS=DQNConfig(**_PINBALL_LEARNER),
                ENV_HYPERS=_ENV_HYPERS,
            ),
            seeds=_SEEDS,
            shard_size=10,
        ),
        Component(
            name="dqn_noheadbias_easy",
            config=ExperimentConfig(
                AGENT="dqn_noheadbias",
                ENV="pinball",
                AGENT_HYPERS=DQNNoHeadBiasConfig(**_PINBALL_LEARNER),
                ENV_HYPERS=_ENV_HYPERS,
            ),
            seeds=_SEEDS,
            shard_size=10,
        ),
    ],
)
