from pathlib import Path

from experiment.design import Component, Experiment

from agents.dqn import DQNConfig
from environments.atari import RevisitingALEConfig
from main import ExperimentConfig

# A short run with the full-size replay buffer, which is allocated up front, so
# it costs as much GPU memory as a full-length run.
_DQN_HYPERS = {
    "TOTAL_TIMESTEPS": 100_000,
    "LR": 6.25e-05,
    "ADAM_EPS": 1.5e-4,
    "BUFFER_SIZE": 1_000_000,
    "BATCH_SIZE": 32,
    "LEARNING_STARTS": 20_000,
    "TRAIN_FREQUENCY": 4,
    "TARGET_NETWORK_FREQUENCY": 8_000,
    "GAMMA": 0.99,
    "EPSILON_START": 1.0,
    "EPSILON_END": 0.01,
    "EPSILON_DECAY_STEPS": 250_000,
    "NETWORK_PRESET": "nature_cnn",
}

EXPERIMENT = Experiment(
    name="atari_100k",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name="dqn_pong",
            config=ExperimentConfig(
                AGENT="dqn",
                ENV="atari",
                AGENT_HYPERS=DQNConfig(**_DQN_HYPERS, REWARD_CLIP=True),
                ENV_HYPERS=RevisitingALEConfig(GAME="pong"),
            ),
            seeds=list(range(10)),
            shard_size=1,
            parallel_shards=5,  # five runs fit in one L40S
        ),
    ],
)
