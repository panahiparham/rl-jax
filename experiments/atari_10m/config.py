from pathlib import Path

from experiment.design import Component, Experiment

from agents.dqn import DQNConfig
from environments.atari import RevisitingALEConfig
from main import ExperimentConfig

_GAMES = ["battle_zone", "ms_pacman"]

_DQN_HYPERS = {
    "TOTAL_TIMESTEPS": 2_500_000,       # 10M frames at FRAMESKIP=4
    "LR": 6.25e-05,
    "ADAM_EPS": 1.5e-4,
    "BUFFER_SIZE": 100_000,
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

_ATARI = RevisitingALEConfig()

_REAL_ATARI = RevisitingALEConfig(
    LIMITED_ACTION_SPACE=False,
    NOOP_MAX=0,
)

EXPERIMENT = Experiment(
    name="atari_10m",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name="dqn_atari",
            config=ExperimentConfig(
                AGENT="dqn",
                ENV="atari",
                AGENT_HYPERS=DQNConfig(**_DQN_HYPERS, REWARD_CLIP=True),
                ENV_HYPERS=_ATARI,
            ),
            sweep={"ENV_HYPERS.GAME": _GAMES},
            seeds=[0, 1, 2],
            shard_size=1,
        ),
        Component(
            name="dqn_real_atari",
            config=ExperimentConfig(
                AGENT="dqn",
                ENV="atari",
                AGENT_HYPERS=DQNConfig(**_DQN_HYPERS, REWARD_CLIP=False),
                ENV_HYPERS=_REAL_ATARI,
            ),
            sweep={"ENV_HYPERS.GAME": _GAMES},
            seeds=[0, 1, 2],
            shard_size=1,
        ),
    ],
)
