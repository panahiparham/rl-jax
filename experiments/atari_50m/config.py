from pathlib import Path

from experiment.design import Component, Experiment

from agents.agent0 import Agent0Config
from agents.dqn import DQNConfig
from environments.atari import AtariConfig
from main import ExperimentConfig

_GAMES = ["battle_zone", "pong", "breakout", "ms_pacman"]

# Shared by both components; REWARD_CLIP is the one hyper that varies per component.
_DQN_HYPERS = {
    "TOTAL_TIMESTEPS": 12_500_000,      # 50M frames at FRAMESKIP=4
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
    "EPSILON_FRACTION": 0.05, # Should be 0.02 to match Dopamine.
    "NETWORK_PRESET": "nature_cnn",
}

# Agent0 has no TARGET_NETWORK_FREQUENCY (no target network) and uses the
# LN variant of the Nature CNN; derive its hypers from the same source of
# truth minus that field, with NETWORK_PRESET overridden.
_AGENT0_HYPERS = {
    **{k: v for k, v in _DQN_HYPERS.items() if k != "TARGET_NETWORK_FREQUENCY"},
    "NETWORK_PRESET": "nature_cnn_ln",
}

_ATARI = AtariConfig()

_REAL_ATARI = AtariConfig(
    LIMITED_ACTION_SPACE=False,
    EPISODIC_LIFE=False,
    NOOP_MAX=0,
)

EXPERIMENT = Experiment(
    name="atari_50m",
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
            seeds=[0],
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
            seeds=[0],
            shard_size=1,
        ),
        Component(
            name="agent0_atari",
            config=ExperimentConfig(
                AGENT="agent0",
                ENV="atari",
                AGENT_HYPERS=Agent0Config(**_AGENT0_HYPERS, REWARD_CLIP=True),
                ENV_HYPERS=_ATARI,
            ),
            sweep={"ENV_HYPERS.GAME": _GAMES},
            seeds=[0],
            shard_size=1,
        ),
        Component(
            name="agent0_real_atari",
            config=ExperimentConfig(
                AGENT="agent0",
                ENV="atari",
                AGENT_HYPERS=Agent0Config(**_AGENT0_HYPERS, REWARD_CLIP=False),
                ENV_HYPERS=_REAL_ATARI,
            ),
            sweep={"ENV_HYPERS.GAME": _GAMES},
            seeds=[0],
            shard_size=1,
        ),
    ],
)
