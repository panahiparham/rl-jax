from pathlib import Path

from experiment.design import Component, Experiment, SlurmResources

from agents.agent0 import Agent0Config
from agents.dqn import DQNConfig
from environments.atari import AtariConfig
from main import ExperimentConfig

_GAMES = [
    "pong",
    "breakout",
    "seaquest",
    "centipede",
    "ms_pacman",
    "beam_rider",
    "space_invaders",
    "battle_zone",
    "double_dunk",
    "name_this_game",
    "phoenix",
    "qbert",
]

_DQN_HYPERS = {
    "TOTAL_TIMESTEPS": 12_500_000,      # 50M frames at FRAMESKIP=4
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

# Agent0 has no target network and uses the LN variant of the Nature CNN.
_AGENT0_HYPERS = {
    **{k: v for k, v in _DQN_HYPERS.items() if k != "TARGET_NETWORK_FREQUENCY"},
    "NETWORK_PRESET": "nature_cnn_ln",
}

EXPERIMENT = Experiment(
    name="atari_50m_deterministic",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name="dqn_atari",
            config=ExperimentConfig(
                AGENT="dqn",
                ENV="atari",
                AGENT_HYPERS=DQNConfig(**_DQN_HYPERS, REWARD_CLIP=True),
                ENV_HYPERS=AtariConfig(),
            ),
            sweep={"ENV_HYPERS.GAME": _GAMES},
            seeds=[0],
            shard_size=1,
            parallel_shards=5,  # five runs fit in one L40S
        ),
        Component(
            name="agent0_atari",
            config=ExperimentConfig(
                AGENT="agent0",
                ENV="atari",
                AGENT_HYPERS=Agent0Config(**_AGENT0_HYPERS, REWARD_CLIP=True),
                ENV_HYPERS=AtariConfig(),
            ),
            sweep={"ENV_HYPERS.GAME": _GAMES},
            seeds=[0],
            shard_size=1,
            parallel_shards=5,
        ),
    ],
    # Five packed runs (parallel_shards=5) share one GPU through CUDA MPS,
    # with a CPU each.
    # atari_10m_deterministic's packed runs took up to 1h20, so 50M frames
    # need about 7h.
    slurm=SlurmResources(time="11:59:00", gpus=1, mps=True, cpus_per_task=5),
)
