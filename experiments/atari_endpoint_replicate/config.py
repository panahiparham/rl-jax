from pathlib import Path

from experiment.design import Component, Experiment, SlurmResources

from agents.ddqn import DDQNConfig
from environments.atari import EPRAtariConfig
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

_DDQN_HYPERS = {
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

EXPERIMENT = Experiment(
    name="atari_endpoint_replicate",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name="ddqn_atari",
            config=ExperimentConfig(
                AGENT="ddqn",
                ENV="atari",
                AGENT_HYPERS=DDQNConfig(**_DDQN_HYPERS, REWARD_CLIP=True),
                ENV_HYPERS=EPRAtariConfig(),
            ),
            sweep={"ENV_HYPERS.GAME": _GAMES},
            seeds=[0],
            shard_size=1,
            parallel_shards=5,  # five runs fit in one L40S
        ),
    ],
    # Five packed runs (parallel_shards=5) share one GPU through CUDA MPS,
    # with a CPU each.
    slurm=SlurmResources(time="11:59:00", gpus=1, mps=True, cpus_per_task=5),
)
