from pathlib import Path

from experiment.design import Component, Experiment, SlurmResources

from agents.dqn import DQNConfig
from environments.atari import AtariConfig
from main import ExperimentConfig

_GAMES = [
    "asterix",
    "venture",
    "atlantis",
    "tetris",
    "frogger",
]

_DQN_HYPERS = {
    "TOTAL_TIMESTEPS": 50_000_000,      # 200M frames at FRAMESKIP=4
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
    name="atari_200m_deterministic",
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
    ],
    # Five packed runs (parallel_shards=5) share one GPU through CUDA MPS,
    # with a CPU each.
    # atari_50m_deterministic's full packed workers took 4h10-4h55, so 200M
    # frames need about 17-20h.
    slurm=SlurmResources(
        time="23:59:00", gpus=1, mps=True, cpus_per_task=5, account="aip-whitem"
    ),
)
