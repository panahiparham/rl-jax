from pathlib import Path

from experiment.design import Component, Experiment, SlurmResources

from agents.ddqn import DDQNConfig
from agents.dqn import DQNConfig
from agents.endpoint import EndpointConfig
from environments.atari import AtariConfig, EPRAtariConfig
from main import ExperimentConfig

_GAMES = ["pong", "breakout", "seaquest", "ms_pacman"]

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

EXPERIMENT = Experiment(
    name="atari_50m_endpoint",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        # Reproduces atari_50m_deterministic's dqn_atari on these games.
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
            parallel_shards=4,
        ),
        Component(
            name="ddqn_atari",
            config=ExperimentConfig(
                AGENT="ddqn",
                ENV="atari",
                AGENT_HYPERS=DDQNConfig(**_DQN_HYPERS, REWARD_CLIP=True),
                ENV_HYPERS=EPRAtariConfig(),
            ),
            sweep={"ENV_HYPERS.GAME": _GAMES},
            seeds=[0],
            shard_size=1,
            parallel_shards=4,
        ),
        # Defaults mirror the original endpoint_medium run: 10k recency,
        # 90k 10-step long-term transitions, expectile 0.7.
        Component(
            name="endpoint_atari",
            config=ExperimentConfig(
                AGENT="endpoint",
                ENV="atari",
                AGENT_HYPERS=EndpointConfig(),
                ENV_HYPERS=EPRAtariConfig(),
            ),
            sweep={"ENV_HYPERS.GAME": _GAMES},
            seeds=[0],
            shard_size=1,
            parallel_shards=4,
        ),
    ],
    # Four packed runs (parallel_shards=4) share one GPU through CUDA MPS,
    # with a CPU each. The endpoint agent's runtime is unknown; 12h leaves
    # room above the expected ~5.5h.
    slurm=SlurmResources(time="12:00:00", gpus=1, mps=True, cpus_per_task=4),
)
