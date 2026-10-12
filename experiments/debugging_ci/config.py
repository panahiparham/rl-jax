from pathlib import Path

from experiment.design import Component, Experiment, SlurmResources

from agents.agent0 import Agent0Config
from agents.dqn import MinAtarDQNConfig
from environments.atari import RealAtariConfigV1
from environments.minatar import MinAtarConfig
from main import ExperimentConfig

# Agent0's hypers from benchmarking_atari_50m_realv1. They must match exactly so
# its stored runs keep their run ids here.
_AGENT0_HYPERS = {
    "TOTAL_TIMESTEPS": 12_500_000,      # 50M frames at FRAMESKIP=4
    "LR": 6.25e-05,
    "ADAM_EPS": 1.5e-4,
    "BUFFER_SIZE": 1_000_000,
    "BATCH_SIZE": 32,
    "LEARNING_STARTS": 20_000,
    "TRAIN_FREQUENCY": 4,
    "GAMMA": 0.99,
    "EPSILON_START": 1.0,
    "EPSILON_END": 0.01,
    "EPSILON_DECAY_STEPS": 250_000,
    "NETWORK_PRESET": "nature_cnn_ln",
}

# The components whose confidence bands looked noisy in their parent
# experiments: Agent0 on Seaquest from benchmarking_atari_50m_realv1 and DQN on
# Asterix from minatar_deterministic.
EXPERIMENT = Experiment(
    name="debugging_ci",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name="agent0_seaquest",
            config=ExperimentConfig(
                AGENT="agent0",
                ENV="atari",
                AGENT_HYPERS=Agent0Config(**_AGENT0_HYPERS, REWARD_CLIP=False),
                ENV_HYPERS=RealAtariConfigV1(),
            ),
            sweep={"ENV_HYPERS.GAME": ["seaquest"]},
            seeds=list(range(10)),
            shard_size=1,
            parallel_shards=5,
        ),
        Component(
            name="dqn_asterix",
            config=ExperimentConfig(
                AGENT="dqn",
                ENV="minatar",
                AGENT_HYPERS=MinAtarDQNConfig(),
                ENV_HYPERS=MinAtarConfig(),
            ),
            sweep={"ENV_HYPERS.GAME": ["asterix"]},
            seeds=list(range(30)),
            shard_size=None,  # one vmap of every seed - what a GPU wants
        ),
    ],
    # Agent0's five packed runs share one GPU through CUDA MPS, with a CPU
    # each, as in benchmarking_atari_50m_realv1.
    slurm=SlurmResources(time="11:59:00", gpus=1, mps=True, cpus_per_task=5),
)
