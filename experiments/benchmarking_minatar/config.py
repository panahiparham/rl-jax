from pathlib import Path

from experiment.design import Component, Experiment, SlurmResources

from agents.agent0 import Agent0Config
from environments.minatar import MinAtarConfig
from main import ExperimentConfig

_GAMES = ["asterix", "breakout", "freeway", "space_invaders"]

# Agent0 has no target network and uses the LN variant of the MinAtar CNN. The
# rest follows MinAtarDQNConfig, reg-duel-q's MinAtar baseline.
_AGENT0_HYPERS = {
    "LR": 0.00025,
    "ADAM_EPS": 3.125e-4,
    "BUFFER_SIZE": 100_000,
    "BATCH_SIZE": 32,
    "TOTAL_TIMESTEPS": 10_000_000,
    "LEARNING_STARTS": 1_000,
    "TRAIN_FREQUENCY": 4,
    "GAMMA": 0.99,
    "EPSILON_START": 1.0,
    "EPSILON_END": 0.01,
    "EPSILON_DECAY_STEPS": 250_000,
    "NETWORK_PRESET": "minatar_cnn_ln",
}

EXPERIMENT = Experiment(
    name="benchmarking_minatar",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name="agent0_minatar",
            config=ExperimentConfig(
                AGENT="agent0",
                ENV="minatar",
                AGENT_HYPERS=Agent0Config(**_AGENT0_HYPERS, REWARD_CLIP=False),
                ENV_HYPERS=MinAtarConfig(),
            ),
            sweep={"ENV_HYPERS.GAME": _GAMES},
            seeds=list(range(30)),
            shard_size=None,  # one vmap of every seed - what a GPU wants
        ),
    ],
    slurm=SlurmResources(time="02:59:00", gpus=1),
)
