from pathlib import Path

from experiment.design import Component, Experiment, SlurmResources

from agents.agent0 import Agent0Config
from agents.agent1 import Agent1Config
from environments.minatar import MinAtarConfig
from main import ExperimentConfig

_GAMES = ["asterix", "breakout", "freeway", "space_invaders"]

# Agent0 and Agent1 have no target network and use the LN variant of the
# MinAtar CNN. The rest follows MinAtarDQNConfig, reg-duel-q's MinAtar baseline.
_AGENT_HYPERS = {
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

_AGENTS = {"agent0": Agent0Config, "agent1": Agent1Config}

EXPERIMENT = Experiment(
    name="benchmarking_minatar",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name=f"{agent}_minatar",
            config=ExperimentConfig(
                AGENT=agent,
                ENV="minatar",
                AGENT_HYPERS=config_cls(**_AGENT_HYPERS, REWARD_CLIP=False),
                ENV_HYPERS=MinAtarConfig(),
            ),
            sweep={"ENV_HYPERS.GAME": _GAMES},
            seeds=list(range(30)),
            shard_size=None,  # one vmap of every seed - what a GPU wants
        )
        for agent, config_cls in _AGENTS.items()
    ],
    slurm=SlurmResources(time="02:59:00", gpus=1),
)
