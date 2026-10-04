from pathlib import Path

from experiment.design import Component, Experiment, SlurmResources

from agents.dqn import MinAtarDQNConfig
from environments.minatar import MinAtarConfig
from main import ExperimentConfig

_GAMES = ["asterix", "breakout", "freeway", "space_invaders"]

EXPERIMENT = Experiment(
    name="minatar_dqn",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name="dqn_minatar",
            config=ExperimentConfig(
                AGENT="dqn",
                ENV="minatar",
                AGENT_HYPERS=MinAtarDQNConfig(),
                ENV_HYPERS=MinAtarConfig(),
            ),
            sweep={"ENV_HYPERS.GAME": _GAMES},
            seeds=list(range(30)),
            shard_size=None,  # one vmap of every seed - what a GPU wants
        ),
    ],
    slurm=SlurmResources(time="02:59:00", gpus=1),
)
