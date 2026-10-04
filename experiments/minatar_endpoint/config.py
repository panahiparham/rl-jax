import dataclasses
from pathlib import Path

from experiment.design import Component, Experiment, SlurmResources

from agents.ddqn import DDQNConfig
from agents.dqn import MinAtarDQNConfig
from agents.endpoint import EndpointConfig
from environments.minatar import MinAtarConfig
from main import ExperimentConfig

_GAMES = ["asterix", "breakout", "freeway", "space_invaders"]
_SEEDS = list(range(30))

# Every component learns with the minatar_dqn hyperparameters; only the memory
# differs between them.
_LEARNER = dataclasses.asdict(MinAtarDQNConfig())


def _component(name: str, agent: str, hypers: DDQNConfig) -> Component:
    return Component(
        name=name,
        config=ExperimentConfig(
            AGENT=agent,
            ENV="minatar",
            AGENT_HYPERS=hypers,
            ENV_HYPERS=MinAtarConfig(),
        ),
        sweep={"ENV_HYPERS.GAME": _GAMES},
        seeds=_SEEDS,
        shard_size=None,  # one vmap of every seed - what a GPU wants
    )


def _ddqn(name: str, buffer_size: int) -> Component:
    hypers = DDQNConfig(**{**_LEARNER, "BUFFER_SIZE": buffer_size})
    return _component(name, "ddqn", hypers)


# 1k recency steps, then chained 10-step transitions; 4 of every 32 rows come
# from the long-term buffer.
def _endpoint(name: str, long_term_size: int) -> Component:
    hypers = EndpointConfig(
        **{
            **_LEARNER,
            "BUFFER_SIZE": 1_000,
            "LONG_TERM_SIZE": long_term_size,
            "LONG_TERM_N_STEP": 10,
            "LONG_TERM_BATCH_SIZE": 4,
            "EXPECTILE_TAU": 0.7,
        }
    )
    return _component(name, "endpoint", hypers)


EXPERIMENT = Experiment(
    name="minatar_endpoint",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        _ddqn("ddqn_minatar", buffer_size=100_000),
        _ddqn("ddqn_medium_minatar", buffer_size=10_000),
        _ddqn("ddqn_small_minatar", buffer_size=2_000),
        _endpoint("endpoint_minatar", long_term_size=9_000),
        _endpoint("endpoint_small_minatar", long_term_size=1_000),
    ],
    slurm=SlurmResources(time="12:00:00", gpus=1),
)
