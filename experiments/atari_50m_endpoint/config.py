from pathlib import Path

from experiment.design import Component, Experiment, SlurmResources

from agents.ddqn import DDQNConfig
from agents.endpoint import EndpointConfig
from agents.reservoir import ReservoirConfig
from agents.unanchored import UnanchoredConfig
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
_SEEDS = list(range(10))

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


def _ddqn(name: str, **hypers: int) -> Component:
    return Component(
        name=name,
        config=ExperimentConfig(
            AGENT="ddqn",
            ENV="atari",
            AGENT_HYPERS=DDQNConfig(
                **{**_DDQN_HYPERS, **hypers}, REWARD_CLIP=True
            ),
            ENV_HYPERS=EPRAtariConfig(),
        ),
        sweep={"ENV_HYPERS.GAME": _GAMES},
        seeds=_SEEDS,
        shard_size=1,
        parallel_shards=4,
    )


# Endpoint replay's layout, storing 1-step transitions instead: every 10th for
# unanchored, a reservoir sample for reservoir. Both train DDQN on all rows, as
# in the original endpoint_*_noexpectilesarsanstep and reservoir_* runs.
def _baseline(
    name: str, agent: str, hypers: UnanchoredConfig | ReservoirConfig
) -> Component:
    return Component(
        name=name,
        config=ExperimentConfig(
            AGENT=agent,
            ENV="atari",
            AGENT_HYPERS=hypers,
            ENV_HYPERS=EPRAtariConfig(),
        ),
        sweep={"ENV_HYPERS.GAME": _GAMES},
        seeds=_SEEDS,
        shard_size=1,
        parallel_shards=4,
    )


EXPERIMENT = Experiment(
    name="atari_50m_endpoint",
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
            seeds=_SEEDS,
            shard_size=1,
            parallel_shards=4,
        ),
        _ddqn("ddqn_medium_atari", BUFFER_SIZE=100_000),
        _ddqn("ddqn_medium_nstep_atari", BUFFER_SIZE=100_000, N_STEP=10),
        _ddqn("ddqn_small_atari", BUFFER_SIZE=20_000),
        _ddqn("ddqn_small_nstep_atari", BUFFER_SIZE=20_000, N_STEP=10),
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
            seeds=_SEEDS,
            shard_size=1,
            parallel_shards=4,
        ),
        # The original endpoint_small run: 10k recency, 10k long-term.
        Component(
            name="endpoint_small_atari",
            config=ExperimentConfig(
                AGENT="endpoint",
                ENV="atari",
                AGENT_HYPERS=EndpointConfig(LONG_TERM_SIZE=10_000),
                ENV_HYPERS=EPRAtariConfig(),
            ),
            sweep={"ENV_HYPERS.GAME": _GAMES},
            seeds=_SEEDS,
            shard_size=1,
            parallel_shards=4,
        ),
        _baseline("unanchored_atari", "unanchored", UnanchoredConfig()),
        _baseline(
            "unanchored_small_atari",
            "unanchored",
            UnanchoredConfig(LONG_TERM_SIZE=10_000),
        ),
        _baseline("reservoir_atari", "reservoir", ReservoirConfig()),
        _baseline(
            "reservoir_small_atari",
            "reservoir",
            ReservoirConfig(LONG_TERM_SIZE=10_000),
        ),
    ],
    # Four packed runs (parallel_shards=4) share one GPU through CUDA MPS,
    # with a CPU each. The endpoint agent's runtime is unknown; 12h leaves
    # room above the expected ~5.5h.
    slurm=SlurmResources(time="12:00:00", gpus=1, mps=True, cpus_per_task=4),
)
