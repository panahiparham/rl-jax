from pathlib import Path

from experiment.design import Component, Experiment

from agents.dqn import DQNConfig
from environments.atari import AtariConfig
from main import ExperimentConfig

_GAMES = ["battle_zone", "ms_pacman"]

# DQNConfig.SEED is declared but never read by the agent - the run's actual
# PRNG seed is Component.seeds below. Sweeping it here is a dummy hyper: it
# only forces each replicate into its own run_id, so 3 identical (hyper,
# seed) runs land as 3 distinct rows instead of deduping to one.
_REPLICATES = [0, 1, 2]

_DQN_HYPERS = {
    "TOTAL_TIMESTEPS": 2_500_000,       # 10M frames at FRAMESKIP=4
    "LR": 6.25e-05,
    "ADAM_EPS": 1.5e-4,
    "BUFFER_SIZE": 100_000,
    "BATCH_SIZE": 32,
    "LEARNING_STARTS": 20_000,
    "TRAIN_FREQUENCY": 4,
    "TARGET_NETWORK_FREQUENCY": 8_000,
    "GAMMA": 0.99,
    "EPSILON_START": 1.0,
    "EPSILON_END": 0.01,
    "EPSILON_FRACTION": 0.1,
    "NETWORK_PRESET": "nature_cnn",
}

_ATARI = AtariConfig()

_REAL_ATARI = AtariConfig(
    LIMITED_ACTION_SPACE=False,
    NOOP_MAX=0,
)

EXPERIMENT = Experiment(
    name="atari_reproducibility",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name="dqn_atari",
            config=ExperimentConfig(
                AGENT="dqn",
                ENV="atari",
                AGENT_HYPERS=DQNConfig(**_DQN_HYPERS, REWARD_CLIP=True),
                ENV_HYPERS=_ATARI,
            ),
            sweep={"ENV_HYPERS.GAME": _GAMES, "AGENT_HYPERS.SEED": _REPLICATES},
            seeds=[0],
            shard_size=1,
        ),
        Component(
            name="dqn_real_atari",
            config=ExperimentConfig(
                AGENT="dqn",
                ENV="atari",
                AGENT_HYPERS=DQNConfig(**_DQN_HYPERS, REWARD_CLIP=False),
                ENV_HYPERS=_REAL_ATARI,
            ),
            sweep={"ENV_HYPERS.GAME": _GAMES, "AGENT_HYPERS.SEED": _REPLICATES},
            seeds=[0],
            shard_size=1,
        ),
    ],
)
