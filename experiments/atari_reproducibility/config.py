from pathlib import Path

from experiment.design import Component, Experiment

from agents.dqn import DQNConfig
from environments.atari import RevisitingALEConfig
from main import ExperimentConfig

_GAMES = ["battle_zone", "ms_pacman"]

# Every replicate reruns the same config at the same seed. Each gets its own
# component, and so its own table, since identical runs within one component
# share a run id and would be stored once.
_REPLICATES = [0, 1, 2]

_DQN_HYPERS = {
    "TOTAL_TIMESTEPS": 2_500_000,       # 10M frames at FRAMESKIP=4
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

_ATARI = RevisitingALEConfig()

_REAL_ATARI = RevisitingALEConfig(
    LIMITED_ACTION_SPACE=False,
    NOOP_MAX=0,
)

_SETTINGS = {
    "dqn_atari": (DQNConfig(**_DQN_HYPERS, REWARD_CLIP=True), _ATARI),
    "dqn_real_atari": (DQNConfig(**_DQN_HYPERS, REWARD_CLIP=False), _REAL_ATARI),
}

EXPERIMENT = Experiment(
    name="atari_reproducibility",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name=f"{setting}_replicate_{replicate}",
            config=ExperimentConfig(
                AGENT="dqn",
                ENV="atari",
                AGENT_HYPERS=agent_hypers,
                ENV_HYPERS=env_hypers,
            ),
            sweep={"ENV_HYPERS.GAME": _GAMES},
            seeds=[0],
            shard_size=1,
        )
        for setting, (agent_hypers, env_hypers) in _SETTINGS.items()
        for replicate in _REPLICATES
    ],
)
