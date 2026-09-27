from pathlib import Path

from experiment.design import Component, Experiment, SlurmResources

from agents.dqn import DQNConfig
from environments.atari import AtariConfig
from main import ExperimentConfig

_GAMES = ["battle_zone", "ms_pacman"]

# Every replicate reruns the same config at the same seed. Each gets its own
# component, and so its own table, since identical runs within one component
# share a run id and would be stored once. A worker packs only one component's
# shards at a time, so each replicate runs its two games in parallel.
_REPLICATES = [0, 1, 2, 3, 4]

_DQN_HYPERS = {
    "TOTAL_TIMESTEPS": 100_000,
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
    name="atari_reproducibility_deterministic_ops",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name=f"dqn_atari_replicate_{replicate}",
            config=ExperimentConfig(
                AGENT="dqn",
                ENV="atari",
                AGENT_HYPERS=DQNConfig(**_DQN_HYPERS, REWARD_CLIP=True),
                ENV_HYPERS=AtariConfig(),
            ),
            sweep={"ENV_HYPERS.GAME": _GAMES},
            seeds=[0],
            shard_size=1,
            parallel_shards=len(_GAMES),
        )
        for replicate in _REPLICATES
    ],
    # The packed games share one GPU through CUDA MPS.
    slurm=SlurmResources(time="00:59:00", gpus=1, mps=True, cpus_per_task=5),
)
