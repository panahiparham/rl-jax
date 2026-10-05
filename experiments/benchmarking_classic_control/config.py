"""
Define: Agent0 on MountainCar, Acrobot and Cartpole, 100 seeds each.

Agent0 takes the tuned DQN hypers from ``experiments/tuned`` - including the
tuned LR - minus the target network, with the LN variant of the MLP. Seeds are
offset by 10 - ``range(10, 110)`` - like ``experiments/tuned``'s, so the curves
compare seed for seed with its DQN.
"""

from __future__ import annotations

from pathlib import Path

from experiment.design import Component, Experiment

from agents.agent0 import Agent0Config
from environments.classic_control import (
    AcrobotConfig,
    CartpoleConfig,
    MountainCarConfig,
)
from main import ExperimentConfig

_LR = 0.0009765625  # tuned DQN's LR for all three environments
_AGENT0_HYPERS = {
    "LR": _LR,
    "TOTAL_TIMESTEPS": 100_000,
    "BUFFER_SIZE": 10_000,
    "BATCH_SIZE": 64,
    "LEARNING_STARTS": 1_000,
    "TRAIN_FREQUENCY": 1,
    "GAMMA": 0.99,
    "EPSILON_START": 0.1,
    "EPSILON_END": 0.1,
    "NETWORK_PRESET": "mlp_ln",
}

_ENVIRONMENTS = {
    "mountaincar": ("mountaincar", MountainCarConfig(EPISODE_CUTOFF=1_000)),
    "acrobot": ("acrobot", AcrobotConfig(EPISODE_CUTOFF=500)),
    "cartpole": ("cartpole", CartpoleConfig(EPISODE_CUTOFF=500)),
}

EXPERIMENT = Experiment(
    name="benchmarking_classic_control",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name=f"agent0_{key}",
            config=ExperimentConfig(
                AGENT="agent0",
                ENV=env,
                AGENT_HYPERS=Agent0Config(**_AGENT0_HYPERS),
                ENV_HYPERS=env_hypers,
            ),
            seeds=list(range(10, 110)),
            shard_size=10,
        )
        for key, (env, env_hypers) in _ENVIRONMENTS.items()
    ],
)
