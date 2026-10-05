"""
Define: Agent0 and Agent1 on Pinball's empty, box and easy settings, 100 seeds
each.

Both agents take the tuned DQN hypers from ``experiments/tuned`` - including the
tuned LR - minus the target network, with the LN variant of the MLP. Seeds are
offset by 10 - ``range(10, 110)`` - like ``experiments/tuned``'s, so the curves
compare seed for seed with its DQN.
"""

from __future__ import annotations

from pathlib import Path

from experiment.design import Component, Experiment

from agents.agent0 import Agent0Config
from agents.agent1 import Agent1Config
from environments.pinball import PinballConfig
from main import ExperimentConfig

_SETTINGS = ["empty", "box", "easy"]

_LR = 0.00390625  # tuned DQN's LR for every Pinball setting
_AGENT_HYPERS = {
    "LR": _LR,
    "TOTAL_TIMESTEPS": 100_000,
    "BUFFER_SIZE": 10_000,
    "BATCH_SIZE": 32,
    "LEARNING_STARTS": 1_000,
    "EPSILON_START": 0.1,
    "EPSILON_END": 0.1,
    "HIDDEN_SIZE": 32,
    "GAMMA": 0.99,
    "TRAIN_FREQUENCY": 1,
    "NETWORK_PRESET": "mlp_ln",
}

_AGENTS = {"agent0": Agent0Config, "agent1": Agent1Config}

EXPERIMENT = Experiment(
    name="benchmarking_pinball",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name=f"{agent}_pinball_{setting}",
            config=ExperimentConfig(
                AGENT=agent,
                ENV="pinball",
                AGENT_HYPERS=config_cls(**_AGENT_HYPERS),
                ENV_HYPERS=PinballConfig(SETTING=setting, EPISODE_CUTOFF=1_000),
            ),
            seeds=list(range(10, 110)),
            shard_size=10,
        )
        for agent, config_cls in _AGENTS.items()
        for setting in _SETTINGS
    ],
)
