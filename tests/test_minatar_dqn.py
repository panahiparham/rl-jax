"""DQN on MinAtar: the hyperparameters of reg-duel-q's baseline."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from agents import AGENTS
from agents.ddqn import DDQNConfig
from agents.dqn import DQNAgent, DQNConfig, MinAtarDQNConfig
from agents.endpoint import EndpointConfig
from agents.unanchored import UnanchoredConfig
from environments.minatar import MinAtarConfig
from main import ExperimentConfig, process_shard


class TestMinAtarDQNConfig:
    def test_hyperparameters_match_reg_duel_q(self):
        """The reproduction target is fixed: 10M steps, 1000 warmup steps, an
        update every 4 steps, a target copy every 1000 steps, epsilon 1.0 to
        0.01 over 250k steps, and Adam with a 3.125e-4 epsilon."""
        config = MinAtarDQNConfig()

        assert config.TOTAL_TIMESTEPS == 10_000_000
        assert config.BATCH_SIZE == 32
        assert config.BUFFER_SIZE == 100_000
        assert config.LEARNING_STARTS == 1_000
        assert config.TRAIN_FREQUENCY == 4
        assert config.TARGET_NETWORK_FREQUENCY == 1_000
        assert config.GAMMA == 0.99
        assert config.EPSILON_START == 1.0
        assert config.EPSILON_END == 0.01
        assert config.EPSILON_DECAY_STEPS == 250_000
        assert config.LR == 0.00025
        assert config.ADAM_EPS == 3.125e-4
        assert not config.REWARD_CLIP

    def test_uses_the_minatar_network(self):
        """The preset is the small MinAtar CNN, not the MLP default."""
        assert MinAtarDQNConfig().NETWORK_PRESET == "minatar_cnn"

    def test_is_a_dqn_config_for_the_dqn_agent(self):
        """It needs no agent of its own: the registered DQN builds from it."""
        config = MinAtarDQNConfig()

        assert isinstance(config, DQNConfig)
        assert isinstance(AGENTS["dqn"].build(config), DQNAgent)


class TestSmokeRun:
    @pytest.mark.parametrize(
        "game", ["asterix", "breakout", "freeway", "space_invaders"]
    )
    def test_dqn_trains_on_every_game_vmapped_over_seeds(self, game):
        """A short run per game goes through the real stack: the environment,
        the MinAtar network, the replay buffer and the shard runner."""
        config = ExperimentConfig(
            AGENT="dqn",
            ENV="minatar",
            AGENT_HYPERS=MinAtarDQNConfig(
                TOTAL_TIMESTEPS=60,
                BUFFER_SIZE=64,
                LEARNING_STARTS=8,
                TARGET_NETWORK_FREQUENCY=16,
            ),
            ENV_HYPERS=MinAtarConfig(GAME=game),
        )
        runs = process_shard([config] * 2, [0, 1])

        assert [run["reward"].shape for run in runs] == [(60,)] * 2
        assert all(np.isfinite(run["reward"]).all() for run in runs)


class TestReplayAgents:
    @pytest.mark.parametrize("agent", ["ddqn", "endpoint", "unanchored"])
    def test_replay_agents_train_on_minatar(self, agent):
        """DDQN, endpoint and unanchored replay train on MinAtar's bool
        observations, vmapped over seeds, with the experiment's layout scaled
        down: a recency buffer plus a long-term memory."""
        learner = dataclasses.asdict(
            MinAtarDQNConfig(
                TOTAL_TIMESTEPS=120,
                LEARNING_STARTS=40,
                TARGET_NETWORK_FREQUENCY=16,
            )
        )
        memory = {"BUFFER_SIZE": 32, "LONG_TERM_SIZE": 32}
        hypers = {
            "ddqn": DDQNConfig(**{**learner, "BUFFER_SIZE": 64}),
            "endpoint": EndpointConfig(**{**learner, **memory}),
            "unanchored": UnanchoredConfig(**{**learner, **memory}),
        }[agent]
        config = ExperimentConfig(
            AGENT=agent,
            ENV="minatar",
            AGENT_HYPERS=hypers,
            ENV_HYPERS=MinAtarConfig(GAME="freeway"),
        )
        runs = process_shard([config] * 2, [0, 1])

        assert [run["reward"].shape for run in runs] == [(120,)] * 2
        assert all(np.isfinite(run["reward"]).all() for run in runs)
