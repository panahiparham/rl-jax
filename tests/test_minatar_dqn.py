"""DQN on MinAtar: the hyperparameters of reg-duel-q's baseline."""

from __future__ import annotations

from agents import AGENTS
from agents.dqn import DQNAgent, DQNConfig, MinAtarDQNConfig


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
