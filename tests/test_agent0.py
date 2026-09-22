"""Agent0: DQN without a target network - it bootstraps directly off the
online q-network it is training, rather than a periodically-synced copy.
"""

from __future__ import annotations

import jax
import numpy as np

from agents import AGENTS
from agents.agent0 import Agent0Agent, Agent0Config, Agent0State
from environments import ENVIRONMENTS
from environments.catch import CatchConfig
from main import interaction


def test_agent0_state_has_no_target_network_field():
    assert "target_q" not in Agent0State._fields


def test_agent0_default_network_preset_is_mlp_ln():
    assert Agent0Config().NETWORK_PRESET == "mlp_ln"


def test_agent0_is_registered():
    spec = AGENTS["agent0"]
    assert spec.config_cls is Agent0Config
    assert spec.agent_cls is Agent0Agent


def test_agent0_runs_under_jit_and_trains():
    env = ENVIRONMENTS["catch"].build(CatchConfig(EPISODE_CUTOFF=5))
    agent = Agent0Agent(
        Agent0Config(
            TOTAL_TIMESTEPS=20,
            BUFFER_SIZE=32,
            BATCH_SIZE=4,
            LEARNING_STARTS=4,
            HIDDEN_SIZE=8,
        )
    )
    run = jax.jit(lambda key: interaction(key, agent, env, 20))
    metrics, final_carry = jax.block_until_ready(run(jax.random.key(0)))
    assert metrics["reward"].shape == (20,)
    assert np.isfinite(np.asarray(final_carry[1].q.layer3.weight)).all()
