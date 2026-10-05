"""Agent1: Agent0 with an Expected Sarsa target. Its bootstrap is the
epsilon-greedy expectation of the next q-values instead of their max, so it
matches Agent0 exactly when epsilon is 0 and departs from it otherwise.
"""

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np

from agents.agent0 import Agent0Agent, Agent0Config
from agents.agent1 import Agent1Agent, Agent1Config
from environments import ENVIRONMENTS
from environments.catch import CatchConfig
from main import interaction

NUM_STEPS = 20
SHARED_HYPERS = {
    "TOTAL_TIMESTEPS": NUM_STEPS,
    "BUFFER_SIZE": 32,
    "BATCH_SIZE": 4,
    "LEARNING_STARTS": 4,
    "HIDDEN_SIZE": 8,
}


def _trained_q_leaves(agent):
    env = ENVIRONMENTS["catch"].build(CatchConfig(EPISODE_CUTOFF=5))
    run = jax.jit(lambda key: interaction(key, agent, env, NUM_STEPS))
    _, final_carry = jax.block_until_ready(run(jax.random.key(0)))
    return jax.tree.leaves(eqx.filter(final_carry[1].q, eqx.is_array))


def _agent0_and_agent1_leaves(epsilon: float):
    hypers = {**SHARED_HYPERS, "EPSILON_START": epsilon, "EPSILON_END": epsilon}
    agent0 = Agent0Agent(Agent0Config(**hypers))
    agent1 = Agent1Agent(Agent1Config(**hypers))
    return _trained_q_leaves(agent0), _trained_q_leaves(agent1)


def test_greedy_policy_reduces_to_agent0_q_learning():
    """With epsilon 0 the policy expectation is the max, so Agent1 trains
    to the same parameters as Agent0 from the same seed."""
    agent0_leaves, agent1_leaves = _agent0_and_agent1_leaves(epsilon=0.0)
    for agent0_leaf, agent1_leaf in zip(agent0_leaves, agent1_leaves, strict=True):
        assert np.allclose(agent0_leaf, agent1_leaf)


def test_exploratory_policy_departs_from_agent0_q_learning():
    """With epsilon 1 the bootstrap is the mean of the next q-values, so
    Agent1's parameters diverge from Agent0's despite identical experience."""
    agent0_leaves, agent1_leaves = _agent0_and_agent1_leaves(epsilon=1.0)
    assert not all(
        np.allclose(agent0_leaf, agent1_leaf)
        for agent0_leaf, agent1_leaf in zip(agent0_leaves, agent1_leaves, strict=True)
    )
    assert all(np.isfinite(leaf).all() for leaf in agent1_leaves)
