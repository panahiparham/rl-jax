"""Target-network syncing for ``dqn`` and ``ddqn``.

The target network is a hard copy of the online network: on a step where
``t % TARGET_NETWORK_FREQUENCY == 0`` it becomes the online parameters
exactly, and between two such steps it does not move while the online
network keeps training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from agents.ddqn import DDQNAgent, DDQNConfig
from agents.dqn import DQNAgent, DQNConfig
from environments import ENVIRONMENTS
from environments.catch import CatchConfig
from main import interaction

AGENTS = [(DQNAgent, DQNConfig), (DDQNAgent, DDQNConfig)]


def _params(network) -> list[np.ndarray]:
    arrays, _ = eqx.partition(network, eqx.is_array)
    return [np.asarray(leaf) for leaf in jax.tree.leaves(arrays)]


def _final_state(agent_cls, config_cls, *, freq: int, steps: int):
    agent = agent_cls(
        config_cls(
            TOTAL_TIMESTEPS=steps,
            BUFFER_SIZE=64,
            BATCH_SIZE=2,
            LEARNING_STARTS=1,
            HIDDEN_SIZE=8,
            TARGET_NETWORK_FREQUENCY=freq,
        )
    )
    env = ENVIRONMENTS["catch"].build(CatchConfig(EPISODE_CUTOFF=10))
    run = jax.jit(lambda key: interaction(key, agent, env, steps))
    _metrics, final_carry = jax.block_until_ready(run(jax.random.key(0)))
    return final_carry[1]


@pytest.mark.parametrize("agent_cls,config_cls", AGENTS)
def test_target_is_a_hard_copy_on_sync(agent_cls, config_cls):
    freq = 8
    # t runs 0..freq over freq + 1 updates, so the last one is a sync step
    state = _final_state(agent_cls, config_cls, freq=freq, steps=freq + 1)
    for target, online in zip(_params(state.target_q), _params(state.q), strict=True):
        np.testing.assert_array_equal(target, online)


@pytest.mark.parametrize("agent_cls,config_cls", AGENTS)
def test_target_is_frozen_between_syncs(agent_cls, config_cls):
    freq = 16
    # both runs stop after the t = 0 sync and before the t = freq one
    early = _final_state(agent_cls, config_cls, freq=freq, steps=freq - 1)
    late = _final_state(agent_cls, config_cls, freq=freq, steps=freq)

    for before, after in zip(
        _params(early.target_q), _params(late.target_q), strict=True
    ):
        np.testing.assert_array_equal(before, after)
    # the online network did keep training, so the equality above is not trivial
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(_params(early.q), _params(late.q), strict=True)
    )
