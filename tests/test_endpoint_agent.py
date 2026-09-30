import equinox as eqx
import jax
import numpy as np
import pytest

from agents.endpoint import EndpointAgent, EndpointConfig
from components.buffers.composed import ComposedState
from components.buffers.transition import TransitionState
from environments import ENVIRONMENTS
from environments.catch import CatchConfig
from main import interaction


def _small_config(total: int, tau: float = 0.7) -> EndpointConfig:
    return EndpointConfig(
        TOTAL_TIMESTEPS=total,
        BUFFER_SIZE=32,
        BATCH_SIZE=8,
        LONG_TERM_SIZE=64,
        LONG_TERM_N_STEP=3,
        LONG_TERM_BATCH_SIZE=2,
        LEARNING_STARTS=8,
        TRAIN_FREQUENCY=1,
        TARGET_NETWORK_FREQUENCY=100,
        NETWORK_PRESET="mlp",
        HIDDEN_SIZE=16,
        LR=1e-2,
        # all-random actions -> every run sees identical data
        EPSILON_START=1.0,
        EPSILON_END=1.0,
        EXPECTILE_TAU=tau,
    )


def _trained_q_leaves(total: int, tau: float) -> list[jax.Array]:
    env = ENVIRONMENTS["catch"].build(CatchConfig(EPISODE_CUTOFF=10))
    agent = EndpointAgent(_small_config(total, tau))
    run = jax.jit(lambda key: interaction(key, agent, env, total))
    _metrics, final_carry = jax.block_until_ready(run(jax.random.key(0)))
    return jax.tree.leaves(eqx.filter(final_carry[1].q, eqx.is_array))


def test_endpoint_runs_under_jit_and_vmap():
    """A vmapped run trains and moves aged-out experience into long-term."""
    env = ENVIRONMENTS["catch"].build(CatchConfig(EPISODE_CUTOFF=5))
    agent = EndpointAgent(_small_config(100))
    run = jax.jit(jax.vmap(lambda key: interaction(key, agent, env, 100)))

    metrics, final_carry = jax.block_until_ready(
        run(jax.random.split(jax.random.key(0), 3))
    )

    assert metrics["reward"].shape == (3, 100)
    assert np.isfinite(np.asarray(final_carry[1].q.layer3.weight)).all()
    buffer_state = final_carry[1].buffer_state
    assert isinstance(buffer_state, ComposedState)
    long_term = buffer_state.states[2]
    assert isinstance(long_term, TransitionState)
    assert (np.asarray(long_term.size) > 0).all()


@pytest.mark.parametrize(
    ("total", "tau_changes_learning"),
    [(200, True), (30, False)],
    ids=["long-term-filled", "nothing-aged-out"],
)
def test_expectile_only_affects_long_term_rows(
    total: int, tau_changes_learning: bool
):
    """The expectile reshapes learning only once long-term rows exist.

    With 30 steps nothing leaves the 32-step recency buffer, so every
    long-term row is masked and the expectile cannot matter.
    """
    symmetric = _trained_q_leaves(total, tau=0.5)
    asymmetric = _trained_q_leaves(total, tau=0.9)

    differs = any(
        not np.allclose(a, b) for a, b in zip(symmetric, asymmetric, strict=True)
    )
    assert differs is tau_changes_learning
