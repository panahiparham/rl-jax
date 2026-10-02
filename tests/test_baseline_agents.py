import jax
import numpy as np

from agents.unanchored import UnanchoredAgent, UnanchoredConfig
from components.buffers.composed import ComposedState
from components.buffers.transition import TransitionState
from environments import ENVIRONMENTS
from environments.catch import CatchConfig
from main import interaction

_SMALL = {
    "BUFFER_SIZE": 32,
    "BATCH_SIZE": 8,
    "LONG_TERM_SIZE": 64,
    "LONG_TERM_BATCH_SIZE": 2,
    "LEARNING_STARTS": 8,
    "TRAIN_FREQUENCY": 1,
    "TARGET_NETWORK_FREQUENCY": 100,
    "NETWORK_PRESET": "mlp",
    "HIDDEN_SIZE": 16,
    "LR": 1e-2,
}


def _run_vmapped(agent, total: int):
    env = ENVIRONMENTS["catch"].build(CatchConfig(EPISODE_CUTOFF=5))
    run = jax.jit(jax.vmap(lambda key: interaction(key, agent, env, total)))
    return jax.block_until_ready(run(jax.random.split(jax.random.key(0), 3)))


# Unanchored


def test_unanchored_runs_under_jit_and_vmap():
    """A vmapped run trains and fills long-term with thinned 1-step transitions.

    With 100 steps and 32 recency slots, 68 timesteps age out. Keeping every
    10th of their transitions leaves at most 7 in the long-term buffer.
    """
    agent = UnanchoredAgent(UnanchoredConfig(TOTAL_TIMESTEPS=100, **_SMALL))

    metrics, final_carry = _run_vmapped(agent, 100)

    assert metrics["reward"].shape == (3, 100)
    assert np.isfinite(np.asarray(final_carry[1].q.layer3.weight)).all()
    buffer_state = final_carry[1].buffer_state
    assert isinstance(buffer_state, ComposedState)
    long_term = buffer_state.states[3]
    assert isinstance(long_term, TransitionState)
    sizes = np.asarray(long_term.size)
    assert ((sizes > 0) & (sizes <= 7)).all()
