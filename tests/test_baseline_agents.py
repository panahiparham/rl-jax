import jax
import numpy as np

from agents.reservoir import ReservoirAgent, ReservoirConfig
from agents.unanchored import UnanchoredAgent, UnanchoredConfig
from components.buffers.composed import ComposedState
from components.buffers.reservoir import ReservoirState
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


# Reservoir


def test_reservoir_runs_under_jit_and_vmap():
    """A vmapped run trains and caps the reservoir at its capacity.

    With 100 steps and 32 recency slots, more 1-step transitions age out than
    the 16-slot reservoir holds, so it ends full.
    """
    hypers = {**_SMALL, "LONG_TERM_SIZE": 16}
    agent = ReservoirAgent(ReservoirConfig(TOTAL_TIMESTEPS=100, **hypers))

    metrics, final_carry = _run_vmapped(agent, 100)

    assert metrics["reward"].shape == (3, 100)
    assert np.isfinite(np.asarray(final_carry[1].q.layer3.weight)).all()
    buffer_state = final_carry[1].buffer_state
    assert isinstance(buffer_state, ComposedState)
    reservoir = buffer_state.states[2]
    assert isinstance(reservoir, ReservoirState)
    assert (np.asarray(reservoir.seen) > 16).all()
    assert np.asarray(np.minimum(reservoir.seen, 16)).tolist() == [16, 16, 16]


def test_reservoir_seeds_draw_their_own_slots():
    """Each seed's reservoir gets its own key, derived from the agent's key."""
    agent = ReservoirAgent(ReservoirConfig(TOTAL_TIMESTEPS=10, **_SMALL))

    _metrics, final_carry = _run_vmapped(agent, 10)

    reservoir = final_carry[1].buffer_state.states[2]
    assert isinstance(reservoir, ReservoirState)
    keys = np.asarray(jax.random.key_data(reservoir.key))
    assert len({tuple(row) for row in keys}) == 3
