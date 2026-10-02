from dataclasses import dataclass
from typing import cast

import jax
import optax
from experiment.hypers import traced

from agents.ddqn import DDQNAgent, DDQNConfig
from agents.dqn import DQNState
from components import build_buffer
from components.buffers.composed import ComposedBuffer, ComposedState
from components.buffers.contract import ObservationSpace
from components.buffers.reservoir import ReservoirBuffer, ReservoirState
from components.buffers.selector import NStepSelector


@dataclass(frozen=True, kw_only=True)
class ReservoirConfig(DDQNConfig):
    """Reservoir replay's hyperparameters; defaults are for Atari experiments."""

    TOTAL_TIMESTEPS: int = 12_500_000
    LR: float = traced(6.25e-5)
    ADAM_EPS: float = traced(1.5e-4)
    BUFFER_SIZE: int = 10_000
    BATCH_SIZE: int = 32
    LEARNING_STARTS: int = traced(20_000)
    TRAIN_FREQUENCY: int = traced(4)
    TARGET_NETWORK_FREQUENCY: int = traced(8_000)
    EPSILON_END: float = traced(0.01)
    EPSILON_DECAY_STEPS: int = traced(250_000)
    NETWORK_PRESET: str = "nature_cnn"
    REWARD_CLIP: bool = True
    LONG_TERM_SIZE: int = 90_000
    LONG_TERM_BATCH_SIZE: int = 4


class ReservoirAgent(DDQNAgent):
    def __init__(self, config: ReservoirConfig) -> None:
        self._config = config
        self._buffer = ComposedBuffer(
            [
                build_buffer(
                    config.BUFFER,
                    capacity=config.BUFFER_SIZE,
                    batch_size=config.BATCH_SIZE,
                    n_step=config.N_STEP,
                    gamma=config.GAMMA,
                ),
                NStepSelector(n_step=1, gamma=config.GAMMA),
                ReservoirBuffer(
                    capacity=config.LONG_TERM_SIZE,
                    batch_size=config.LONG_TERM_BATCH_SIZE,
                ),
            ]
        )
        self._optimizer = optax.adam(config.LR, eps=config.ADAM_EPS)

    def init(
        self,
        key: jax.Array,
        observation_space: ObservationSpace,
        action_space: object,
    ) -> DQNState:
        state: DQNState = super().init(key, observation_space, action_space)
        # Each seed draws its own reservoir slots; folding in leaves the
        # network's initialization key untouched.
        buffer_state = cast(ComposedState, state.buffer_state)
        *upstream, reservoir = buffer_state.states
        reservoir = cast(ReservoirState, reservoir)._replace(
            key=jax.random.fold_in(key, 1)
        )
        states = (*upstream, reservoir)
        return state._replace(buffer_state=buffer_state._replace(states=states))
