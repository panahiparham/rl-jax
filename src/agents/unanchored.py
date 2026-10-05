from dataclasses import dataclass

import optax
from experiment.hypers import traced

from agents.ddqn import DDQNAgent, DDQNConfig
from components import build_buffer
from components.buffers.composed import ComposedBuffer
from components.buffers.selector import NStepSelector
from components.buffers.subsample import Subsample
from components.buffers.transition import TransitionBuffer


@dataclass(frozen=True, kw_only=True)
class UnanchoredConfig(DDQNConfig):
    """Unanchored replay's hyperparameters; defaults are for Atari experiments."""

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
    LONG_TERM_N_STEP: int = 1
    SUBSAMPLE: int = 10


class UnanchoredAgent(DDQNAgent):
    def __init__(self, config: UnanchoredConfig) -> None:
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
                NStepSelector(n_step=config.LONG_TERM_N_STEP, gamma=config.GAMMA),
                Subsample(every=config.SUBSAMPLE),
                TransitionBuffer(
                    capacity=config.LONG_TERM_SIZE,
                    batch_size=config.LONG_TERM_BATCH_SIZE,
                ),
            ]
        )
        self._optimizer = optax.adam(config.LR, eps=config.ADAM_EPS)
