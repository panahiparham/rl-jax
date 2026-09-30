from dataclasses import dataclass

from experiment.hypers import traced

from agents.ddqn import DDQNConfig


@dataclass(frozen=True, kw_only=True)
class EndpointConfig(DDQNConfig):
    """Endpoint replay's hyperparameters; defaults are for Atari experiments."""

    TOTAL_TIMESTEPS: int = 12_500_000
    LR: float = traced(6.25e-5)
    ADAM_EPS: float = traced(1.5e-4)
    BUFFER_SIZE: int = 10_000
    BATCH_SIZE: int = 32
    LEARNING_STARTS: int = traced(20_000)
    TRAIN_FREQUENCY: int = traced(4)
    TARGET_NETWORK_FREQUENCY: int = traced(8_000)
    EPSILON_END: float = traced(0.01)
    # decays over 250k steps after warmup; the original finished at step 250k
    # (230k steps of decay)
    EPSILON_DECAY_STEPS: int = traced(250_000)
    NETWORK_PRESET: str = "nature_cnn"
    REWARD_CLIP: bool = True
    LONG_TERM_SIZE: int = 90_000
    LONG_TERM_N_STEP: int = 10
    LONG_TERM_BATCH_SIZE: int = 4
    EXPECTILE_TAU: float = traced(0.7)
