from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from experiment.hypers import traced

from components import BufferState, build_buffer


@dataclass(frozen=True, kw_only=True)
class RandomBufferConfig:
    TOTAL_TIMESTEPS: int = 100_000
    BUFFER: str = "uniform"
    BUFFER_SIZE: int = 10_000
    BATCH_SIZE: int = 32
    N_STEP: int = 1
    GAMMA: float = traced(0.99)
    SEED: int = 0


class RandomBufferState(NamedTuple):
    action_dim: jax.Array
    buffer_state: BufferState


class RandomBufferAgent:
    """Acts uniformly at random and fills a replay buffer without learning."""

    def __init__(self, config: RandomBufferConfig):
        self._config = config
        self._buffer = build_buffer(
            config.BUFFER,
            capacity=config.BUFFER_SIZE,
            batch_size=config.BATCH_SIZE,
            n_step=config.N_STEP,
            gamma=config.GAMMA,
        )

    def init(self, key: jax.Array, observation_space, action_space):
        del key
        return RandomBufferState(
            action_dim=jnp.asarray(action_space.n, jnp.int32),
            buffer_state=self._buffer.init(observation_space),
        )

    def act(self, state: RandomBufferState, key: jax.Array, obs: jax.Array):
        del obs
        return jax.random.randint(key, (), 0, state.action_dim, dtype=jnp.int32)

    def update(
        self,
        state: RandomBufferState,
        key: jax.Array,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        termination: jax.Array,
        truncation: jax.Array,
        discount: jax.Array,
    ):
        del key
        buffer_state = self._buffer.add(
            state.buffer_state,
            obs,
            action,
            reward,
            termination,
            truncation,
            discount,
        )
        return state._replace(buffer_state=buffer_state)

    def can_sample(self, state: RandomBufferState):
        return self._buffer.can_sample(state.buffer_state)
