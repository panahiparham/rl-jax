import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from experiment.hypers import traced

from components import (
    BufferState,
    NatureCNN,
    NatureCNNLN,
    QNetwork,
    QNetworkLN,
    build_buffer,
    epsilon_greedy_action,
    linear_epsilon,
)


@dataclass(frozen=True, kw_only=True)
class Agent0Config:
    """Agent0's hyperparameters: DQN without a target network - it bootstraps
    directly off the online q-network instead of a periodically-synced copy.

    The traced ones are read as numbers while a run steps, so runs differing
    only in those can be computed together. The rest fix shapes and objects -
    buffer capacity, the network, the length of the scan - and runs differing
    in any of them are computed separately.
    """

    LR: float = traced(3e-4)
    BUFFER: str = "uniform"
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 64
    N_STEP: int = 1
    TOTAL_TIMESTEPS: int = 200_000
    LEARNING_STARTS: int = traced(1_000)
    TRAIN_FREQUENCY: int = traced(1)
    GAMMA: float = traced(0.99)
    EPSILON_START: float = traced(1.0)
    EPSILON_END: float = traced(0.05)
    EPSILON_FRACTION: float = traced(0.5)
    HIDDEN_SIZE: int = 64
    # "mlp"/"mlp_ln" (vector obs) or "nature_cnn"/"nature_cnn_ln" (image obs)
    NETWORK_PRESET: str = "mlp_ln"
    ADAM_EPS: float = traced(1e-8)
    SEED: int = 42
    REWARD_CLIP: bool = False  # clip to sign(reward) for the buffer and update only


class Agent0State(NamedTuple):
    q: eqx.Module
    opt_state: Any
    buffer_state: BufferState
    t: jax.Array


class Agent0Agent:
    def __init__(self, config: Agent0Config):
        self._config = config
        self._buffer = build_buffer(
            config.BUFFER,
            capacity=config.BUFFER_SIZE,
            batch_size=config.BATCH_SIZE,
            n_step=config.N_STEP,
            gamma=config.GAMMA,
        )
        self._optimizer = optax.adam(config.LR, eps=config.ADAM_EPS)

    def _build_q(self, key, obs_shape, action_dim) -> eqx.Module:
        if self._config.NETWORK_PRESET == "nature_cnn":
            return NatureCNN(obs_shape, action_dim, key)
        if self._config.NETWORK_PRESET == "nature_cnn_ln":
            return NatureCNNLN(obs_shape, action_dim, key)
        if self._config.NETWORK_PRESET == "mlp":
            return QNetwork(
                math.prod(obs_shape), action_dim, self._config.HIDDEN_SIZE, key
            )
        if self._config.NETWORK_PRESET == "mlp_ln":
            return QNetworkLN(
                math.prod(obs_shape), action_dim, self._config.HIDDEN_SIZE, key
            )
        raise ValueError(f"unknown NETWORK_PRESET {self._config.NETWORK_PRESET!r}")

    def init(self, key: jax.Array, observation_space, action_space):
        obs_shape = observation_space.shape
        q = self._build_q(key, obs_shape, action_space.n)
        return Agent0State(
            q=q,
            opt_state=self._optimizer.init(eqx.filter(q, eqx.is_array)),
            buffer_state=self._buffer.init(observation_space),
            t=jnp.asarray(0, jnp.int32),
        )

    def act(self, state: Agent0State, key: jax.Array, obs: jax.Array):
        config = self._config
        q_values = state.q(obs)
        epsilon = linear_epsilon(
            state.t,
            config.EPSILON_START,
            config.EPSILON_END,
            config.TOTAL_TIMESTEPS * config.EPSILON_FRACTION,
        )
        return epsilon_greedy_action(q_values, epsilon, q_values.shape[-1], key)

    def _train_step(self, state: Agent0State, key: jax.Array):
        """One gradient step on the masked n-step TD loss, bootstrapping off
        the same q-network being updated (no target network)."""
        batch = self._buffer.sample(state.buffer_state, key)

        def loss_fn(q: eqx.Module) -> jax.Array:
            q_sa = jax.vmap(q)(batch.obs)
            q_a = jnp.take_along_axis(q_sa, batch.action[:, None], axis=-1).squeeze(-1)
            boot = jnp.max(jax.vmap(q)(batch.boot_obs), axis=-1)
            target = jax.lax.stop_gradient(batch.ret + batch.discount * boot)
            sq_err = jnp.where(batch.mask, jnp.square(q_a - target), 0.0)
            return jnp.sum(sq_err) / jnp.maximum(jnp.sum(batch.mask), 1)

        grads = eqx.filter_grad(loss_fn)(state.q)
        updates, opt_state = self._optimizer.update(
            grads, state.opt_state, eqx.filter(state.q, eqx.is_array)
        )
        return state._replace(
            q=eqx.apply_updates(state.q, updates), opt_state=opt_state
        )

    def update(
        self,
        state: Agent0State,
        key: jax.Array,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        termination: jax.Array,
        truncation: jax.Array,
        discount: jax.Array,
    ):
        config = self._config
        if config.REWARD_CLIP:
            reward = jnp.sign(reward)
        buffer_state = self._buffer.add(
            state.buffer_state,
            obs,
            action,
            reward,
            termination,
            truncation,
            discount,
        )
        state = state._replace(buffer_state=buffer_state)

        can_train = (
            self._buffer.can_sample(buffer_state)
            & (state.t >= config.LEARNING_STARTS)
            & (state.t % config.TRAIN_FREQUENCY == 0)
        )
        state = jax.lax.cond(
            can_train, lambda: self._train_step(state, key), lambda: state
        )
        return state._replace(t=state.t + 1)
