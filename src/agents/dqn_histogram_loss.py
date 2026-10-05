import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from experiment.hypers import traced
from agents.dqn import DQNConfig

from components import (
    BufferState,
    NatureCNN,
    NatureCNNLN,
    QNetwork,
    QNetworkLN,
    QNetworkHistogramLoss,
    build_buffer,
    epsilon_greedy_action,
    linear_epsilon,
)


@dataclass(frozen=True, kw_only=True)
class DQNHistogramConfig(DQNConfig):
    """DQN hyperparameters with histogramloss.

    The traced ones are read as numbers while a run steps, so runs differing
    only in those can be computed together. The rest fix shapes and objects -
    buffer capacity, the network, the length of the scan - and runs differing
    in any of them are computed separately.
    """

    # Different network architecture needed for histogram loss
    NETWORK_PRESET: str = "mlp_histogram"

    # Histogram params
    NUM_BINS: int = 100
    # Support
    SUPPORT_LOWER_BOUND: float = -100
    SUPPORT_UPPER_BOUND: float = 100
    SIGMA_RATIO: float = 2.0


class DQNState(NamedTuple):
    q: eqx.Module
    target_q: eqx.Module
    opt_state: Any
    buffer_state: BufferState
    t: jax.Array


class DQNHistogramAgent:
    def __init__(self, config: DQNHistogramConfig):
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
        config = self._config
        if config.NETWORK_PRESET == "mlp_histogram":
            return QNetworkHistogramLoss(math.prod(obs_shape), action_dim, config.HIDDEN_SIZE, key, config.NUM_BINS, config.SUPPORT_LOWER_BOUND, config.SUPPORT_UPPER_BOUND, config.SIGMA_RATIO)
        raise ValueError(f"unknown NETWORK_PRESET {config.NETWORK_PRESET!r}")

    def init(self, key: jax.Array, observation_space, action_space):
        obs_shape = observation_space.shape
        q = self._build_q(key, obs_shape, action_space.n)
        return DQNState(
            q=q,
            target_q=q,
            opt_state=self._optimizer.init(eqx.filter(q, eqx.is_array)),
            buffer_state=self._buffer.init(observation_space),
            t=jnp.asarray(0, jnp.int32),
        )

    def act(self, state: DQNState, key: jax.Array, obs: jax.Array):
        config = self._config
        logits = state.q(obs)
        q_values = state.q.get_action_values(logits)

        epsilon = linear_epsilon(
            state.t,
            config.EPSILON_START,
            config.EPSILON_END,
            config.LEARNING_STARTS,
            config.EPSILON_DECAY_STEPS,
        )
        return epsilon_greedy_action(q_values, epsilon, q_values.shape[-1], key)

    def _train_step(self, state: DQNState, key: jax.Array):
        """One gradient step on the masked n-step TD loss."""
        batch = self._buffer.sample(state.buffer_state, key)

        def loss_fn(q: QNetworkHistogramLoss) -> jax.Array:
            # Get logits from network
            logits = jax.vmap(q)(batch.obs) # batch by num_actions by num_bins
            logits_a = logits[jnp.arange(logits.shape[0]), batch.action]

            # Compute histogram targets
            target_logits = jax.vmap(state.target_q)(batch.boot_obs)
            q_values = jax.vmap(state.target_q.get_action_values)(target_logits)
            max_q = jnp.max(q_values, axis=-1)
            target = jax.lax.stop_gradient(batch.ret + batch.discount * max_q)

            # If the target value is super far outside the range of the histogram it could cause numerical issues
            target = jnp.clip(
                target,
                state.target_q.histogram_support_min,
                state.target_q.histogram_support_max,
            )

            histogram_targets = jax.vmap(state.target_q.get_histogram_values)(target)

            cross_entropy = optax.losses.softmax_cross_entropy(logits_a, histogram_targets)
            cross_entropy = jnp.where(batch.mask, cross_entropy, 0.0)
            return jnp.sum(cross_entropy) / jnp.maximum(jnp.sum(batch.mask), 1)

        grads = eqx.filter_grad(loss_fn)(state.q)
        updates, opt_state = self._optimizer.update(
            grads, state.opt_state, eqx.filter(state.q, eqx.is_array)
        )
        return state._replace(
            q=eqx.apply_updates(state.q, updates), opt_state=opt_state
        )

    def update(
        self,
        state: DQNState,
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
        target_q = jax.lax.cond(
            state.t % config.TARGET_NETWORK_FREQUENCY == 0,
            lambda: state.q,
            lambda: state.target_q,
        )
        return state._replace(target_q=target_q, t=state.t + 1)
