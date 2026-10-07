import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from agents.dqn import DQNAgent, DQNConfig, DQNState
from components import (
    QNetworkHL,
    QNetworkHLLN,
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
    NETWORK_PRESET: str = "mlp_hl"

    # Histogram params
    NUM_BINS: int = 100
    # Support
    SUPPORT_LOWER_BOUND: float = -100
    SUPPORT_UPPER_BOUND: float = 100
    SIGMA_RATIO: float = 2.0


class DQNHistogramAgent(DQNAgent):
    def _build_q(self, key, obs_shape, action_dim) -> eqx.Module:
        config = self._config
        if config.NETWORK_PRESET == "mlp_hl":
            return QNetworkHL(math.prod(obs_shape), action_dim, config.HIDDEN_SIZE, key, config.NUM_BINS, config.SUPPORT_LOWER_BOUND, config.SUPPORT_UPPER_BOUND, config.SIGMA_RATIO)
        elif config.NETWORK_PRESET == "mlp_hl_ln":
            return QNetworkHLLN(math.prod(obs_shape), action_dim, config.HIDDEN_SIZE, key, config.NUM_BINS, config.SUPPORT_LOWER_BOUND, config.SUPPORT_UPPER_BOUND, config.SIGMA_RATIO)
        raise ValueError(f"unknown NETWORK_PRESET {config.NETWORK_PRESET!r}")

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
        return epsilon_greedy_action(q_values, epsilon, key)

    def _train_step(self, state: DQNState, key: jax.Array):
        """One gradient step on the masked n-step TD loss."""
        batch = self._buffer.sample(state.buffer_state, key)

        def loss_fn(q: QNetworkHL) -> jax.Array:
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
