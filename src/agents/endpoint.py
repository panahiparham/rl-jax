from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from experiment.hypers import traced

from agents.ddqn import DDQNAgent, DDQNConfig
from agents.dqn import DQNState
from components import QNetwork, build_buffer
from components.buffers.composed import ComposedBuffer
from components.buffers.selector import NStepSelector
from components.buffers.transition import TransitionBuffer


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


class EndpointAgent(DDQNAgent):
    _config: EndpointConfig

    def __init__(self, config: EndpointConfig) -> None:
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
                TransitionBuffer(
                    capacity=config.LONG_TERM_SIZE,
                    batch_size=config.LONG_TERM_BATCH_SIZE,
                ),
            ]
        )
        self._optimizer = optax.adam(config.LR, eps=config.ADAM_EPS)

    def _train_step(self, state: DQNState, key: jax.Array) -> DQNState:
        config = self._config
        recent, long_term = self._buffer.sample_components(state.buffer_state, key)

        def loss_fn(q: QNetwork) -> jax.Array:
            # recency rows: double-Q target
            q_sa = jax.vmap(q)(recent.obs)
            q_a = jnp.take_along_axis(q_sa, recent.action[:, None], axis=-1).squeeze(-1)
            boot_action = jnp.argmax(jax.vmap(q)(recent.boot_obs), axis=-1)
            boot = jnp.take_along_axis(
                jax.vmap(state.target_q)(recent.boot_obs),
                boot_action[:, None],
                axis=-1,
            ).squeeze(-1)
            target = jax.lax.stop_gradient(recent.ret + recent.discount * boot)
            recent_err = jnp.where(recent.mask, jnp.square(q_a - target), 0.0)

            # long-term rows: SARSA target from the stored boot action
            q_sa = jax.vmap(q)(long_term.obs)
            q_a = jnp.take_along_axis(
                q_sa, long_term.action[:, None], axis=-1
            ).squeeze(-1)
            boot = jnp.take_along_axis(
                jax.vmap(state.target_q)(long_term.boot_obs),
                long_term.boot_action[:, None],
                axis=-1,
            ).squeeze(-1)
            target = jax.lax.stop_gradient(long_term.ret + long_term.discount * boot)
            delta = target - q_a
            tau = config.EXPECTILE_TAU
            weight = jnp.where(delta > 0, tau, 1.0 - tau)
            # 2x keeps the original's weighting, whose recency rows carried 0.5
            long_term_err = jnp.where(
                long_term.mask, 2.0 * weight * jnp.square(delta), 0.0
            )

            count = jnp.sum(recent.mask) + jnp.sum(long_term.mask)
            total = jnp.sum(recent_err) + jnp.sum(long_term_err)
            return total / jnp.maximum(count, 1)

        grads = eqx.filter_grad(loss_fn)(state.q)
        updates, opt_state = self._optimizer.update(
            grads, state.opt_state, eqx.filter(state.q, eqx.is_array)
        )
        return state._replace(
            q=eqx.apply_updates(state.q, updates), opt_state=opt_state
        )
