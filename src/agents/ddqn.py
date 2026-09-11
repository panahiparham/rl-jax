from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from agents.dqn import DQNAgent, DQNConfig, DQNState
from components import QNetwork


@dataclass(frozen=True, kw_only=True)
class DDQNConfig(DQNConfig):
    """Double-DQN's hyperparameters, identical to DQN's."""


class DDQNAgent(DQNAgent):
    def _train_step(self, state: DQNState, key: jax.Array):
        batch = self._buffer.sample(state.buffer_state, key)

        def loss_fn(q: QNetwork) -> jax.Array:
            q_sa = jax.vmap(q)(batch.obs)
            q_a = jnp.take_along_axis(q_sa, batch.action[:, None], axis=-1).squeeze(-1)
            boot_action = jnp.argmax(jax.vmap(q)(batch.boot_obs), axis=-1)
            boot = jnp.take_along_axis(
                jax.vmap(state.target_q)(batch.boot_obs),
                boot_action[:, None],
                axis=-1,
            ).squeeze(-1)
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
