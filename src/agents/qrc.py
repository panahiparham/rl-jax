from dataclasses import dataclass
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from experiment.hypers import traced

from components import (
    Batch,
    BufferState,
    DualHeadNetwork,
    build_buffer,
    build_q_network,
    epsilon_greedy_action,
    linear_epsilon,
)


@dataclass(frozen=True, kw_only=True)
class QRCConfig:
    """QRC's hyperparameters: DQN without a target network, plus a second head
    h estimating the expected TD error, which corrects the Q update.

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
    EPSILON_DECAY_STEPS: int = traced(100_000)
    HIDDEN_SIZE: int = 64
    # "mlp"/"mlp_ln" (vector obs), "nature_cnn"/"nature_cnn_ln" (image obs) or
    # "minatar_cnn"/"minatar_cnn_ln" (MinAtar's 10x10 grid)
    NETWORK_PRESET: str = "mlp"
    ADAM_EPS: float = traced(1e-8)
    REWARD_CLIP: bool = False  # clip to sign(reward) for the buffer and update only
    BETA: float = traced(1.0)  # L2 penalty on the h head's weights and bias
    CORRECTION_SCALE: float = traced(1.0)  # weight of the h-corrected TD term


@dataclass(frozen=True, kw_only=True)
class MinAtarQRCConfig(QRCConfig):
    """QRC hyperparameters matching DQN's MinAtar baseline."""

    LR: float = traced(0.00025)
    ADAM_EPS: float = traced(3.125e-4)
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 32
    TOTAL_TIMESTEPS: int = 10_000_000
    LEARNING_STARTS: int = traced(1_000)
    TRAIN_FREQUENCY: int = traced(4)
    GAMMA: float = traced(0.99)
    EPSILON_START: float = traced(1.0)
    EPSILON_END: float = traced(0.01)
    EPSILON_DECAY_STEPS: int = traced(250_000)
    NETWORK_PRESET: str = "minatar_cnn"
    REWARD_CLIP: bool = False


class QRCState(NamedTuple):
    network: DualHeadNetwork
    opt_state: Any
    buffer_state: BufferState
    t: jax.Array


def qrc_loss(
    network: DualHeadNetwork,
    batch: Batch,
    beta: float | jax.Array,
    correction_scale: float | jax.Array,
) -> jax.Array:
    """Masked mean of QRC's per-sample loss plus the h head's L2 penalty.

    The Q head descends the TD error with a correction through the bootstrap,
    weighted by h; h regresses onto the TD error.
    """
    sg = jax.lax.stop_gradient
    q_sa, h_sa = jax.vmap(network)(batch.obs)
    q_next, _ = jax.vmap(network)(batch.boot_obs)
    action = batch.action[:, None]
    v = jnp.take_along_axis(q_sa, action, axis=-1).squeeze(-1)
    h = jnp.take_along_axis(h_sa, action, axis=-1).squeeze(-1)
    v_next = jnp.max(q_next, axis=-1)

    td_error = batch.ret + batch.discount * sg(v_next) - v
    corrected_td_error = batch.ret + batch.discount * v_next - sg(v)
    q_loss = 0.5 * jnp.square(td_error) + correction_scale * sg(h) * corrected_td_error
    h_loss = 0.5 * jnp.square(sg(td_error) - h)

    per_sample = jnp.where(batch.mask, q_loss + h_loss, 0.0)
    mean_loss = jnp.sum(per_sample) / jnp.maximum(jnp.sum(batch.mask), 1)
    h_params = jax.tree.leaves(network.h_out)
    penalty = 0.5 * beta * sum(jnp.sum(jnp.square(p)) for p in h_params)
    return mean_loss + penalty


class QRCAgent:
    def __init__(self, config: QRCConfig):
        self._config = config
        self._buffer = build_buffer(
            config.BUFFER,
            capacity=config.BUFFER_SIZE,
            batch_size=config.BATCH_SIZE,
            n_step=config.N_STEP,
            gamma=config.GAMMA,
        )
        self._optimizer = optax.adam(config.LR, eps=config.ADAM_EPS)

    def init(self, key: jax.Array, observation_space, action_space):
        q = build_q_network(
            self._config.NETWORK_PRESET,
            observation_space.shape,
            action_space.n,
            self._config.HIDDEN_SIZE,
            key,
        )
        network = DualHeadNetwork(q)
        return QRCState(
            network=network,
            opt_state=self._optimizer.init(eqx.filter(network, eqx.is_array)),
            buffer_state=self._buffer.init(observation_space),
            t=jnp.asarray(0, jnp.int32),
        )

    def act(self, state: QRCState, key: jax.Array, obs: jax.Array):
        config = self._config
        q_values = state.network.q(obs)
        epsilon = linear_epsilon(
            state.t,
            config.EPSILON_START,
            config.EPSILON_END,
            config.LEARNING_STARTS,
            config.EPSILON_DECAY_STEPS,
        )
        return epsilon_greedy_action(q_values, epsilon, key)

    def _train_step(self, state: QRCState, key: jax.Array):
        config = self._config
        batch = self._buffer.sample(state.buffer_state, key)
        grads = eqx.filter_grad(qrc_loss)(
            state.network, batch, config.BETA, config.CORRECTION_SCALE
        )
        params = eqx.filter(state.network, eqx.is_array)
        updates, opt_state = self._optimizer.update(grads, state.opt_state, params)
        return state._replace(
            network=eqx.apply_updates(state.network, updates), opt_state=opt_state
        )

    def update(
        self,
        state: QRCState,
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
