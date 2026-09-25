import jax
import jax.numpy as jnp

Scalar = float | jax.Array


def epsilon_greedy_action(
    q_values: jax.Array, epsilon: jax.Array, action_dim: int, key: jax.Array
) -> jax.Array:
    is_max = q_values == jnp.max(q_values)
    greedy_probs = is_max / jnp.sum(is_max)
    probs = epsilon / action_dim + (1.0 - epsilon) * greedy_probs
    return jax.random.categorical(key, jnp.log(probs)).astype(jnp.int32)


def linear_epsilon(
    t: jax.Array, start: Scalar, end: Scalar, decay_steps: Scalar
) -> jax.Array:
    return jnp.maximum(end, start - (start - end) * (t / decay_steps))
