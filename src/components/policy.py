import jax
import jax.numpy as jnp

Scalar = float | jax.Array


def epsilon_greedy_probs(q_values: jax.Array, epsilon: Scalar) -> jax.Array:
    is_max = q_values == jnp.max(q_values)
    greedy_probs = is_max / jnp.sum(is_max)
    return epsilon / q_values.shape[-1] + (1.0 - epsilon) * greedy_probs


def epsilon_greedy_action(
    q_values: jax.Array, epsilon: jax.Array, key: jax.Array
) -> jax.Array:
    probs = epsilon_greedy_probs(q_values, epsilon)
    return jax.random.categorical(key, jnp.log(probs)).astype(jnp.int32)


def linear_epsilon(
    t: jax.Array,
    start: Scalar,
    end: Scalar,
    warmup_steps: Scalar,
    decay_steps: Scalar,
) -> jax.Array:
    progress = (t - warmup_steps) / jnp.maximum(decay_steps, 1)
    return start + (end - start) * jnp.clip(progress, 0.0, 1.0)
