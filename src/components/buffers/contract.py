from typing import NamedTuple

import jax


class TimeStep(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    termination: jax.Array
    truncation: jax.Array
    discount: jax.Array


class Batch(NamedTuple):
    obs: jax.Array
    action: jax.Array
    ret: jax.Array
    discount: jax.Array
    boot_obs: jax.Array
    mask: jax.Array
