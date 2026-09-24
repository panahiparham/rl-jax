import os
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class AtariConfig:
    # Dopamine baselines settings (matching published curves, but current version
    # uses 100k frame cutoff)
    GAME: str = "pong"
    FRAMESKIP: int = 4
    STICKY_ACTIONS: float = 0.25
    MAX_FRAMES_PER_EPISODE: int = 108_000
    IMG_HEIGHT: int = 84
    IMG_WIDTH: int = 84
    GRAYSCALE: bool = True
    LIMITED_ACTION_SPACE: bool = True
    NOOP_MAX: int = 0
    STACK_NUM: int = 4
    MAXPOOL: bool = True
    USE_FIRE_RESET: bool = False
    ZERO_DISCOUNT_ON_LIFE_LOSS: bool = False
    REWARD_CLIPPING: bool = False

@dataclass(frozen=True)
class ClassicAtariConfig(AtariConfig):
    # Original DQN settings (DQN Zoo)
    STICKY_ACTIONS: float = 0.0
    NOOP_MAX: int = 31
    ZERO_DISCOUNT_ON_LIFE_LOSS: bool = True

@dataclass(frozen=True)
class RevisitingALEConfig(AtariConfig):
    # revisiting ALE recommendations (using frameskip 4 instead of 5 on purpose)
    MAX_FRAMES_PER_EPISODE: int = 18_000
    LIMITED_ACTION_SPACE: bool = False

@dataclass(frozen=True)
class EPRAtariConfig(AtariConfig): # Endpoint replay settings
    ZERO_DISCOUNT_ON_LIFE_LOSS: bool = True


class _Box:
    def __init__(
        self, shape: tuple[int, ...], dtype: Any, frame_channels: int | None = None
    ):
        self.shape = shape
        self.dtype = dtype
        self.frame_channels = frame_channels


class _Discrete:
    def __init__(self, n: int):
        self.n = int(n)


class AtariState(NamedTuple):
    handle: jax.Array
    lives: jax.Array


class AtariEnvLike:
    def __init__(self, vector_env):
        if vector_env.num_envs != 1:
            raise ValueError(
                "AtariEnvLike supports a single environment, got "
                f"num_envs={vector_env.num_envs}."
            )
        self._num_envs = 1
        self._init_handle, self._reset_fn, self._step_fn = vector_env.xla()
        obs_shape = vector_env.single_observation_space.shape
        self._colour = len(obs_shape) == 4
        if self._colour:
            frames, height, width, colours = obs_shape
            self._obs_shape = (height, width, frames * colours)
            self._frame_channels = colours
        else:
            frames, height, width = obs_shape
            self._obs_shape = (height, width, frames)  # channel-last
            self._frame_channels = 1
        self._n_actions = int(vector_env.single_action_space.n)

    def observation_space(self, params: object | None = None):
        return _Box(self._obs_shape, jnp.uint8, self._frame_channels)

    def action_space(self, params: object | None = None):
        return _Discrete(self._n_actions)

    def _to_hwc(self, obs: jax.Array):
        if self._colour:
            # (1, frames, H, W, colours) -> (H, W, frames * colours), folding the
            # frame stack and RGB channels into a single channel-last axis.
            stacked = jnp.transpose(obs[0], (1, 2, 0, 3))
            return stacked.reshape(self._obs_shape)
        return jnp.transpose(obs[0], (1, 2, 0))  # (1, frames, H, W) -> (H, W, frames)

    def reset(self, key: jax.Array, params: object | None = None):
        seed = jax.random.randint(key, (1,), 0, jnp.iinfo(jnp.int32).max).astype(
            jnp.int32
        )
        handle, (obs, info) = self._reset_fn(self._init_handle, seed)
        return self._to_hwc(obs), AtariState(handle=handle, lives=info["lives"][0])

    def step(
        self,
        key: jax.Array,
        state: AtariState,
        action: jax.Array,
        params: object | None = None,
    ):
        del key  # ale keeps its own RNG
        actions = jnp.asarray(action, dtype=jnp.int32).reshape((1,))
        handle, (obs, rewards, terminations, truncations, info) = self._step_fn(
            state.handle, actions
        )
        # Guards the FFI boundary against an intermittent macOS-CPU ale-py XLA segfault.
        handle = jax.lax.optimization_barrier(handle)

        reward = rewards[0].astype(jnp.float32)
        terminated = terminations[0]
        truncated = truncations[0]
        state = AtariState(handle=handle, lives=info["lives"][0])
        return self._to_hwc(obs), state, reward, terminated, truncated, {}


class AtariEnv:
    def __init__(self, inner: AtariEnvLike, zero_discount_on_life_loss: bool = False):
        self._env = inner
        self._zero_discount_on_life_loss = zero_discount_on_life_loss

    def observation_space(self):
        return self._env.observation_space()

    def action_space(self):
        return self._env.action_space()

    def init(self, key: jax.Array):
        obs, state = self._env.reset(key)
        return state, obs

    def step(self, state: AtariState, key: jax.Array, action: jax.Array):
        obs, stepped, reward, termination, truncation, _info = self._env.step(
            key, state, action
        )
        # ale does not count losing the last life; that step is the game over
        life_lost = (stepped.lives < state.lives) & (stepped.lives > 0)
        no_bootstrap = termination | (self._zero_discount_on_life_loss & life_lost)

        def _consume_dead_step(carry):
            st, _boundary_obs = carry
            fresh, st, _r, _te, _tr, _i = self._env.step(key, st, action)
            return st, fresh

        next_state, next_obs = jax.lax.cond(
            termination | truncation,
            _consume_dead_step,
            lambda carry: carry,
            (stepped, obs),
        )
        discount = 1.0 - no_bootstrap.astype(jnp.float32)
        return next_state, reward, termination, truncation, discount, next_obs


def require_ale_xla(game: str):
    import ale_py._ale_py as _c
    import ale_py.roms

    script = "scripts/install_ale_wheel.sh"
    if not hasattr(_c, "VectorXLAReset"):
        raise RuntimeError(
            "ale-py was installed without the XLA vector-env FFI (no VectorXLAReset). "
            f"Run `{script}` to install a build with XLA "
            "(see README 'Atari (XLA) setup')."
        )
    if jax.default_backend() != "cpu" and not hasattr(_c, "VectorXLAResetGPU"):
        raise RuntimeError(
            "jax is on GPU but ale-py has no CUDA XLA FFI targets "
            "(no VectorXLAResetGPU) - this is the PyPI build, not the PR #707 "
            f"one. Run `{script}` (and `uv sync --extra cuda`)."
        )
    try:
        rom = ale_py.roms.get_rom_path(game)
    except Exception as exc:
        raise RuntimeError(
            f"Atari ROM '{game}' could not be looked up: {exc}. The PR #707 wheels "
            f"bundle all ROMs; run `{script}`, or `AutoROM --accept-license` to "
            "fetch ROMs."
        ) from exc
    if rom is None or not os.path.exists(str(rom)):
        raise RuntimeError(
            f"Atari ROM '{game}' is not installed. The PR #707 wheels bundle all ROMs; "
            f"run `{script}`, or `AutoROM --accept-license` to fetch ROMs."
        )


def build(config: AtariConfig):
    import ale_py  # lazy: the base project installs without the atari extra

    require_ale_xla(config.GAME)

    kwargs = {
        "game": config.GAME,
        "num_envs": 1,
        "max_num_frames_per_episode": int(config.MAX_FRAMES_PER_EPISODE),
        "frameskip": int(config.FRAMESKIP),
        "repeat_action_probability": float(config.STICKY_ACTIONS),
        "img_height": int(config.IMG_HEIGHT),
        "img_width": int(config.IMG_WIDTH),
        "grayscale": bool(config.GRAYSCALE),
        "full_action_space": not config.LIMITED_ACTION_SPACE,
        "noop_max": int(config.NOOP_MAX),
        "stack_num": int(config.STACK_NUM),
        "maxpool": bool(config.MAXPOOL),
        "use_fire_reset": bool(config.USE_FIRE_RESET),
        "reward_clipping": bool(config.REWARD_CLIPPING),
    }
    return AtariEnv(
        AtariEnvLike(ale_py.AtariVectorEnv(**kwargs)),
        zero_discount_on_life_loss=config.ZERO_DISCOUNT_ON_LIFE_LOSS,
    )
