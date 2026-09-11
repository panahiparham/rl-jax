from collections.abc import Callable
from typing import Any, NamedTuple

from environments.atari import AtariConfig
from environments.atari import build as build_atari
from environments.catch import CatchConfig
from environments.catch import build as build_catch
from environments.classic_control import (
    AcrobotConfig,
    CartpoleConfig,
    MountainCarConfig,
    build_acrobot,
    build_cartpole,
    build_mountaincar,
)
from environments.pinball import PinballConfig
from environments.pinball import build as build_pinball
from environments.real_atari import RealAtariConfig
from environments.real_atari import build as build_real_atari


class EnvSpec(NamedTuple):
    config_cls: type  # the environment's config dataclass
    build: Callable[..., Any]  # (config) -> env (an EnvProtocol)
    vmappable: bool = True  # False for a stateful env that cannot run under jax.vmap


ENVIRONMENTS: dict[str, EnvSpec] = {
    "pinball": EnvSpec(PinballConfig, build_pinball),
    "mountaincar": EnvSpec(MountainCarConfig, build_mountaincar),  # gymnax
    "cartpole": EnvSpec(CartpoleConfig, build_cartpole),  # gymnax
    "acrobot": EnvSpec(AcrobotConfig, build_acrobot),  # gymnax
    "catch": EnvSpec(CatchConfig, build_catch),
    "atari": EnvSpec(AtariConfig, build_atari, vmappable=False),  # stateful ale-py FFI
    "real-atari": EnvSpec(RealAtariConfig, build_real_atari, vmappable=False),
}


def get_config(name: str):
    if name not in ENVIRONMENTS:
        raise ValueError(
            f"unknown environment {name!r}; registered: {sorted(ENVIRONMENTS)}"
        )
    return ENVIRONMENTS[name].config_cls()


__all__ = [
    "ENVIRONMENTS",
    "AcrobotConfig",
    "AtariConfig",
    "CartpoleConfig",
    "CatchConfig",
    "EnvSpec",
    "MountainCarConfig",
    "PinballConfig",
    "RealAtariConfig",
    "get_config",
]
