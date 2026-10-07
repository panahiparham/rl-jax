"""
Define: DQN and Agent0, with and without the HL-Gauss histogram loss, on Classic
Control environments (Cartpole, MountainCar), 100 seeds each.

- ``dqn_hl`` / ``dqn_hl_ln_notarget``: each variant of
  ``experiments/histogram_ln_notarget_classic_control`` at its best grid point
  per environment (see ``_BEST`` below).
- ``dqn``: ``experiments/tuned``'s DQN, at its tuned LR.
- ``agent0``: ``experiments/benchmarking_classic_control``'s Agent0.

The baselines are rerun here rather than read from those experiments' stores. Their
configs are taken from those experiments' components, so the hypers match exactly.
Seeds are ``range(10, 110)`` like theirs, so none of the sweep's seeds (0-9) that
picked the histogram configs are reused. Acrobot is added once its sweep finishes.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

from experiment.design import Component, Experiment, SlurmResources

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent


def _load_config(experiment: str):
    """Another experiment's ``config`` module, under a name that can't clash with this one."""
    spec = importlib.util.spec_from_file_location(
        f"{experiment}_config", _EXPERIMENTS_DIR / experiment / "config.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SWEEP = _load_config("histogram_ln_notarget_classic_control").EXPERIMENT
_TUNED = _load_config("tuned").EXPERIMENT
_BENCHMARKING = _load_config("benchmarking_classic_control").EXPERIMENT

# Best (support, SIGMA_RATIO, NUM_BINS, LR) per (variant, env), by mean lifetime
# return, read off experiments/histogram_ln_notarget_classic_control/analysis.ipynb.
_BEST = {
    ("dqn_hl", "cartpole"): (200, 1.0, 50, 4.0 ** -6),
    ("dqn_hl", "mountaincar"): (200, 1.5, 100, 4.0 ** -5),
    ("dqn_hl_ln_notarget", "cartpole"): (100, 2.5, 200, 4.0 ** -6),
    ("dqn_hl_ln_notarget", "mountaincar"): (100, 2.5, 100, 4.0 ** -6),
}

ENVIRONMENTS = ["cartpole", "mountaincar"]
_SEEDS = list(range(10, 110))


def _histogram_component(variant: str, env: str) -> Component:
    """``variant`` on ``env`` at its best grid point from the sweep."""
    support, sigma, bins, lr = _BEST[variant, env]
    config = _SWEEP.component(f"{variant}_{env}_support{support}").config
    hypers = dataclasses.replace(
        config.AGENT_HYPERS, SIGMA_RATIO=sigma, NUM_BINS=bins, LR=lr
    )
    return Component(
        name=f"{variant}_{env}",
        config=dataclasses.replace(config, AGENT_HYPERS=hypers),
        seeds=_SEEDS,
        shard_size=10,
    )


def _baseline_component(reference: Experiment, name: str) -> Component:
    """``reference``'s component ``name``, rerun on this experiment's seeds."""
    return Component(
        name=name,
        config=reference.component(name).config,
        seeds=_SEEDS,
        shard_size=10,
    )


EXPERIMENT = Experiment(
    name="histogram_tuned_classic_control",
    results_dir=Path(__file__).resolve().parent / "results",
    slurm=SlurmResources(time="02:59:00"),
    components=[
        component
        for env in ENVIRONMENTS
        for component in (
            _baseline_component(_TUNED, f"dqn_{env}"),
            _baseline_component(_BENCHMARKING, f"agent0_{env}"),
            *(_histogram_component(variant, env) for variant in ("dqn_hl", "dqn_hl_ln_notarget")),
        )
    ],
)
