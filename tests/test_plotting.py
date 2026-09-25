"""Tests for the analysis-notebook plotting helpers (``analysis.plotting``).

Builds tiny fake stores with the real harness (a planned sweep run through
``run_shards``) so the sweep-column plumbing (``AGENT_HYPERS.LR`` flattening
etc.) is exercised for real, without pulling in jax agents/envs.
"""

from __future__ import annotations

import dataclasses

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiment.design import Component, Experiment
from experiment.plan import plan_experiment
from experiment.results import load_runs
from experiment.runner import run_shards

from analysis.plotting import (
    average_lifetime_reward,
    average_lifetime_reward_stack,
    bootstrap_mean_ci,
    ema_reward,
    ema_reward_grids_for,
    episode_lengths,
    episode_returns,
    interp_on_grid,
    mean_over_seeds,
    min_max_normalize,
    plot_mean_ci,
    seed_grids_for,
    style,
    weighted_lifetime_return,
    weighted_lifetime_return_stack,
)

T = 6  # curve length used by the fake agent


@dataclasses.dataclass(frozen=True)
class Hypers:
    LR: float = 0.1


@dataclasses.dataclass(frozen=True)
class Cfg:
    HYPERS: Hypers = dataclasses.field(default_factory=Hypers)


def fake_process(configs, seeds):
    """Two episodes per run (length 2, then 4), reward == LR every step."""
    return [
        {
            "reward": np.full((T,), config.HYPERS.LR),
            "done": np.array([0, 1, 0, 0, 0, 1], dtype=np.float32),
        }
        for config in configs
    ]


def swept(results_dir, lrs=(0.1, 0.2), seeds=(0, 1)) -> Experiment:
    """An experiment with one component sweeping LR, already run."""
    experiment = Experiment(
        name="toy",
        results_dir=results_dir,
        components=[
            Component(
                name="comp",
                config=Cfg(),
                sweep={"HYPERS.LR": list(lrs)},
                seeds=list(seeds),
                shard_size=len(seeds),
            )
        ],
    )
    run_shards(experiment, plan_experiment(experiment), fake_process)
    return experiment


# --- episode_lengths / weighted_lifetime_return -----------------------------


def test_episode_lengths():
    # ends are inclusive 0-indexed positions: episode 0 spans indices 0-1 (len
    # 2), episode 1 spans 2-5 (len 4), episode 2 is just index 6 (len 1)
    assert list(episode_lengths(np.array([1, 5, 6]))) == [2, 4, 1]


def test_episode_lengths_span_every_timestep():
    """Every timestep belongs to exactly one episode - the environments
    autoreset in place, so no step sits between two of them."""
    ends = np.array([2, 5, 10])
    lengths = episode_lengths(ends)
    np.testing.assert_array_equal(lengths, [3, 3, 5])
    assert lengths.sum() == ends[-1] + 1


def test_weighted_lifetime_return_weights_by_length():
    # episode 1: length 2, return 2*0.1; episode 2: length 4, return 4*0.2
    reward = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]
    done = [0, 1, 0, 0, 0, 1]
    got = weighted_lifetime_return(reward, done)
    assert got == pytest.approx((0.2 * 2 + 0.8 * 4) / 6)


def test_weighted_lifetime_return_nan_with_no_completed_episode():
    reward, done = [0.1] * 4, [0] * 4
    assert np.isnan(weighted_lifetime_return(reward, done))


# --- average_lifetime_reward -------------------------------------------------


def test_average_lifetime_reward_is_mean_reward():
    assert average_lifetime_reward([0.0, 1.0, 1.0, -1.0]) == pytest.approx(0.25)


def test_average_lifetime_reward_ignores_episode_boundaries():
    # unlike weighted_lifetime_return, no done flag needed at all,
    # and a run with zero completed episodes is still a valid mean, not NaN
    assert average_lifetime_reward([2.0, 2.0, 2.0]) == pytest.approx(2.0)


def test_average_lifetime_reward_stack_shape_and_values(tmp_path):
    experiment = swept(tmp_path)

    stack = average_lifetime_reward_stack(experiment, "comp", "HYPERS.LR", [0.1, 0.2])
    assert stack.shape == (2, 2)  # 2 seeds x 2 LR values
    # the fake reward is constant == LR every step, so the mean is LR itself
    np.testing.assert_allclose(stack[:, 0], 0.1)
    np.testing.assert_allclose(stack[:, 1], 0.2)


# --- ema_reward ---------------------------------------------------------------


def test_ema_reward_first_value_is_unsmoothed():
    assert ema_reward([5.0, 1.0, 1.0], beta=0.5)[0] == pytest.approx(5.0)


def test_ema_reward_matches_manual_recursion():
    reward = [1.0, 0.0, 1.0, 0.0]
    expected = [1.0, 0.5, 0.75, 0.375]
    np.testing.assert_allclose(ema_reward(reward, beta=0.5), expected)


def test_ema_reward_empty():
    assert ema_reward([]).size == 0


def test_ema_reward_grids_for_shape_and_values(tmp_path):
    experiment = swept(tmp_path)

    df = load_runs(experiment, "comp")
    rids = df.filter(df["HYPERS.LR"] == 0.1)["run_id"].to_list()
    stack = ema_reward_grids_for(
        experiment, "comp", np.array([0.0, 5.0]), beta=0.5, run_ids=rids
    )
    assert stack.shape == (2, 2)
    # the fake reward is constant == LR every step, so the EMA of it is
    # the same constant regardless of beta
    np.testing.assert_allclose(stack, 0.1)


# --- seed_grids_for(run_ids=...) --------------------------------------------


def test_seed_grids_for_run_ids_filters(tmp_path):
    experiment = swept(tmp_path)

    all_grid = seed_grids_for(experiment, "comp", np.array([0.0, 6.0]))
    assert all_grid.shape == (4, 2)  # 2 LRs x 2 seeds

    df = load_runs(experiment, "comp")
    lo_rids = df.filter(df["HYPERS.LR"] == 0.1)["run_id"].to_list()
    lo_grid = seed_grids_for(experiment, "comp", np.array([0.0, 6.0]), run_ids=lo_rids)
    assert lo_grid.shape == (2, 2)


# --- weighted_lifetime_return_stack -----------------------------------------


def test_weighted_lifetime_return_stack_shape_and_values(tmp_path):
    experiment = swept(tmp_path)

    stack = weighted_lifetime_return_stack(experiment, "comp", "HYPERS.LR", [0.1, 0.2])
    assert stack.shape == (2, 2)  # 2 seeds x 2 LR values
    # episode returns are 2*LR (length 2) and 4*LR (length 4): weighted mean
    # = (2*LR*2 + 4*LR*4) / 6 = 10*LR/3
    np.testing.assert_allclose(stack[:, 0], 10 * 0.1 / 3)
    np.testing.assert_allclose(stack[:, 1], 10 * 0.2 / 3)


def test_weighted_lifetime_return_stack_pads_missing_value(tmp_path):
    # only LR=0.1 has any stored runs; LR=0.2 is entirely absent
    experiment = swept(tmp_path, lrs=(0.1,), seeds=(0,))

    stack = weighted_lifetime_return_stack(experiment, "comp", "HYPERS.LR", [0.1, 0.2])
    assert stack.shape == (1, 2)
    assert stack[0, 0] == pytest.approx(10 * 0.1 / 3)
    assert np.isnan(stack[0, 1])  # the absent value pads rather than shortens


# --- style() axis-label overrides -------------------------------------------


def test_style_default_and_override_labels():
    fig, ax = plt.subplots()
    style(ax)
    assert ax.get_xlabel() == "Timestep"
    assert ax.get_ylabel() == "Return"
    plt.close(fig)

    fig, ax = plt.subplots()
    style(ax, xlabel="Learning rate", ylabel="Weighted lifetime return")
    assert ax.get_xlabel() == "Learning rate"
    assert ax.get_ylabel() == "Weighted lifetime return"
    plt.close(fig)


# --- episode returns -> mean + bootstrap CI band ----------------------------


def _constant_return_stack(n_seeds=5):
    # one run: an episode ends every 4 steps, reward -1/step -> each return is -4
    reward = -np.ones(12)
    done = np.zeros(12)
    done[[3, 7, 11]] = 1
    ends, rets = episode_returns(reward, done)
    grid = np.linspace(3, 11, 5)  # within [ends[0], ends[-1]]: no NaN to mask
    stack = np.vstack([interp_on_grid(ends, rets, grid) for _ in range(n_seeds)])
    return grid, stack


def test_episode_returns_finds_end_and_return_per_episode():
    reward = -np.ones(12)
    done = np.zeros(12)
    done[[3, 7, 11]] = 1

    ends, rets = episode_returns(reward, done)

    assert list(ends) == [3, 7, 11]
    np.testing.assert_array_equal(rets, [-4, -4, -4])


def test_bootstrap_mean_ci_mean_matches_constant_return_across_seeds():
    _grid, stack = _constant_return_stack()

    mean, _ci_lo, _ci_hi = bootstrap_mean_ci(stack, n_boot=200)

    np.testing.assert_allclose(mean, -4)


def test_bootstrap_mean_ci_band_has_zero_width_when_seeds_agree():
    _grid, stack = _constant_return_stack()

    _mean, ci_lo, ci_hi = bootstrap_mean_ci(stack, n_boot=200)

    np.testing.assert_allclose(ci_lo, -4)
    np.testing.assert_allclose(ci_hi, -4)


def test_mean_over_seeds_matches_constant_return_across_seeds():
    _grid, stack = _constant_return_stack()

    np.testing.assert_allclose(mean_over_seeds(stack), -4)


def test_plot_mean_ci_returns_matching_mean():
    grid, stack = _constant_return_stack()
    fig, ax = plt.subplots()

    out = plot_mean_ci(ax, grid, stack, "x", "tab:blue", n_boot=200)
    plt.close(fig)

    np.testing.assert_allclose(out, -4)


# --- min_max_normalize --------------------------------------------------------


def test_min_max_normalize_shares_bounds_across_arrays():
    """The lowest value anywhere maps to 0 and the highest to 1, so arrays from
    different agents on one environment stay comparable."""
    weak, strong = min_max_normalize([[-20.0, 0.0], [10.0, 20.0]])
    assert np.allclose(weak, [0.0, 0.5])
    assert np.allclose(strong, [0.75, 1.0])


def test_min_max_normalize_keeps_stack_shape_in_unit_range():
    """A seed stack keeps its shape, and every value lands in [0, 1]."""
    stack = np.random.default_rng(0).normal(size=(5, 7)) * 100
    (normalized,) = min_max_normalize([stack])
    assert normalized.shape == stack.shape
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0


def test_min_max_normalize_ignores_nan_for_bounds():
    """NaN gaps in a curve neither set the bounds nor get filled in."""
    (normalized,) = min_max_normalize([[np.nan, 2.0, 4.0, np.nan]])
    assert np.isnan(normalized[[0, 3]]).all()
    assert np.allclose(normalized[1:3], [0.0, 1.0])
