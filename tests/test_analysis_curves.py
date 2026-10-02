"""Tests for one run's signals over time (``analysis.curves``)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from analysis.curves import (
    ema,
    episode_average,
    lifetime_average,
    return_curve,
    subsample,
)

# --- ema ----------------------------------------------------------------------


def test_ema_first_value_is_unsmoothed():
    """There is nothing to average the first reward against yet."""
    assert ema([5.0, 1.0, 1.0], beta=0.5)[0] == pytest.approx(5.0)


def test_ema_matches_manual_recursion():
    """Each value blends the previous average with the new reward by ``beta``."""
    reward = [1.0, 0.0, 1.0, 0.0]
    expected = [1.0, 0.5, 0.75, 0.375]
    assert np.allclose(ema(reward, beta=0.5), expected)


def test_ema_empty():
    """An empty reward stream smooths to an empty one."""
    assert ema([]).size == 0


def test_ema_smooths_each_run_of_a_stack_on_its_own():
    """Runs are rows; smoothing a stack equals smoothing each row alone."""
    stack = np.random.default_rng(0).normal(size=(3, 50))
    smoothed = ema(stack, beta=0.9)
    assert smoothed.shape == stack.shape
    assert all(np.allclose(smoothed[i], ema(stack[i], beta=0.9)) for i in range(3))


def test_ema_with_zero_beta_is_the_reward_itself():
    """No weight on the past leaves every value unsmoothed."""
    reward = np.random.default_rng(0).normal(size=20)
    assert np.allclose(ema(reward, beta=0.0), reward)


# --- return_curve -------------------------------------------------------------


def test_return_curve_holds_each_episode_return_over_its_steps():
    """Every step of an episode reports the return of that whole episode."""
    curve = return_curve([1.0, 1.0, 1.0, 2.0, 2.0], [0, 0, 1, 0, 1])
    assert np.array_equal(curve, [3.0, 3.0, 3.0, 4.0, 4.0])


def test_return_curve_is_nan_after_the_last_completed_episode():
    """An episode cut off by the end of the run has no return yet."""
    curve = return_curve([1.0, 1.0, 1.0, 1.0], [0, 1, 0, 0])
    assert np.array_equal(curve, [2.0, 2.0, np.nan, np.nan], equal_nan=True)


def test_return_curve_is_all_nan_without_a_completed_episode():
    """A run that never finished an episode has no return anywhere."""
    assert np.isnan(return_curve(np.ones(4), np.zeros(4))).all()


def test_return_curve_handles_one_step_episodes():
    """An episode ending on its first step, including the run's first step."""
    curve = return_curve([5.0, -1.0, 1.0], [1, 1, 1])
    assert np.array_equal(curve, [5.0, -1.0, 1.0])


def test_return_curve_of_a_stack_matches_each_run_alone():
    """Runs are rows, and each row's episodes are found independently."""
    rng = np.random.default_rng(0)
    reward = rng.normal(size=(4, 30))
    done = rng.random((4, 30)) < 0.2
    curve = return_curve(reward, done)
    assert curve.shape == reward.shape
    for row in range(4):
        expected = return_curve(reward[row], done[row])
        assert np.array_equal(curve[row], expected, equal_nan=True)


# --- lifetime_average ---------------------------------------------------------


def _episode_weighted_return(reward, done):
    weighted, steps, episode_return, episode_length = 0.0, 0, 0.0, 0
    for step_reward, step_done in zip(reward, done, strict=True):
        episode_return += step_reward
        episode_length += 1
        if step_done:
            weighted += episode_return * episode_length
            steps += episode_length
            episode_return, episode_length = 0.0, 0
    return weighted / steps if steps else np.nan


def test_lifetime_average_weights_episodes_by_length():
    """A 3-step episode returning 3 outweighs a 1-step episode returning 7."""
    reward = [1.0, 1.0, 1.0, 7.0]
    done = [0, 0, 1, 1]
    assert np.isclose(lifetime_average(return_curve(reward, done)), (9 + 7) / 4)


@settings(derandomize=True)
@given(
    st.lists(
        st.tuples(st.floats(-10, 10), st.booleans()), min_size=1, max_size=40
    )
)
def test_lifetime_average_of_a_return_curve_is_the_episode_weighted_return(steps):
    """For any run, it equals sum(return * length) / sum(length) over the
    completed episodes, ignoring an unfinished last one."""
    reward, done = zip(*steps, strict=True)
    expected = _episode_weighted_return(reward, done)
    actual = lifetime_average(return_curve(reward, done))
    assert np.isclose(actual, expected, equal_nan=True)


def test_lifetime_average_of_reward_is_the_average_reward():
    """On raw reward, as for a continuing task, it is the mean reward rate."""
    reward = np.random.default_rng(0).normal(size=(3, 20))
    assert np.allclose(lifetime_average(reward), reward.mean(axis=-1))


def test_lifetime_average_is_nan_for_a_run_with_no_defined_step():
    """A run that finished no episode has no lifetime return, next to one that
    did."""
    curve = return_curve([[1.0, 1.0], [1.0, 1.0]], [[0, 0], [0, 1]])
    averages = lifetime_average(curve)
    assert np.isnan(averages[0])
    assert averages[1] == 2.0


# --- episode_average ----------------------------------------------------------


def _completed_episode_returns(reward, done):
    returns, episode_return = [], 0.0
    for step_reward, step_done in zip(reward, done, strict=True):
        episode_return += step_reward
        if step_done:
            returns.append(episode_return)
            episode_return = 0.0
    return returns


def test_episode_average_weights_every_episode_equally():
    """A 3-step episode returning 3 and a 1-step one returning 7 average to 5,
    where the step-weighted average gives 4."""
    assert np.isclose(episode_average([1.0, 1.0, 1.0, 7.0], [0, 0, 1, 1]), 5.0)


def test_episode_average_ignores_an_unfinished_last_episode():
    """Steps after the last completed episode have no return to count."""
    assert np.isclose(episode_average([2.0, 4.0, 100.0], [0, 1, 0]), 6.0)


def test_episode_average_is_nan_without_a_completed_episode():
    """A run that never finished an episode has no average."""
    assert np.isnan(episode_average(np.ones(4), np.zeros(4)))


def test_episode_average_of_a_stack_matches_each_run_alone():
    """Runs are rows, and the result has one value per run."""
    rng = np.random.default_rng(0)
    reward = rng.normal(size=(4, 30))
    done = rng.random((4, 30)) < 0.2
    average = episode_average(reward, done)
    assert average.shape == (4,)
    for row in range(4):
        expected = episode_average(reward[row], done[row])
        assert np.isclose(average[row], expected, equal_nan=True)


@settings(derandomize=True)
@given(
    st.lists(
        st.tuples(st.floats(-10, 10), st.booleans()), min_size=1, max_size=40
    )
)
def test_episode_average_is_the_mean_completed_episode_return(steps):
    """For any run, it equals the plain mean of its completed episodes' returns."""
    reward, done = zip(*steps, strict=True)
    returns = _completed_episode_returns(reward, done)
    expected = np.mean(returns) if returns else np.nan
    assert np.isclose(episode_average(reward, done), expected, equal_nan=True)


# --- subsample ----------------------------------------------------------------


def test_subsample_spans_the_whole_run():
    """The first and last timesteps are always kept, counted from 1."""
    timesteps, values = subsample(np.arange(1000.0), points=7)
    assert timesteps[0] == 1
    assert timesteps[-1] == 1000
    assert np.array_equal(values, timesteps - 1)


def test_subsample_keeps_every_step_of_a_short_run():
    """Asking for more points than timesteps returns each timestep once."""
    timesteps, values = subsample([4.0, 5.0, 6.0], points=500)
    assert np.array_equal(timesteps, [1, 2, 3])
    assert np.array_equal(values, [4.0, 5.0, 6.0])


def test_subsample_picks_the_same_timesteps_for_every_run():
    """Runs are rows, so a stack shares one set of timesteps."""
    stack = np.arange(20.0).reshape(2, 10)
    timesteps, values = subsample(stack, points=4)
    assert values.shape == (2, len(timesteps))
    assert np.array_equal(values[1] - values[0], np.full(len(timesteps), 10.0))
