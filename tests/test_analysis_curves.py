"""Tests for one run's signals over time (``analysis.curves``)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from analysis.curves import ema, lifetime_average, return_curve

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
