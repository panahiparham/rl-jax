"""Tests for one run's signals over time (``analysis.curves``)."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.curves import ema

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
