"""Gârleanu-Pedersen ProportionalTradeToTargets — unit tests.

Pins the partial-rebalance arithmetic:
  * N=1 → all-or-nothing (legacy parity)
  * N>1 → fractional move toward target
  * Per-regime horizon resolution per PRIME DIRECTIVE

Anchors against the canonical reference impl in cvxportfolio.
"""
from __future__ import annotations

import numpy as np
import pytest

from backtesting.renquant_104.kernel.portfolio_qp.proportional_trade import (
    proportional_trade_target,
    resolve_trade_horizon_days,
)


def test_n_equals_one_is_all_or_nothing():
    """N=1 = current behavior: partial target IS the QP target."""
    current = np.array([0.1, 0.2, 0.3])
    target = np.array([0.0, 0.5, 0.1])
    partial = proportional_trade_target(current_w=current, target_w=target, n_days=1)
    np.testing.assert_array_almost_equal(partial, target)


def test_n_equals_two_is_half_move():
    """N=2 = move halfway toward target this bar."""
    current = np.array([0.0])
    target = np.array([0.1])
    partial = proportional_trade_target(current_w=current, target_w=target, n_days=2)
    np.testing.assert_array_almost_equal(partial, [0.05])


def test_n_equals_twenty_is_5_percent_move():
    """N=20 = move 5% of the gap this bar."""
    current = np.array([0.05])
    target = np.array([0.15])
    partial = proportional_trade_target(current_w=current, target_w=target, n_days=20)
    # gap = 0.10; move 0.10/20 = 0.005; new = 0.055
    np.testing.assert_array_almost_equal(partial, [0.055])


def test_large_n_means_tiny_move():
    """N → ∞ means current_w unchanged."""
    current = np.array([0.1, 0.2])
    target = np.array([0.0, 0.5])
    partial = proportional_trade_target(current_w=current, target_w=target, n_days=10000)
    # gap collapses; new ≈ current
    np.testing.assert_array_almost_equal(partial, current, decimal=4)


def test_n_zero_or_negative_coerced_to_one():
    """Defensive: bogus N defaults to 1 (= immediate move)."""
    current = np.array([0.1])
    target = np.array([0.2])
    p0 = proportional_trade_target(current_w=current, target_w=target, n_days=0)
    p_neg = proportional_trade_target(current_w=current, target_w=target, n_days=-5)
    np.testing.assert_array_almost_equal(p0, target)
    np.testing.assert_array_almost_equal(p_neg, target)


def test_shape_mismatch_raises():
    current = np.array([0.1, 0.2])
    target = np.array([0.0, 0.5, 0.3])
    with pytest.raises(ValueError, match="must match"):
        proportional_trade_target(current_w=current, target_w=target, n_days=5)


def test_does_not_mutate_inputs():
    """Pure: never modify the input arrays."""
    current = np.array([0.1, 0.2, 0.3])
    target = np.array([0.0, 0.5, 0.1])
    current_copy = current.copy()
    target_copy = target.copy()
    proportional_trade_target(current_w=current, target_w=target, n_days=5)
    np.testing.assert_array_equal(current, current_copy)
    np.testing.assert_array_equal(target, target_copy)


def test_resolve_horizon_picks_regime_value():
    """Per-regime override (PRIME DIRECTIVE)."""
    regime_params = {
        "BULL_CALM": {"qp_partial_trade_horizon_days": 20},
        "BEAR": {"qp_partial_trade_horizon_days": 3},
    }
    assert resolve_trade_horizon_days(regime="BULL_CALM", regime_params=regime_params,
                                       default_days=10) == 20
    assert resolve_trade_horizon_days(regime="BEAR", regime_params=regime_params,
                                       default_days=10) == 3


def test_resolve_horizon_falls_back_to_default():
    """Regime not in config → default."""
    assert resolve_trade_horizon_days(
        regime="CHOPPY",
        regime_params={"BULL_CALM": {"qp_partial_trade_horizon_days": 20}},
        default_days=10,
    ) == 10


def test_resolve_horizon_returns_one_when_nothing_configured():
    """No regime, no default → 1 (legacy all-or-nothing)."""
    assert resolve_trade_horizon_days(
        regime=None, regime_params={}, default_days=None,
    ) == 1.0


def test_resolve_horizon_handles_missing_knob():
    """Regime present but knob missing → default."""
    assert resolve_trade_horizon_days(
        regime="BULL_CALM",
        regime_params={"BULL_CALM": {"other_knob": 0.05}},
        default_days=5,
    ) == 5


def test_meta_scenario_with_partial_rebalance():
    """The 2026-05-30 META scenario, but with N=5 partial-trade.

    Legacy: current=0.057, QP target=0.0194 → Δw=-0.0376 (skipped by 5% band).
    Partial N=5: partial_target = 0.057 + (0.0194 - 0.057)/5 = 0.0495.
    Δw_partial = -0.0075 (smaller — still might skip the legacy band).
    Partial N=2: partial = 0.0382. Δw_partial = -0.0188.

    The key insight: partial rebalance + DN band (≈1%) means SMALLER but
    EXECUTABLE trades. Pure all-or-nothing would skip. This is the GP-2013
    optimum.
    """
    current = np.array([0.057])
    target = np.array([0.0194])

    p5 = proportional_trade_target(current_w=current, target_w=target, n_days=5)
    np.testing.assert_array_almost_equal(p5, [0.0495], decimal=4)

    p2 = proportional_trade_target(current_w=current, target_w=target, n_days=2)
    np.testing.assert_array_almost_equal(p2, [0.0382], decimal=4)
