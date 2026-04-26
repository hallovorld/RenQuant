"""Tests for kernel.portfolio_qp.signal_combiner — Treynor-Black 1973.

Inverse-variance weighted combination of multiple signal sources.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.signal_combiner import combine_signals  # noqa: E402


class TestCombineSignals:
    def test_empty_returns_empty(self):
        c, w = combine_signals({})
        assert c.shape == (0,) and w == {}

    def test_single_signal_unchanged(self):
        x = np.array([0.05, -0.02, 0.10])
        c, w = combine_signals({"src": x})
        np.testing.assert_allclose(c, x)
        assert w == {"src": 1.0}

    def test_equal_ic_equal_weights(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        c, w = combine_signals(
            {"A": a, "B": b},
            ic_means={"A": 0.05, "B": 0.05},
            ic_stds={"A": 0.10, "B": 0.10},
        )
        np.testing.assert_allclose(c, [0.5, 0.5])
        assert w["A"] == pytest.approx(0.5)
        assert w["B"] == pytest.approx(0.5)

    def test_higher_ir_higher_weight(self):
        """Source with better IR² gets more weight."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        c, w = combine_signals(
            {"A": a, "B": b},
            ic_means={"A": 0.10, "B": 0.05},     # A's IC twice B's
            ic_stds={"A": 0.10, "B": 0.10},      # same noise
        )
        # IR_A² = 1, IR_B² = 0.25 → w_A = 1/1.25 = 0.8, w_B = 0.2
        assert w["A"] == pytest.approx(0.8, rel=1e-6)
        assert w["B"] == pytest.approx(0.2, rel=1e-6)

    def test_zero_ir_falls_back_to_equal_weights(self):
        """All zero IC → equal weight fallback (graceful)."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        c, w = combine_signals(
            {"A": a, "B": b},
            ic_means={"A": 0.0, "B": 0.0},
            ic_stds={"A": 1.0, "B": 1.0},
        )
        assert w["A"] == pytest.approx(0.5)
        assert w["B"] == pytest.approx(0.5)

    def test_nan_signal_treated_as_zero(self):
        a = np.array([0.05, float("nan"), 0.10])
        c, _ = combine_signals({"A": a})
        np.testing.assert_allclose(c, [0.05, 0.0, 0.10])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            combine_signals({
                "A": np.array([1.0, 2.0]),
                "B": np.array([1.0]),
            })

    def test_three_source_combination(self):
        """Realistic case: per-ticker + panel + NGBoost μ̂ vectors."""
        per_ticker = np.array([0.05,  0.02, -0.01])
        panel       = np.array([0.03,  0.01, -0.02])
        ngboost     = np.array([0.02,  0.00, -0.03])
        c, w = combine_signals(
            {"per_ticker": per_ticker, "panel": panel, "ngboost": ngboost},
            ic_means={"per_ticker": 0.06, "panel": 0.05, "ngboost": 0.04},
            ic_stds={"per_ticker": 0.03, "panel": 0.03, "ngboost": 0.03},
        )
        # All weights positive; sum to 1
        assert sum(w.values()) == pytest.approx(1.0)
        assert w["per_ticker"] > w["ngboost"]
        # Combined mu should retain sign of input (all three agree direction)
        assert (c * np.sign(per_ticker) >= 0).all()

    def test_returns_zero_when_only_nans(self):
        a = np.array([float("nan"), float("nan")])
        c, _ = combine_signals({"A": a})
        np.testing.assert_allclose(c, [0.0, 0.0])


# ── SC-NEG-IC audit fix ────────────────────────────────────────────────────────

class TestNegativeICFlipping:
    """Audit fix SC-NEG-IC (2026-04-26): a source with IC<0 is BIASED
    (signal points wrong direction). Pre-fix, IR² dropped the sign and
    propagated the wrong direction. Post-fix, we flip the source vector
    when its IC is negative."""

    def test_negative_ic_signal_flipped(self):
        """A -0.05 IC source should contribute with reversed sign."""
        a = np.array([1.0, -1.0, 0.5])
        c, w = combine_signals(
            {"NEG": a},
            ic_means={"NEG": -0.05},
            ic_stds={"NEG": 0.10},
        )
        # Single source → weight 1.0; flipped because IC<0
        np.testing.assert_allclose(c, -a)

    def test_pos_and_neg_ic_blend_with_flip(self):
        """Positive and negative IC sources both contribute correctly."""
        pos = np.array([1.0, -1.0])
        neg = np.array([-1.0, 1.0])    # SAME info as `pos` but flipped
        # If both have IC magnitudes equal, the two should reinforce
        # (not cancel) after sign-flip on the negative-IC source.
        c, w = combine_signals(
            {"POS": pos, "NEG": neg},
            ic_means={"POS": 0.05, "NEG": -0.05},
            ic_stds={"POS": 0.10, "NEG": 0.10},
        )
        # weights both ≈ 0.5; combined = 0.5*pos + 0.5*(-1)*neg = pos
        np.testing.assert_allclose(c, pos)

    def test_zero_ic_source_keeps_sign_one(self):
        """IC=0 → sign treated as +1 (no flip)."""
        a = np.array([1.0, -1.0])
        c, w = combine_signals(
            {"ZERO": a},
            ic_means={"ZERO": 0.0},
        )
        # IC=0 → IR²=0 → fallback to equal weight (1.0); no sign flip
        np.testing.assert_allclose(c, a)
