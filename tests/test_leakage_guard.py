"""Tests for kernel.walk_forward.leakage_guard.

Track P2 (2026-05-10) — single-source-of-truth leakage helper used by
both legacy SimAdapter load and walk-forward per-bar lookup. Per
CLAUDE.md §5.13.5: there is exactly one function for this decision.
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

_STRATEGY_DIR = (
    Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
)
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


class TestLeakageGuardBasic:
    def test_strict_inequality_raises_on_equal(self):
        """Equal dates count as leakage — fwd-label horizon means a model
        trained ON `today` has already seen `today`'s features, so any
        sim bar at exactly that date risks contamination."""
        from kernel.walk_forward.leakage_guard import assert_no_leakage
        with pytest.raises(ValueError, match="leakage"):
            assert_no_leakage("2026-05-09", "2026-05-09")

    def test_trained_after_today_raises(self):
        from kernel.walk_forward.leakage_guard import assert_no_leakage
        with pytest.raises(ValueError, match="leakage"):
            assert_no_leakage("2026-05-09", "2024-01-15")

    def test_trained_before_today_passes(self):
        from kernel.walk_forward.leakage_guard import assert_no_leakage
        # Should not raise.
        assert_no_leakage("2024-01-01", "2024-01-02")

    def test_message_includes_context_label(self):
        from kernel.walk_forward.leakage_guard import assert_no_leakage
        with pytest.raises(ValueError, match=r"my-context"):
            assert_no_leakage("2026-05-09", "2024-06-01",
                              context="my-context")


class TestLeakageGuardTypeCoercion:
    def test_accepts_pd_timestamp(self):
        from kernel.walk_forward.leakage_guard import assert_no_leakage
        assert_no_leakage(pd.Timestamp("2024-01-01"),
                          pd.Timestamp("2024-01-02"))

    def test_accepts_datetime(self):
        from kernel.walk_forward.leakage_guard import assert_no_leakage
        assert_no_leakage(datetime(2024, 1, 1), datetime(2024, 1, 2))

    def test_accepts_date(self):
        from kernel.walk_forward.leakage_guard import assert_no_leakage
        assert_no_leakage(date(2024, 1, 1), date(2024, 1, 2))

    def test_rejects_none(self):
        from kernel.walk_forward.leakage_guard import assert_no_leakage
        with pytest.raises(TypeError, match="None"):
            assert_no_leakage(None, "2024-01-02")
        with pytest.raises(TypeError, match="None"):
            assert_no_leakage("2024-01-01", None)


class TestLeakageGuardRegression:
    """AUDIT REGRESSION GUARD (CLAUDE.md §5.13.3).

    Pin the invariant: the 2026-05-10 audit class — model trained
    2026-05-09 used in a sim covering 2024-01 → 2026-03 — must always
    raise. If anyone removes / weakens this guard, this test fails
    LOUD before the regression hits production.

    Per §5.13.5 (single source of truth): both legacy sim load AND
    walk-forward per-bar lookup must call assert_no_leakage. This
    test only verifies the guard itself; the integration tests in
    test_sim_walkforward.py verify the call sites.
    """
    def test_audit_2026_05_10_class_blocks(self):
        """The exact scenario that triggered Track P2."""
        from kernel.walk_forward.leakage_guard import assert_no_leakage
        # Prod model trained 2026-05-09; sim claims to evaluate "as of"
        # any date inside 2024-01-01 → 2026-03-31 — that's the leakage.
        for sim_today in ["2024-01-15", "2024-06-01", "2025-09-30",
                          "2026-03-31"]:
            with pytest.raises(ValueError, match="leakage"):
                assert_no_leakage("2026-05-09", sim_today,
                                  context=f"audit-2026-05-10 sim={sim_today}")

    def test_walkforward_correct_pattern_passes(self):
        """The correct walk-forward pattern: model trained at cutoff,
        sim evaluates strictly AFTER the cutoff. No leakage."""
        from kernel.walk_forward.leakage_guard import assert_no_leakage
        # Cutoff date = 2024-01-01 → model "trained_date" stamped that
        # day → walk-forward sim evaluates 2024-Q2, Q3, Q4 etc. All
        # must pass.
        for sim_today in ["2024-04-01", "2024-07-01", "2024-10-01",
                          "2025-01-01"]:
            # Should not raise.
            assert_no_leakage("2024-01-01", sim_today)
