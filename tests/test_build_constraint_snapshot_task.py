"""Tests for BuildConstraintSnapshotTask — §8 Step 1c wiring.

The Task is strictly additive: it runs after the existing 4-Task
constraint-composition pipeline (Compute / ApplyExposureScaling /
ApplyConvictionCap / sector+corr) and freezes the assembled state
into ``ctx._qp_constraint_snapshot``. Downstream allocators consume
the contract; existing Tasks unaffected.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.constraint_snapshot import ConstraintSnapshot  # noqa: E402
from kernel.portfolio_qp.tasks import BuildConstraintSnapshotTask  # noqa: E402


def _ctx(**fields):
    ctx = SimpleNamespace()
    for k, v in fields.items():
        setattr(ctx, k, v)
    return ctx


class TestBuildConstraintSnapshotTaskNominal:
    """Healthy ctx → snapshot stamped + Task returns None (= continue)."""

    def test_stamps_snapshot_on_valid_ctx(self):
        ctx = _ctx(
            _qp_tickers=["AAPL", "MSFT"],
            _qp_w_current=np.array([0.10, 0.05]),
            _qp_w_upper_hard=np.array([0.20, 0.20]),
            _qp_w_upper=np.array([0.15, 0.20]),
            _qp_w_lower=0.0,
            _qp_dw_max=np.full(2, 0.5),
            _qp_cash_reserve=0.05,
            _qp_turnover_max=0.30,
            _qp_drawdown=0.0,
            _qp_drawdown_limit=0.20,
            _qp_wash_mask=np.array([False, True]),
            regime="BULL_CALM",
            confidence=0.7,
        )

        rv = BuildConstraintSnapshotTask().run(ctx)

        assert rv is None, "healthy ctx must not short-circuit"
        snap = ctx._qp_constraint_snapshot
        assert isinstance(snap, ConstraintSnapshot)
        assert snap.tickers == ("AAPL", "MSFT")
        np.testing.assert_array_equal(snap.w_upper_hard, [0.20, 0.20])
        np.testing.assert_array_equal(snap.w_upper, [0.15, 0.20])
        assert snap.regime == "BULL_CALM"
        # No error diagnostic on the healthy path
        assert getattr(ctx, "_qp_constraint_snapshot_error", None) is None

    def test_empty_universe_stamps_none_and_continues(self):
        """No tickers → no constraints to snapshot, no failure either."""
        ctx = _ctx(_qp_tickers=[])
        rv = BuildConstraintSnapshotTask().run(ctx)
        assert rv is None
        assert ctx._qp_constraint_snapshot is None


class TestBuildConstraintSnapshotTaskFailLoud:
    """Constraint contract violations short-circuit the Job."""

    def test_soft_above_hard_fails_loud(self, caplog):
        """The #123 v1/v2/v3 bug class — soft cap > hard cap. The Task
        catches the constructor ValueError, logs it, stamps the
        diagnostic on ctx, and returns False so SolveMarkowitzQPTask
        does not run on the contradictory state.
        """
        ctx = _ctx(
            _qp_tickers=["ORCL"],
            _qp_w_current=np.array([0.0]),
            _qp_w_upper_hard=np.array([0.15]),
            _qp_w_upper=np.array([0.22]),   # CONTRADICTION
            _qp_w_lower=0.0,
            _qp_dw_max=np.array([0.5]),
            _qp_cash_reserve=0.0,
            _qp_drawdown=0.0,
            _qp_drawdown_limit=0.20,
            _qp_wash_mask=np.array([False]),
        )

        with caplog.at_level(logging.ERROR):
            rv = BuildConstraintSnapshotTask().run(ctx)

        assert rv is False, "contradictory state must short-circuit"
        assert ctx._qp_constraint_snapshot is None
        assert ctx._qp_constraint_snapshot_error
        assert "soft cap exceeds hard cap" in ctx._qp_constraint_snapshot_error
        # The error was also logged so the daily run trace surfaces it
        assert any(
            "BuildConstraintSnapshotTask" in rec.message
            for rec in caplog.records
        )

    def test_shape_mismatch_fails_loud(self):
        ctx = _ctx(
            _qp_tickers=["A", "B"],
            _qp_w_current=np.array([0.0, 0.0]),
            _qp_w_upper_hard=np.array([0.20, 0.20, 0.20]),   # wrong n
            _qp_w_upper=np.array([0.20, 0.20]),
            _qp_w_lower=0.0,
            _qp_dw_max=np.array([0.5, 0.5]),
            _qp_cash_reserve=0.0,
            _qp_drawdown=0.0,
            _qp_drawdown_limit=0.20,
            _qp_wash_mask=np.array([False, False]),
        )

        rv = BuildConstraintSnapshotTask().run(ctx)
        assert rv is False
        assert ctx._qp_constraint_snapshot is None
        assert "shape" in ctx._qp_constraint_snapshot_error

    def test_failure_path_stamps_qp_attribution_fields(self):
        """**Codex #129 review regression guard.** The fail-loud branch
        is a first-class QP failure path — it MUST stamp the standard
        QP attribution fields (``_qp_status``, ``_qp_failure_reason``,
        zero buys/sells, the all-tickers-blocked map, the idempotent
        failure counter) the way every other early-QP-failure path
        does (e.g. ``ComputeFullSigmaTask._fail_full_sigma``).
        Otherwise ``live.runner._why_no_trade`` can fall through to a
        stale or missing reason on the snapshot-invalid path.
        """
        ctx = _ctx(
            _qp_tickers=["A", "B"],
            _qp_w_current=np.array([0.0, 0.0]),
            _qp_w_upper_hard=np.array([0.15, 0.15]),
            _qp_w_upper=np.array([0.22, 0.15]),  # row 0 violates soft<=hard
            _qp_w_lower=0.0,
            _qp_dw_max=np.array([0.5, 0.5]),
            _qp_cash_reserve=0.0,
            _qp_drawdown=0.0,
            _qp_drawdown_limit=0.20,
            _qp_wash_mask=np.array([False, False]),
            # No prior _qp_status / _qp_failure_reason / counters —
            # this is the failure-attribution gap codex flagged.
            counters={},
        )

        rv = BuildConstraintSnapshotTask().run(ctx)

        assert rv is False
        # Standard QP failure attribution stamped
        assert ctx._qp_status == "infeasible:qp_constraint_snapshot_invalid"
        assert ctx._qp_failure_reason == "qp_constraint_snapshot_invalid"
        assert ctx._qp_n_buys == 0
        assert ctx._qp_n_sells == 0
        # Every QP ticker blocked with the constraint-snapshot reason
        blocked = getattr(ctx, "_blocked_by_ticker", {}) or {}
        for t in ("A", "B"):
            assert blocked.get(t) == "qp_constraint_snapshot_invalid", (
                f"ticker {t} not blocked with the snapshot-invalid reason; "
                f"blocked_map={blocked}"
            )
        # Failure counter incremented (idempotent). The shared helper
        # maps any "infeasible*" status to the canonical ``qp_infeasible``
        # key per its docstring, so live.runner._why_no_trade() can
        # surface the constraint-snapshot failure via the same code
        # path as ComputeFullSigmaTask / SolveMarkowitzQPTask failures.
        counters = getattr(ctx, "counters", {}) or {}
        assert counters.get("qp_infeasible", 0) >= 1, (
            f"QP failure counter not stamped; counters={counters}"
        )


class TestBuildConstraintSnapshotTaskAdditivity:
    """Building the snapshot must NOT mutate any pre-existing ctx state."""

    def test_existing_qp_arrays_remain_writable_after_build(self):
        """Codex #126 review applies to the Task too — building the
        snapshot must not freeze the caller's ctx arrays.
        """
        ctx = _ctx(
            _qp_tickers=["A", "B"],
            _qp_w_current=np.array([0.10, 0.05]),
            _qp_w_upper_hard=np.array([0.20, 0.20]),
            _qp_w_upper=np.array([0.15, 0.20]),
            _qp_w_lower=0.0,
            _qp_dw_max=np.full(2, 0.5),
            _qp_cash_reserve=0.05,
            _qp_turnover_max=0.30,
            _qp_drawdown=0.0,
            _qp_drawdown_limit=0.20,
            _qp_wash_mask=np.array([False, False]),
        )
        BuildConstraintSnapshotTask().run(ctx)

        # All ctx arrays must remain writable (caller-owned)
        ctx._qp_w_upper[0] = 0.10
        ctx._qp_w_current[1] = 0.99
        ctx._qp_w_upper_hard[0] = 0.30
        ctx._qp_dw_max[0] = 0.40
        ctx._qp_wash_mask[1] = True
        # And mutating the ctx must NOT leak into the snapshot
        assert float(ctx._qp_constraint_snapshot.w_upper[0]) == 0.15

    def test_does_not_overwrite_existing_qp_fields(self):
        """The Task only writes ctx._qp_constraint_snapshot[_error]; it
        must not touch any other ctx._qp_* fields the existing Tasks
        produced.
        """
        ctx = _ctx(
            _qp_tickers=["A"],
            _qp_w_current=np.array([0.0]),
            _qp_w_upper_hard=np.array([0.20]),
            _qp_w_upper=np.array([0.20]),
            _qp_w_lower=0.0,
            _qp_dw_max=np.array([0.5]),
            _qp_cash_reserve=0.0,
            _qp_drawdown=0.0,
            _qp_drawdown_limit=0.20,
            _qp_wash_mask=np.array([False]),
            _qp_turnover_max=0.30,
            _qp_status="prior_value",          # set by an unrelated Task
            _qp_failure_reason="prior_reason", # set by an unrelated Task
        )
        BuildConstraintSnapshotTask().run(ctx)
        # Unrelated fields untouched
        assert ctx._qp_status == "prior_value"
        assert ctx._qp_failure_reason == "prior_reason"
