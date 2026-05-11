"""AUDIT REGRESSION GUARD — A2 (2026-05-11)

ApplyConvictionCapTask wires conviction_multiplier into the QP path.
Without it, the QP gave every name the same regime+confidence cap
regardless of model conviction — issue A2 from the 2026-05-09 audit.

These tests pin the invariants:
  1. Flag OFF (default) ⇒ w_upper unchanged (no regression for prod).
  2. Flag ON + sizing OFF ⇒ no-op (defense in depth).
  3. Flag ON + sizing ON ⇒ low-conviction names get shrunk caps;
     high-conviction names keep their full cap.
  4. NaN / None panel_score ⇒ multiplier = 1.0 (no accidental cap-to-zero).
  5. The conviction-scaled cap is committed BEFORE the sector / corr
     tasks read `_qp_w_upper` (wiring order test via job_qp).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


# ── Fixtures ─────────────────────────────────────────────────────────────

def _mk_ctx(panel_scores: dict[str, float | None],
            flag_on: bool = True,
            sizing_on: bool = True,
            w_upper: float = 0.20,
            sizing_overrides: dict | None = None):
    """Build a minimal ctx for ApplyConvictionCapTask."""
    sizing_cfg = {
        "enabled": sizing_on,
        "floor":    0.0,
        "ceiling":  1.0,
        "min_mult": 0.5,
    }
    if sizing_overrides:
        sizing_cfg.update(sizing_overrides)

    tickers = list(panel_scores.keys())
    src_map = {
        t: SimpleNamespace(panel_score=ps, ticker=t)
        for t, ps in panel_scores.items()
    }

    ctx = SimpleNamespace()
    ctx.config = {
        "rotation": {"joint_actions": {
            "qp_conviction_cap_enabled": flag_on,
        }},
        "ranking": {"panel_scoring": {"sizing": sizing_cfg}},
    }
    ctx._qp_tickers        = tickers
    ctx._qp_mu_source_map  = src_map
    ctx._qp_w_upper        = np.full(len(tickers), float(w_upper))
    return ctx


# ── Invariant 1: flag OFF preserves behaviour ────────────────────────────

class TestConvictionCapRegressionGuard:
    """Pins each invariant identified in audit A2 (Issue #2)."""

    def test_flag_off_preserves_w_upper(self):
        from kernel.portfolio_qp.tasks import ApplyConvictionCapTask
        ctx = _mk_ctx({"A": 0.10, "B": 0.50, "C": 0.90}, flag_on=False)
        before = ctx._qp_w_upper.copy()
        ApplyConvictionCapTask().run(ctx)
        np.testing.assert_array_equal(ctx._qp_w_upper, before)
        assert not hasattr(ctx, "_qp_conviction_caps")

    def test_flag_on_sizing_off_is_noop(self):
        from kernel.portfolio_qp.tasks import ApplyConvictionCapTask
        ctx = _mk_ctx({"A": 0.10, "B": 0.90}, flag_on=True, sizing_on=False)
        before = ctx._qp_w_upper.copy()
        ApplyConvictionCapTask().run(ctx)
        np.testing.assert_array_equal(ctx._qp_w_upper, before)

    def test_flag_on_shrinks_low_conviction_name(self):
        """A: panel=0.10 → frac=0.10 → mult=0.5+0.10*0.5=0.55.
        B: panel=0.50 → frac=0.50 → mult=0.5+0.50*0.5=0.75.
        C: panel=0.90 → frac=0.90 → mult=0.5+0.90*0.5=0.95.
        """
        from kernel.portfolio_qp.tasks import ApplyConvictionCapTask
        ctx = _mk_ctx({"A": 0.10, "B": 0.50, "C": 0.90}, w_upper=0.20)
        ApplyConvictionCapTask().run(ctx)
        assert ctx._qp_w_upper[0] == pytest.approx(0.20 * 0.55, rel=1e-6)
        assert ctx._qp_w_upper[1] == pytest.approx(0.20 * 0.75, rel=1e-6)
        assert ctx._qp_w_upper[2] == pytest.approx(0.20 * 0.95, rel=1e-6)
        # Ordering: lower conviction → smaller cap
        assert ctx._qp_w_upper[0] < ctx._qp_w_upper[1] < ctx._qp_w_upper[2]
        # Diagnostic written
        assert ctx._qp_conviction_caps == pytest.approx([0.55, 0.75, 0.95], rel=1e-6)

    def test_high_conviction_keeps_full_cap(self):
        from kernel.portfolio_qp.tasks import ApplyConvictionCapTask
        ctx = _mk_ctx({"X": 0.99}, w_upper=0.20)
        ApplyConvictionCapTask().run(ctx)
        # frac=0.99 → mult=0.5+0.99*0.5=0.995 ≈ 1.0
        assert ctx._qp_w_upper[0] == pytest.approx(0.20 * 0.995, rel=1e-6)

    def test_above_ceiling_clamps_to_one(self):
        from kernel.portfolio_qp.tasks import ApplyConvictionCapTask
        # ceiling=0.50 → panel_score 0.80 lands above; mult=1.0
        ctx = _mk_ctx({"X": 0.80}, w_upper=0.20,
                      sizing_overrides={"ceiling": 0.50})
        ApplyConvictionCapTask().run(ctx)
        assert ctx._qp_w_upper[0] == pytest.approx(0.20, rel=1e-9)

    def test_at_or_below_floor_clamps_to_min_mult(self):
        from kernel.portfolio_qp.tasks import ApplyConvictionCapTask
        ctx = _mk_ctx({"X": -0.50}, w_upper=0.20,
                      sizing_overrides={"floor": 0.0, "min_mult": 0.3})
        ApplyConvictionCapTask().run(ctx)
        # frac<=0 → min_mult=0.3
        assert ctx._qp_w_upper[0] == pytest.approx(0.20 * 0.30, rel=1e-9)

    # ── §5.13.11 NaN / None guards ──

    def test_nan_panel_score_safe_default(self):
        from kernel.portfolio_qp.tasks import ApplyConvictionCapTask
        ctx = _mk_ctx({"X": float("nan")}, w_upper=0.20)
        ApplyConvictionCapTask().run(ctx)
        # NaN → conviction_multiplier returns 1.0; cap unchanged
        assert ctx._qp_w_upper[0] == pytest.approx(0.20, rel=1e-9)

    def test_none_panel_score_safe_default(self):
        from kernel.portfolio_qp.tasks import ApplyConvictionCapTask
        ctx = _mk_ctx({"X": None}, w_upper=0.20)
        ApplyConvictionCapTask().run(ctx)
        assert ctx._qp_w_upper[0] == pytest.approx(0.20, rel=1e-9)

    def test_missing_source_object_safe_default(self):
        from kernel.portfolio_qp.tasks import ApplyConvictionCapTask
        ctx = _mk_ctx({"A": 0.10}, w_upper=0.20)
        # remove the source object — simulates a ticker without scoring data
        ctx._qp_mu_source_map = {}
        ApplyConvictionCapTask().run(ctx)
        assert ctx._qp_w_upper[0] == pytest.approx(0.20, rel=1e-9)

    def test_inf_panel_score_safe_default(self):
        from kernel.portfolio_qp.tasks import ApplyConvictionCapTask
        ctx = _mk_ctx({"X": float("inf")}, w_upper=0.20)
        ApplyConvictionCapTask().run(ctx)
        assert ctx._qp_w_upper[0] == pytest.approx(0.20, rel=1e-9)

    def test_empty_tickers_noop(self):
        from kernel.portfolio_qp.tasks import ApplyConvictionCapTask
        ctx = _mk_ctx({}, w_upper=0.20)
        # No assertion needed beyond not raising
        ApplyConvictionCapTask().run(ctx)

    def test_w_upper_length_mismatch_noop(self):
        from kernel.portfolio_qp.tasks import ApplyConvictionCapTask
        ctx = _mk_ctx({"A": 0.10, "B": 0.50}, w_upper=0.20)
        ctx._qp_w_upper = np.array([0.20])  # wrong length
        before = ctx._qp_w_upper.copy()
        ApplyConvictionCapTask().run(ctx)
        np.testing.assert_array_equal(ctx._qp_w_upper, before)


# ── Invariant 5: wiring order in the Job ─────────────────────────────────

class TestWiringOrder:
    """ApplyConvictionCapTask must run AFTER ComputeQPConstraintsTask
    (which writes _qp_w_upper) and BEFORE
    BuildSectorConstraintMatrixTask + BuildCorrelationGroupConstraintTask
    (which anchor on _qp_w_upper.max())."""

    def test_apply_conviction_cap_runs_in_correct_slot(self):
        from kernel.portfolio_qp.job_qp import JointPortfolioQPJob
        job = JointPortfolioQPJob()
        task_names = [type(t).__name__ for t in job.tasks]
        assert "ComputeQPConstraintsTask"        in task_names
        assert "ApplyConvictionCapTask"          in task_names
        assert "BuildSectorConstraintMatrixTask" in task_names
        i_compute = task_names.index("ComputeQPConstraintsTask")
        i_apply   = task_names.index("ApplyConvictionCapTask")
        i_sector  = task_names.index("BuildSectorConstraintMatrixTask")
        assert i_compute < i_apply < i_sector, (
            f"Wiring order wrong: compute={i_compute} apply={i_apply} sector={i_sector}"
        )
