"""Long-Short Phase 2A — QP allows w_lower < 0 when enabled.

Pinned invariants:
1. DEFAULT (long_short.enabled missing or false) → w_lower = 0.0 exactly.
   Existing long-only behavior preserved.
2. ENABLED + non-BEAR regime → w_lower = -max_short_pct × confidence.
3. ENABLED + BEAR regime → w_lower = 0.0 (no shorts in bear, risk-symmetric).
4. The patch only touches `ComputeQPConstraintsTask`; downstream solver
   already accepts negative w_lower (verified by inspection of qp_solver:241).

Reference: doc/arch/long-short.md.
Status: Phase 2A foundation. Sector-neutral, gross-cap, short stop-loss,
        wash-sale, tax, broker integration ALL still TODO.
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


def _make_ctx(regime="BULL_CALM", **cfg):
    ctx = SimpleNamespace()
    ctx._qp_tickers = ["AAA", "BBB", "CCC"]
    ctx.regime = regime
    ctx.confidence = 0.8
    ctx.regime_state = {"drawdown": 0.0}
    ctx.config = cfg
    return ctx


class TestLongShortPhase2A:

    def test_default_long_only(self):
        """No long_short config → w_lower = 0.0 (current behavior)."""
        from kernel.portfolio_qp.tasks import ComputeQPConstraintsTask
        ctx = _make_ctx(regime_params={"BULL_CALM": {"max_position_pct": 0.20}})
        ComputeQPConstraintsTask().run(ctx)
        assert ctx._qp_w_lower == 0.0

    def test_explicit_disable_long_only(self):
        from kernel.portfolio_qp.tasks import ComputeQPConstraintsTask
        ctx = _make_ctx(
            regime_params={"BULL_CALM": {"max_position_pct": 0.20}},
            long_short={"enabled": False},
        )
        ComputeQPConstraintsTask().run(ctx)
        assert ctx._qp_w_lower == 0.0

    def test_enabled_bull_allows_shorts(self):
        """long_short.enabled=true in BULL → w_lower = -max_short_pct × confidence."""
        from kernel.portfolio_qp.tasks import ComputeQPConstraintsTask
        ctx = _make_ctx(
            regime="BULL_CALM",
            regime_params={"BULL_CALM": {"max_position_pct": 0.20}},
            long_short={"enabled": True, "max_short_pct": 0.05},
        )
        ComputeQPConstraintsTask().run(ctx)
        # confidence 0.8 → conf_mult ≈ 1.0 for BULL_CALM (test based on
        # actual confidence_to_size_multiplier behavior).
        # Just check sign + magnitude bounds.
        assert ctx._qp_w_lower < 0.0
        assert ctx._qp_w_lower >= -0.05, f"shouldn't exceed max_short_pct, got {ctx._qp_w_lower}"

    def test_enabled_bear_blocks_shorts(self):
        """BEAR regime → w_lower forced to 0 even with long_short.enabled."""
        from kernel.portfolio_qp.tasks import ComputeQPConstraintsTask
        ctx = _make_ctx(
            regime="BEAR",
            regime_params={"BEAR": {"max_position_pct": 0.0}},
            long_short={"enabled": True, "max_short_pct": 0.05},
        )
        ComputeQPConstraintsTask().run(ctx)
        assert ctx._qp_w_lower == 0.0

    def test_no_breakage_for_existing_tests(self):
        """Sanity: ctx without long_short key still works (backward compat)."""
        from kernel.portfolio_qp.tasks import ComputeQPConstraintsTask
        ctx = _make_ctx(regime_params={"BULL_CALM": {"max_position_pct": 0.20}})
        # Should not raise
        ComputeQPConstraintsTask().run(ctx)
        # Output should match pre-Phase-2A behavior
        assert ctx._qp_w_lower == 0.0
        assert ctx._qp_w_upper.shape == (3,)
        # gross_max=None when long-only (no constraint imposed)
        assert ctx._qp_gross_max is None

    def test_gross_max_set_when_shorts_enabled(self):
        """long_short.enabled=true → gross_max read from config (default 1.30)."""
        from kernel.portfolio_qp.tasks import ComputeQPConstraintsTask
        ctx = _make_ctx(
            regime_params={"BULL_CALM": {"max_position_pct": 0.20}},
            long_short={"enabled": True, "max_short_pct": 0.05,
                        "max_gross_exposure": 1.40},
        )
        ComputeQPConstraintsTask().run(ctx)
        assert ctx._qp_gross_max == 1.40

    def test_gross_max_default_1_30(self):
        """When max_gross_exposure not set, defaults to 1.30 (Reg-T conservative)."""
        from kernel.portfolio_qp.tasks import ComputeQPConstraintsTask
        ctx = _make_ctx(
            regime_params={"BULL_CALM": {"max_position_pct": 0.20}},
            long_short={"enabled": True, "max_short_pct": 0.05},
        )
        ComputeQPConstraintsTask().run(ctx)
        assert ctx._qp_gross_max == 1.30

    def test_gross_max_bear_overrides_to_none(self):
        """BEAR regime: w_lower=0 AND gross_max=None (no gross constraint needed)."""
        from kernel.portfolio_qp.tasks import ComputeQPConstraintsTask
        ctx = _make_ctx(
            regime="BEAR",
            regime_params={"BEAR": {"max_position_pct": 0.0}},
            long_short={"enabled": True, "max_short_pct": 0.05,
                        "max_gross_exposure": 1.40},
        )
        ComputeQPConstraintsTask().run(ctx)
        assert ctx._qp_w_lower == 0.0
        assert ctx._qp_gross_max is None


class TestSolverGrossMaxConstraint:
    """Solver-level test: gross_max constraint forces Σ|w| ≤ gross_max."""

    def test_solver_rejects_above_gross(self):
        """When gross_max=1.0 + w_lower<0, solver returns Σ|w| ≤ 1.0."""
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        import numpy as _np
        n = 5
        w_curr = _np.zeros(n)
        mu = _np.array([+1.0, +0.5, 0.0, -0.5, -1.0])
        sigma = _np.full(n, 0.05)
        sol = solve_portfolio_qp(
            w_current=w_curr, mu=mu, sigma=sigma,
            risk_aversion=1.0, cost_kappa=0.0001,
            w_upper=_np.full(n, 0.30), w_lower=_np.full(n, -0.30),
            dw_max=_np.full(n, 1.0), cash_reserve=0.0,
            gross_max=1.0,
        )
        wp = sol.target_w
        gross = float(_np.sum(_np.abs(wp)))
        assert gross <= 1.0 + 1e-4, f"Σ|w|={gross} > gross_max=1.0"

    def test_solver_long_only_no_gross_constraint(self):
        """Default (gross_max=None) → no constraint added; same as pre-Phase-2A."""
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        import numpy as _np
        n = 5
        sol = solve_portfolio_qp(
            w_current=_np.zeros(n),
            mu=_np.array([+1.0, +0.5, 0.0, -0.5, -1.0]),
            sigma=_np.full(n, 0.05),
            risk_aversion=1.0,
            w_upper=_np.full(n, 0.30), w_lower=0.0,  # long-only
            dw_max=_np.full(n, 1.0),
            # gross_max=None (default)
        )
        # Should solve normally
        assert sol.status in ("optimal", "optimal_inaccurate")
        assert _np.all(sol.target_w >= -1e-9)  # no shorts
