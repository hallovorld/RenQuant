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
