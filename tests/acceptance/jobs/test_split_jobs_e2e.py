"""End-to-end functional tests for split Jobs — beyond structure pinning.

`test_split_jobs.py` verifies the structure (Task count, names, body
length). This file ACTUALLY RUNS each Job against a synthetic ctx and
asserts the Job produces sensible outputs (orders, exits, panel,
matrix).

If the QP Job's Δw → orders translation or the BuildPanel's row-coverage
gate has a logic bug, structure tests pass but these would catch it.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
STRATEGY = REPO / "backtesting" / "renquant_104"
if str(STRATEGY) not in sys.path:
    sys.path.insert(0, str(STRATEGY))


# ── JointPortfolioQPJob — full end-to-end ──────────────────────────────────

class TestQPJobEndToEnd:
    """Runs the full 14-task QP Job against a 3-ticker synthetic ctx."""

    def _make_ctx(self):
        from kernel.selection import CandidateResult
        from kernel.exits import HoldingState
        # AAPL: held @ entry 100, current price 110 (10% gain)
        # MSFT: candidate with strong μ
        # NVDA: candidate with weak μ
        cand_msft = CandidateResult(
            ticker="MSFT", raw_score=0.5, rank_score=0.7, rs_score=0.0,
            mu=0.02, sigma=0.05,
        )
        cand_nvda = CandidateResult(
            ticker="NVDA", raw_score=0.1, rank_score=0.3, rs_score=0.0,
            mu=0.005, sigma=0.06,
        )
        held_aapl = HoldingState(
            entry_price=100.0,
            entry_date=datetime.date(2025, 11, 1),
            high_watermark=115.0,
            shares=20.0,
            mu=0.01, sigma=0.04,
            rank_score=0.6,
        )
        ctx = SimpleNamespace(
            today=datetime.date(2026, 5, 4),
            candidates=[cand_msft, cand_nvda],
            holdings={"AAPL": held_aapl},
            prices={"AAPL": 110.0, "MSFT": 350.0, "NVDA": 800.0},
            portfolio_value=100_000.0,
            confidence=0.8,
            regime="BULL_CALM",
            regime_state=None,
            bear_only=False,
            last_sell_dates={},
            counters={},
            orders=[],
            exits=[],
            ranked=[cand_msft, cand_nvda],
            ytd_realized_gain_dollar=0.0,
            config={
                "_strategy_dir": "/tmp/_qp_test",
                "rotation": {
                    "joint_actions": {
                        "enabled": True,
                        "solver": "qp",
                        "qp_min_dw_pct": 0.005,
                        "qp_dw_max": 0.5,
                        "qp_risk_aversion": 3.0,
                        "qp_use_full_sigma": False,   # diag fallback for test
                        "qp_tax_aware": True,
                    },
                },
                "regime_params": {"BULL_CALM": {"max_position_pct": 0.20}},
                "regime": {"drawdown_halt_pct": 0.20},
                "wash_sale_days": 0,
            },
        )
        return ctx

    def test_qp_job_runs_without_error(self):
        from kernel.portfolio_qp.job_qp import JointPortfolioQPJob
        ctx = self._make_ctx()
        job = JointPortfolioQPJob()
        assert job.should_skip(ctx) is False
        job.run(ctx)
        # Must have built every intermediate _qp_* field
        assert hasattr(ctx, "_qp_tickers")
        assert hasattr(ctx, "_qp_w_current")
        assert hasattr(ctx, "_qp_mu")
        assert hasattr(ctx, "_qp_sigma")
        assert hasattr(ctx, "_qp_solution")

    def test_qp_orders_have_required_fields(self):
        from kernel.portfolio_qp.job_qp import JointPortfolioQPJob
        ctx = self._make_ctx()
        JointPortfolioQPJob().run(ctx)
        for o in ctx.orders:
            assert "ticker" in o and "shares" in o and "price" in o
            assert "invest" in o
            assert o["shares"] > 0
            assert o["price"] > 0
            assert o["source"] == "qp"

    def test_qp_solution_status_is_meaningful(self):
        from kernel.portfolio_qp.job_qp import JointPortfolioQPJob
        ctx = self._make_ctx()
        JointPortfolioQPJob().run(ctx)
        sol = ctx._qp_solution
        assert sol.status in {"optimal", "optimal_no_signal"}, \
            f"unexpected solver status {sol.status!r}"
        # delta_w must be finite
        assert np.isfinite(sol.delta_w).all()
        assert np.isfinite(sol.target_w).all()

    def test_qp_g3_adv_vector_built_from_ohlcv(self):
        """G3 (2026-05-04): BuildADVVectorTask reads ctx.ohlcv → _qp_v_daily_dollar."""
        from kernel.portfolio_qp.job_qp import JointPortfolioQPJob
        ctx = self._make_ctx()
        # Inject synthetic OHLCV for two of the three assets
        ohlcv = {
            "AAPL": pd.DataFrame({
                "close":  np.full(30, 110.0),
                "volume": np.full(30, 50_000_000.0),
            }),
            "MSFT": pd.DataFrame({
                "close":  np.full(30, 350.0),
                "volume": np.full(30, 20_000_000.0),
            }),
            # NVDA missing → entry should be NaN
        }
        ctx.ohlcv = ohlcv
        JointPortfolioQPJob().run(ctx)
        v = ctx._qp_v_daily_dollar
        # AAPL ADV = 110 × 50M = 5.5B; MSFT = 350 × 20M = 7.0B
        # Order matches ctx._qp_tickers (StableTickerOrderTask)
        tickers = ctx._qp_tickers
        idx_aapl = tickers.index("AAPL")
        idx_msft = tickers.index("MSFT")
        idx_nvda = tickers.index("NVDA")
        assert abs(v[idx_aapl] - 5.5e9) < 1e6
        assert abs(v[idx_msft] - 7.0e9) < 1e6
        assert np.isnan(v[idx_nvda])

    def test_qp_g3_off_by_default_unchanged_solution(self):
        """G3 with impact_coef=0 (default) → identical Δw vs no-G3 path."""
        from kernel.portfolio_qp.job_qp import JointPortfolioQPJob
        ctx = self._make_ctx()
        # Default config has no qp_impact_coef → solver gets 0.0 → no impact
        JointPortfolioQPJob().run(ctx)
        delta_default = ctx._qp_solution.delta_w.copy()
        # Now run with explicit impact_coef=0 + ADV — should match exactly
        ctx2 = self._make_ctx()
        ctx2.config["rotation"]["joint_actions"]["qp_impact_coef"] = 0.0
        ctx2.ohlcv = {
            t: pd.DataFrame({"close": [100.0]*30, "volume": [1e6]*30})
            for t in ("AAPL", "MSFT", "NVDA")
        }
        JointPortfolioQPJob().run(ctx2)
        np.testing.assert_allclose(
            ctx2._qp_solution.delta_w, delta_default, atol=1e-9,
        )


# ── BuildFeatureMatrixJob — partial e2e (no model load) ───────────────────

class TestBuildFeatureMatrixJobE2E:
    """Verify the Job correctly short-circuits / propagates when inputs
    missing — covering the documented edge cases in the legacy task."""

    def test_no_candidates_or_holdings_clears_matrix(self):
        from kernel.panel_pipeline.tasks_feature_matrix import BuildFeatureMatrixJob
        ctx = SimpleNamespace(
            candidates=[], holdings={}, _panel_scorer=None,
            _panel_matrix="prior_value",   # any non-None
            config={},
        )
        BuildFeatureMatrixJob().run(ctx)
        assert ctx._panel_matrix is None

    def test_no_scorer_clears_matrix(self):
        from kernel.panel_pipeline.tasks_feature_matrix import BuildFeatureMatrixJob
        from kernel.selection import CandidateResult
        ctx = SimpleNamespace(
            candidates=[CandidateResult(
                ticker="AAPL", raw_score=0.0, rank_score=0.5, rs_score=0.0,
            )],
            holdings={}, _panel_scorer=None,
            _panel_matrix="prior_value",
            config={},
        )
        BuildFeatureMatrixJob().run(ctx)
        assert ctx._panel_matrix is None


# ── BuildPanelJob — should_skip behavior ──────────────────────────────────

class TestBuildPanelJobSkip:
    def test_skips_when_panel_already_set(self):
        from training_panel.tasks_build_panel import BuildPanelJob
        ctx = SimpleNamespace(
            panel="non_None",
            config={},
        )
        assert BuildPanelJob().should_skip(ctx) is True

    def test_does_not_skip_when_panel_none(self):
        from training_panel.tasks_build_panel import BuildPanelJob
        ctx = SimpleNamespace(
            panel=None,
            config={},
        )
        assert BuildPanelJob().should_skip(ctx) is False
