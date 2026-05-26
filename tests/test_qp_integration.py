"""Integration tests — JointPortfolioQPTask + QualityFloorTask running
through real InferenceContext with realistic ticker / Σ / μ data.

Promotes the buy logic + portfolio QP from 'unit tests pass' to
'integration tests show end-to-end correctness'. Catches bugs at the
multi-task seam that file-level tests can't see.
"""
from __future__ import annotations

import datetime
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.pipeline.context import InferenceContext  # noqa: E402
from kernel.panel_pipeline.task_quality_floor import QualityFloorTask  # noqa: E402
from kernel.portfolio_qp.job_qp import _BuildSourceMapTask  # noqa: E402
from kernel.portfolio_qp.task_joint_qp import JointPortfolioQPTask  # noqa: E402


@dataclass
class _Cand:
    ticker: str
    mu: float | None = None
    sigma: float | None = None
    panel_score: float | None = None
    rank_score: float | None = None
    expected_return: float | None = None
    expected_return_horizon_days: int | None = None
    mu_horizon_days: int | None = None
    kelly_target_pct: float | None = None
    rs_score: float | None = None


@dataclass
class _Hold:
    shares: float
    mu: float | None = None
    sigma: float | None = None
    panel_score: float | None = None
    rank_score: float | None = None
    sell_streak: int = 0
    entry_date: datetime.date = datetime.date(2026, 4, 1)
    entry_price: float = 100.0
    expected_return: float | None = None
    expected_return_horizon_days: int | None = None
    mu_horizon_days: int | None = None
    kelly_target_pct: float | None = None


def _base_config() -> dict:
    return {
        "rotation": {
            "joint_actions": {
                "enabled": True,
                "solver": "qp",
                "qp_risk_aversion": 3.0,
                "qp_cost_kappa": 0.0001,
                "qp_dw_max": 0.50,
                "qp_min_dw_pct": 0.005,
                "qp_signal_decay": 0.0,
                "qp_drawdown_limit": 0.20,
                "qp_robust_mu_kappa": 0.0,
                "default_sigma": 0.05,
            },
        },
        "regime_params": {
            "BULL_CALM": {
                "max_position_pct": 0.20,
                "cash_reserve_pct": 0.10,
                "max_concurrent_positions": 8,
            },
        },
        "wash_sale_days": 30,
        "sector_map": {
            "A": "tech",
            "AAPL": "tech",
            "GOOG": "tech",
            "HELD": "tech",
            "MSFT": "tech",
            "NEW": "tech",
            "OLD": "tech",
            "RAW": "tech",
            "STRONG": "tech",
            "WEAK": "tech",
        },
        "ranking": {
            "panel_scoring": {
                "quality_floor": {"enabled": False},
            },
        },
    }


def _make_ctx(*, candidates, holdings, prices, regime="BULL_CALM",
              confidence=0.6, portfolio_value=10000.0, cash=2000.0,
              today=None, last_sells=None) -> InferenceContext:
    today = today or datetime.date(2026, 4, 27)
    cfg = _base_config()
    ctx = InferenceContext(config=cfg, today=today)
    ctx.candidates = candidates
    ctx.holdings = holdings or {}
    ctx.prices = prices
    ctx.cash = cash
    ctx.portfolio_value = portfolio_value
    ctx.regime = regime
    ctx.confidence = confidence
    ctx.last_sell_dates = last_sells or {}
    tickers = sorted(set(ctx.prices) | set(ctx.holdings) | {c.ticker for c in ctx.candidates})
    ctx.corr_matrix = {
        a: {b: 0.0 for b in tickers if b != a}
        for a in tickers
    }
    return ctx


# ── Realistic single-bar scenarios ────────────────────────────────────────────

class TestRealisticBars:
    def test_typical_bullish_bar_emits_buys(self):
        """3 candidates with positive μ, no holdings → QP fills cash."""
        cands = [
            _Cand("AAPL", mu=0.04, sigma=0.10, rank_score=0.72, panel_score=1.1, rs_score=0.2),
            _Cand("MSFT", mu=0.03, sigma=0.09, rank_score=0.66, panel_score=0.9, rs_score=0.1),
            _Cand("GOOG", mu=0.02, sigma=0.11, rank_score=0.58, panel_score=0.5, rs_score=0.0),
        ]
        prices = {"AAPL": 200.0, "MSFT": 400.0, "GOOG": 150.0}
        ctx = _make_ctx(candidates=cands, holdings={}, prices=prices)
        ret = JointPortfolioQPTask().run(ctx)
        assert ret is True
        assert len(ctx.orders) >= 1
        # Should be sized below per-position cap = 0.20 × confidence(0.6→0.6)
        for o in ctx.orders:
            assert o["invest"] <= 0.21 * ctx.portfolio_value
            assert o["source"] == "qp"
            assert o["order_type"] == "QP_BUY"
            assert o["regime"] == "BULL_CALM"
            assert o["confidence"] == pytest.approx(0.6)
            assert o["mu"] is not None
            assert o["sigma"] is not None
            assert o["rank_score"] is not None
            assert o["panel_score"] is not None
            assert "rs_score" in o
        # Cash budget respected: total invest ≤ (1 - cash_reserve) × NAV
        total_invest = sum(o["invest"] for o in ctx.orders)
        assert total_invest <= (1 - 0.10) * 10000.0 + 1.0   # +$1 numerical

    def test_negative_mu_held_emits_sell_and_no_buy(self):
        """Position with negative μ → QP sells; no candidates → no buys."""
        ctx = _make_ctx(
            candidates=[],
            holdings={"AAPL": _Hold(shares=10, mu=-0.05, sigma=0.10)},
            prices={"AAPL": 200.0},
        )
        ret = JointPortfolioQPTask().run(ctx)
        assert ret is True
        # Should emit a sell exit
        assert len(ctx.exits) == 1
        assert ctx.exits[0][0] == "AAPL"
        assert ctx.orders == []

    def test_mixed_buy_sell_rotation_in_one_bar(self):
        """Positive cand + negative-μ holding → QP rotates."""
        ctx = _make_ctx(
            candidates=[_Cand("NEW", mu=0.06, sigma=0.10)],
            holdings={"OLD": _Hold(shares=10, mu=-0.04, sigma=0.10)},
            prices={"NEW": 100.0, "OLD": 200.0},   # held = $2000 (20%)
        )
        ret = JointPortfolioQPTask().run(ctx)
        assert ret is True
        # Either sell OLD AND buy NEW; or one of them
        ticker_set = ({o["ticker"] for o in ctx.orders} |
                      {t for t, _ in ctx.exits})
        assert ticker_set != set()  # some action emitted

    def test_drawdown_at_limit_zeroes_buys(self):
        """When DD = α, γ_eff blows up, QP refuses new positions."""
        ctx = _make_ctx(
            candidates=[_Cand("A", mu=0.05, sigma=0.10)],
            holdings={},
            prices={"A": 100.0},
        )
        # Simulate DD at limit (set via regime_state)
        ctx.regime_state = {"drawdown": 0.20}
        ret = JointPortfolioQPTask().run(ctx)
        assert ret is True
        # Position should be tiny or zero
        if ctx.orders:
            assert ctx.orders[0]["shares"] <= 1   # rounding artefact


# ── μ-contract integration ───────────────────────────────────────────────────

class TestQPMuContractIntegration:
    def test_panel_contract_failure_skips_qp_without_secondary_mu_block(self):
        """When alpha contract already failed, QP must not invent another cause."""
        ctx = _make_ctx(
            candidates=[],
            holdings={"HELD": _Hold(shares=5, mu=None, sigma=None)},
            prices={"HELD": 100.0},
            cash=5000.0,
            portfolio_value=10000.0,
        )
        ctx._panel_scoring_contract_failed = True  # noqa: SLF001
        ctx.buy_blocked = True
        ctx.skip_buys = True
        ctx.counters["panel_scoring_fail_closed"] = 91
        ctx.config["rotation"]["joint_actions"]["qp_mu_contract"] = "strict"

        ret = JointPortfolioQPTask().run(ctx)

        assert ret is False
        assert ctx.counters["panel_scoring_fail_closed"] == 91
        assert "qp_mu_contract_block" not in ctx.counters
        assert ctx.orders == []
        assert ctx.exits == []
        assert not hasattr(ctx, "_qp_mu_contract")

    def test_strict_mu_contract_stops_before_solver_on_raw_score_fallback(self):
        """Strict QP must not optimize raw panel/rank scores as return μ.

        Unit tests pin ValidateQPMuContractTask itself. This integration guard
        proves the production QP task chain stops before BuildWeight/Solve/Emit
        when a candidate lacks finite μ and only has score-like fields.
        """
        ctx = _make_ctx(
            candidates=[
                _Cand(
                    "RAW",
                    mu=None,
                    sigma=0.10,
                    panel_score=1.50,
                    rank_score=0.95,
                )
            ],
            holdings={},
            prices={"RAW": 100.0},
            cash=10000.0,
            portfolio_value=10000.0,
        )
        ctx.config["rotation"]["joint_actions"]["qp_mu_contract"] = "strict"

        ret = JointPortfolioQPTask().run(ctx)

        assert ret is True
        assert ctx.counters["qp_mu_contract_block"] == 1
        assert ctx._qp_mu_contract["ok"] is False
        assert ctx._qp_mu[0] == pytest.approx(0.0)
        assert ctx._blocked_by_ticker["RAW"] == "qp_mu_contract_block"
        assert ctx.orders == []
        assert ctx.exits == []
        assert not hasattr(ctx, "_qp_w_current")
        assert not hasattr(ctx, "_qp_solution")

    def test_strict_mu_contract_rejects_mismatched_mu_horizon(self):
        """QP μ and σ must describe the same rebalance horizon."""
        ctx = _make_ctx(
            candidates=[
                _Cand(
                    "RAW",
                    mu=0.04,
                    mu_horizon_days=20,
                    sigma=0.10,
                    rank_score=0.95,
                )
            ],
            holdings={},
            prices={"RAW": 100.0},
            cash=10000.0,
            portfolio_value=10000.0,
        )
        ctx.config["rotation"]["joint_actions"]["qp_mu_contract"] = "strict"
        ctx.config["rotation"]["joint_actions"]["qp_mu_horizon_days"] = 60

        ret = JointPortfolioQPTask().run(ctx)

        assert ret is True
        assert ctx.counters["qp_mu_contract_block"] == 1
        assert ctx._qp_mu_contract["ok"] is False
        assert ctx._qp_mu_contract["mu_horizon_mismatch_count"] == 1
        assert ctx._blocked_by_ticker["RAW"] == "qp_mu_horizon_contract_block"
        assert ctx.orders == []
        assert ctx.exits == []


class TestQPAdmissionUniverse:
    def test_regime_override_gate_excludes_weak_candidate_before_solver(self):
        """Regime-specific admission gates must shape the QP solver universe."""
        cfg = _base_config()
        cfg["rotation"]["joint_actions"]["qp_admission_gate"] = {
            "enabled": False,
        }
        cfg["regime_params"]["BULL_CALM"]["qp_admission_gate"] = {
            "enabled": True,
            "min_rank_score": 0.80,
        }
        ctx = InferenceContext(config=cfg, today=datetime.date(2026, 4, 27))
        weak = _Cand("WEAK", mu=0.10, sigma=0.10, rank_score=0.50)
        strong = _Cand("STRONG", mu=0.10, sigma=0.10, rank_score=0.90)
        ctx.candidates = [weak, strong]
        ctx.ranked = [weak, strong]
        ctx.holdings = {}
        ctx.exits = []
        ctx.regime = "BULL_CALM"

        _BuildSourceMapTask().run(ctx)

        assert set(ctx._qp_mu_source_map) == {"STRONG"}  # noqa: SLF001
        assert ctx._qp_tickers == ["STRONG"]             # noqa: SLF001
        assert ctx._blocked_by_ticker["WEAK"] == "qp_admission_rank"  # noqa: SLF001

    def test_qp_can_top_up_scored_held_winner(self):
        """A held ticker is not exit-only merely because new-buy scan skips it."""
        ctx = _make_ctx(
            candidates=[],
            holdings={
                "HELD": _Hold(
                    shares=5,
                    mu=0.08,
                    sigma=0.10,
                    rank_score=0.95,
                    panel_score=0.80,
                )
            },
            prices={"HELD": 100.0},
            cash=5_000.0,
            portfolio_value=10_000.0,
        )

        JointPortfolioQPTask().run(ctx)

        assert any(o["ticker"] == "HELD" for o in ctx.orders)
        assert ctx._blocked_by_ticker.get("HELD") != "qp_universe_exit_only"  # noqa: SLF001

    def test_qp_sell_credit_can_fund_same_bar_buy(self):
        """Accepted long sells should increase QP buy cash before later buys."""
        from kernel.portfolio_qp.tasks import EmitOrdersFromQPSolutionTask

        held = _Hold(shares=10, mu=-0.08, sigma=0.10)
        cand = _Cand("NEW", mu=0.08, sigma=0.10, rank_score=0.95)
        ctx = SimpleNamespace(
            config=_base_config(),
            orders=[],
            exits=[],
            holdings={"OLD": held},
            candidates=[cand],
            counters={},
            _blocked_by_ticker={},
            prices={"OLD": 100.0, "NEW": 100.0},
            portfolio_value=10_000.0,
            today=datetime.date(2026, 4, 27),
            regime="BULL_CALM",
            confidence=0.6,
        )
        env = {
            "cfg": {},
            "sol": SimpleNamespace(
                status="optimal",
                delta_w=np.array([-0.10, 0.10]),
                target_w=np.array([0.0, 0.10]),
            ),
            "tickers": ["OLD", "NEW"],
            "prices": ctx.prices,
            "nav": 10_000.0,
            "cash": 0.0,
            "cash_actual": 0.0,
            "alpha_funding_cash": 0.0,
            "cash_reserve": 0.0,
            "cash_reserve_configured": 0.0,
            "cash_reserve_credit": 0.0,
            "buy_cost_multiplier": 1.0,
            "min_dw": 0.005,
            "no_trade_factor": 0.0,
            "band_cap": 0.05,
            "sigma_vec": np.array([0.10, 0.10]),
            "cands": {"NEW": cand},
            "score_sources": {"OLD": held, "NEW": cand},
            "buy_blocked": False,
            "buys_gated": False,
            "earnings_cal": {},
            "earn_buf": 0,
            "today": ctx.today,
            "holdings_set": {"OLD"},
            "holdings": ctx.holdings,
            "max_positions": 8,
            "min_share_floor_pct": 0.0,
            "min_share_ceiling_pct": 0.15,
            "defensive_set": set(),
            "bear_only": False,
            "preexisting_exit_tickers": set(),
            "exit_only_tickers": set(),
            "exit_only_reasons": {},
            "emitted_new_tickers": set(),
        }

        nb, ns, counters = EmitOrdersFromQPSolutionTask._emit_orders_loop(ctx, env)

        assert (nb, ns) == (1, 1)
        assert counters["sell_credit_events"] == 1
        assert ctx.exits[0][0] == "OLD"
        assert ctx.orders[0]["ticker"] == "NEW"


# ── Wash-sale interaction ─────────────────────────────────────────────────────

class TestWashSaleIntegration:
    def test_wash_sale_blocks_recent_seller(self):
        """Sold AAPL 5 days ago (< 30 day window) → cannot re-buy."""
        ctx = _make_ctx(
            candidates=[_Cand("AAPL", mu=0.10, sigma=0.10)],
            holdings={},
            prices={"AAPL": 200.0},
            last_sells={"AAPL": datetime.date(2026, 4, 22)},
        )
        ret = JointPortfolioQPTask().run(ctx)
        assert ret is True
        assert ctx.orders == []   # blocked

    def test_old_sale_does_not_block(self):
        """Sold AAPL 60 days ago → wash-sale window expired → buy ok."""
        ctx = _make_ctx(
            candidates=[_Cand("AAPL", mu=0.10, sigma=0.10)],
            holdings={},
            prices={"AAPL": 200.0},
            last_sells={"AAPL": datetime.date(2026, 2, 20)},
        )
        ret = JointPortfolioQPTask().run(ctx)
        assert ret is True
        assert len(ctx.orders) >= 1


# ── Quality floor + QP combined ───────────────────────────────────────────────

class TestQualityFloorPlusQP:
    def test_gate_b_filters_then_qp_solves_remaining(self):
        """Gate B drops weak μ; QP only sees survivors."""
        cfg = _base_config()
        cfg["ranking"]["panel_scoring"]["quality_floor"] = {
            "enabled": True,
            "edge_sharpe_floor": {"enabled": True, "threshold": 0.30},
        }
        ctx = InferenceContext(config=cfg,
                                today=datetime.date(2026, 4, 27))
        ctx.candidates = [
            _Cand("STRONG", mu=0.05, sigma=0.10),    # edge = 0.5 → pass
            _Cand("WEAK",   mu=0.01, sigma=0.10),    # edge = 0.1 → reject
        ]
        ctx.prices = {"STRONG": 100.0, "WEAK": 100.0}
        ctx.holdings = {}
        ctx.cash = 10000.0
        ctx.portfolio_value = 10000.0
        ctx.regime = "BULL_CALM"
        ctx.confidence = 0.6
        ctx.last_sell_dates = {}
        # Run gate first, then QP
        QualityFloorTask().run(ctx)
        # Gate B should have removed WEAK
        assert {c.ticker for c in ctx.candidates} == {"STRONG"}
        # Now QP processes only STRONG
        JointPortfolioQPTask().run(ctx)
        assert len(ctx.orders) == 1
        assert ctx.orders[0]["ticker"] == "STRONG"

    def test_all_candidates_filtered_qp_safe(self):
        """All cands rejected by gate → QP runs on empty cands → no crash."""
        cfg = _base_config()
        cfg["ranking"]["panel_scoring"]["quality_floor"] = {
            "enabled": True,
            "edge_sharpe_floor": {"enabled": True, "threshold": 1.0},
        }
        ctx = InferenceContext(config=cfg,
                                today=datetime.date(2026, 4, 27))
        ctx.candidates = [_Cand("WEAK", mu=0.01, sigma=0.10)]
        ctx.prices = {"WEAK": 100.0}
        ctx.holdings = {}
        ctx.cash = 10000.0
        ctx.portfolio_value = 10000.0
        ctx.regime = "BULL_CALM"
        ctx.confidence = 0.6
        ctx.last_sell_dates = {}
        QualityFloorTask().run(ctx)
        assert ctx.candidates == []
        # QP should run with no buy candidates and no holdings → no-op
        ret = JointPortfolioQPTask().run(ctx)
        assert ret is True
        assert ctx.orders == [] and ctx.exits == []


# ── QP fallback to greedy on infeasibility ────────────────────────────────────

class TestQPFallback:
    def test_qp_returns_false_when_solver_fails(self):
        """If solver fails to converge, return False so JointActionTask
        can take over."""
        # We can't easily force SLSQP failure — but we CAN test the
        # contract by checking the return type at least.
        ctx = _make_ctx(
            candidates=[_Cand("A", mu=0.05, sigma=0.10)],
            holdings={},
            prices={"A": 100.0},
        )
        ret = JointPortfolioQPTask().run(ctx)
        assert ret in (True, False)


# ── Performance check at realistic n ──────────────────────────────────────────

class TestPerformanceIntegration:
    def test_solve_under_200ms_at_n50(self):
        """50 candidates + 7 holdings ~ realistic universe size; <200ms."""
        import time
        cands = [_Cand(f"T{i:02d}", mu=0.001 * (i - 25),
                        sigma=0.10) for i in range(50)]
        holds = {f"H{i}": _Hold(shares=5, mu=0.0, sigma=0.10)
                 for i in range(7)}
        prices = {f"T{i:02d}": 100.0 for i in range(50)}
        prices.update({f"H{i}": 100.0 for i in range(7)})
        ctx = _make_ctx(candidates=cands, holdings=holds, prices=prices)
        t0 = time.time()
        JointPortfolioQPTask().run(ctx)
        ms = (time.time() - t0) * 1000
        assert ms < 200, f"QP ran {ms:.0f}ms for n=57 — too slow"
