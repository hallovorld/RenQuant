"""Tests for JointPortfolioQPTask — QP-based replacement for JointActionTask.

Stage-1 contract: defaults preserve current behaviour (solver=greedy);
flipping `rotation.joint_actions.solver = "qp"` opts in.
"""
from __future__ import annotations

import datetime
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.task_joint_qp import JointPortfolioQPTask  # noqa: E402


@dataclass
class _Cand:
    ticker: str
    rank_score: float | None = None
    panel_score: float | None = None
    mu: float | None = None
    sigma: float | None = None


@dataclass
class _Hold:
    shares: float = 0.0
    rank_score: float | None = None
    panel_score: float | None = None
    mu: float | None = None
    sigma: float | None = None


@dataclass
class _Ctx:
    config: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    holdings:   dict = field(default_factory=dict)
    prices:     dict = field(default_factory=dict)
    cash: float = 10000.0
    portfolio_value: float = 10000.0
    today:  datetime.date = datetime.date(2026, 4, 26)
    regime: str = "BULL_CALM"
    confidence: float = 0.6
    bear_only: bool = False
    last_sell_dates: dict = field(default_factory=dict)
    orders: list = field(default_factory=list)
    exits: list = field(default_factory=list)
    counters: dict = field(default_factory=dict)


def _qp_on() -> dict:
    return {
        "rotation": {"joint_actions": {
            "enabled": True, "solver": "qp",
            "qp_risk_aversion": 3.0,
            "qp_cost_kappa": 0.0001,
            "qp_dw_max": 0.50,
            "qp_min_dw_pct": 0.005,
            "default_sigma": 0.05,
        }},
        "regime_params": {"BULL_CALM": {
            "max_position_pct": 0.20,
            "cash_reserve_pct": 0.0,
        }},
        "wash_sale_days": 0,
    }


# ── Flag dispatch ─────────────────────────────────────────────────────────────

class TestQPDispatch:
    def test_skips_when_joint_disabled(self):
        ctx = _Ctx(config={"rotation": {"joint_actions": {"enabled": False}}})
        ctx.candidates = [_Cand("A", mu=0.05, sigma=0.10)]
        ctx.prices = {"A": 100.0}
        ret = JointPortfolioQPTask().run(ctx)
        assert ret is False
        assert ctx.orders == [] and ctx.exits == []

    def test_skips_when_solver_is_greedy(self):
        ctx = _Ctx(config={
            "rotation": {"joint_actions": {
                "enabled": True, "solver": "greedy",
            }},
        })
        ctx.candidates = [_Cand("A", mu=0.05, sigma=0.10)]
        ctx.prices = {"A": 100.0}
        ret = JointPortfolioQPTask().run(ctx)
        assert ret is False

    def test_skips_when_bear_only(self):
        ctx = _Ctx(config=_qp_on())
        ctx.bear_only = True
        ctx.candidates = [_Cand("A", mu=0.05, sigma=0.10)]
        ctx.prices = {"A": 100.0}
        ret = JointPortfolioQPTask().run(ctx)
        assert ret is False


# ── Buy/sell directions ───────────────────────────────────────────────────────

class TestActionDirections:
    def test_positive_mu_emits_buy(self):
        ctx = _Ctx(config=_qp_on())
        ctx.candidates = [_Cand("A", mu=0.05, sigma=0.10)]
        ctx.prices = {"A": 100.0}
        ctx.cash = ctx.portfolio_value = 10000.0
        ret = JointPortfolioQPTask().run(ctx)
        assert ret is True
        assert len(ctx.orders) == 1
        assert ctx.orders[0]["ticker"] == "A"
        assert ctx.orders[0]["shares"] > 0

    def test_negative_mu_on_held_emits_sell(self):
        ctx = _Ctx(config=_qp_on())
        ctx.holdings = {"H": _Hold(shares=20, mu=-0.05, sigma=0.10)}
        ctx.prices = {"H": 100.0}
        ctx.cash = 8000.0       # held = 20 × 100 = 2000 (~20% of NAV)
        ctx.portfolio_value = 10000.0
        ret = JointPortfolioQPTask().run(ctx)
        assert ret is True
        assert len(ctx.exits) == 1
        ticker, sig = ctx.exits[0]
        assert ticker == "H"
        assert "qp" in sig.exit_type

    def test_zero_signal_no_action(self):
        """μ=0 + flat → optimum is no trade."""
        ctx = _Ctx(config=_qp_on())
        ctx.candidates = [_Cand("A", mu=0.0, sigma=0.10),
                          _Cand("B", mu=0.0, sigma=0.10)]
        ctx.prices = {"A": 100.0, "B": 100.0}
        JointPortfolioQPTask().run(ctx)
        assert ctx.orders == []
        assert ctx.exits == []


# ── Constraints ───────────────────────────────────────────────────────────────

class TestConstraints:
    def test_position_cap_respected(self):
        """Strong signal → position capped at max_position_pct × confidence."""
        ctx = _Ctx(config=_qp_on())
        ctx.candidates = [_Cand("A", mu=10.0, sigma=0.10)]   # very strong
        ctx.prices = {"A": 100.0}
        ctx.cash = ctx.portfolio_value = 10000.0
        # confidence=0.6 < 1.0; conf_scale clipped to ≥0.5 → 0.6
        # max_position = 0.20 × 0.6 = 0.12 of NAV = $1200 = 12 shares
        JointPortfolioQPTask().run(ctx)
        assert len(ctx.orders) == 1
        # Allow some room for solver imprecision
        assert ctx.orders[0]["shares"] <= 13

    def test_wash_sale_blocks_buy(self):
        ctx = _Ctx(config={**_qp_on(), "wash_sale_days": 30})
        ctx.candidates = [_Cand("A", mu=0.10, sigma=0.10)]
        ctx.prices = {"A": 100.0}
        ctx.last_sell_dates = {"A": datetime.date(2026, 4, 20)}  # 6 days ago
        JointPortfolioQPTask().run(ctx)
        assert ctx.orders == []   # blocked

    def test_min_dw_threshold_dropouts(self):
        """Δw < min_dw_pct → drop the trade (avoid dust)."""
        ctx = _Ctx(config=_qp_on())
        # Tiny μ → tiny Δw under any reasonable γ
        ctx.candidates = [_Cand("A", mu=0.0001, sigma=0.10)]
        ctx.prices = {"A": 100.0}
        JointPortfolioQPTask().run(ctx)
        assert ctx.orders == []


# ── Counters / logging ────────────────────────────────────────────────────────

class TestCounters:
    def test_counters_incremented(self):
        ctx = _Ctx(config=_qp_on())
        ctx.candidates = [_Cand("A", mu=0.05, sigma=0.10)]
        ctx.prices = {"A": 100.0}
        JointPortfolioQPTask().run(ctx)
        assert ctx.counters.get("qp_buys", 0) >= 1


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_pipeline_safe(self):
        ctx = _Ctx(config=_qp_on())
        ret = JointPortfolioQPTask().run(ctx)
        assert ret is True
        assert ctx.orders == [] and ctx.exits == []

    def test_zero_portfolio_skips(self):
        ctx = _Ctx(config=_qp_on())
        ctx.candidates = [_Cand("A", mu=0.05, sigma=0.10)]
        ctx.prices = {"A": 100.0}
        ctx.portfolio_value = 0.0
        ret = JointPortfolioQPTask().run(ctx)
        assert ret is True
        assert ctx.orders == []


# ── QP-REGIME-STATE-DUCK regression ───────────────────────────────────────────

class TestRegimeStateDuckTyping:
    """Audit fix QP-REGIME-STATE-DUCK (2026-04-26): regime_state can be
    either a dict (some test ctx) or a RegimeState dataclass (real
    pipeline ctx). QP task crashed with AttributeError when it was the
    dataclass. Fix: duck-type via getattr + isinstance."""

    def test_regime_state_as_dict(self):
        ctx = _Ctx(config=_qp_on())
        ctx.candidates = [_Cand("A", mu=0.05, sigma=0.10)]
        ctx.prices = {"A": 100.0}
        ctx.regime_state = {"drawdown": 0.05}
        ret = JointPortfolioQPTask().run(ctx)
        assert ret is True

    def test_regime_state_as_dataclass(self):
        from dataclasses import dataclass
        @dataclass
        class _RS:
            drawdown: float = 0.05
            regime: str = "BULL_CALM"
            confidence: float = 0.6
            in_transition: bool = False
            countdown: int = 0
        ctx = _Ctx(config=_qp_on())
        ctx.candidates = [_Cand("A", mu=0.05, sigma=0.10)]
        ctx.prices = {"A": 100.0}
        ctx.regime_state = _RS()
        ret = JointPortfolioQPTask().run(ctx)
        assert ret is True

    def test_regime_state_none(self):
        ctx = _Ctx(config=_qp_on())
        ctx.candidates = [_Cand("A", mu=0.05, sigma=0.10)]
        ctx.prices = {"A": 100.0}
        ctx.regime_state = None
        ret = JointPortfolioQPTask().run(ctx)
        assert ret is True
