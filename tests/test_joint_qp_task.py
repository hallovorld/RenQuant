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
    buy_blocked: bool = False
    skip_buys: bool = False
    earnings_calendar: dict = field(default_factory=dict)
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


class TestBuyBlockedRespected:
    """2026-05-05 wl183 incident bug 3: when buy_blocked is set
    (DrawdownGate, VelocityCrash, EarningsBlackout regime), QP rebalance
    must NOT increase any position. Sells still allowed so the circuit
    can de-risk. Pre-fix the QP top-up path ignored buy_blocked → bar X
    QP_BUY +20% (against the circuit), bar X+1 SELL −24% (regime calmed)
    → 10bps round-trip friction every regime flip. Bled wl183 B2 from
    Sharpe 1.10 (wl103 baseline) into −0.07."""

    def test_buy_blocked_suppresses_qp_top_up(self):
        ctx = _Ctx(config=_qp_on())
        ctx.buy_blocked = True   # <-- the new gate
        # Strong-mu candidate that would normally trigger a QP_BUY
        ctx.candidates = [_Cand("A", mu=0.05, sigma=0.10)]
        ctx.prices = {"A": 100.0}
        ctx.cash = ctx.portfolio_value = 10000.0
        JointPortfolioQPTask().run(ctx)
        # Buys must be suppressed; orders empty
        assert ctx.orders == [], (
            f"buy_blocked must suppress all QP_BUY emissions; got "
            f"{len(ctx.orders)} order(s) — bug 3 from wl183 incident"
        )

    def test_buy_blocked_allows_sells(self):
        """Sells must still fire on buy_blocked bars so circuit can
        de-risk during drawdowns."""
        ctx = _Ctx(config=_qp_on())
        ctx.buy_blocked = True
        # Held name with negative-mu signal → optimum is to SELL it
        ctx.holdings = {"H": _Hold(shares=20, mu=-0.05, sigma=0.10)}
        ctx.prices = {"H": 100.0}
        ctx.cash = 8000.0
        ctx.portfolio_value = 10000.0
        JointPortfolioQPTask().run(ctx)
        # Sell still allowed
        assert len(ctx.exits) >= 1, (
            "buy_blocked must NOT block sells — circuit must be able "
            "to de-risk during drawdowns"
        )

    def test_buy_blocked_top_up_on_held_also_blocked(self):
        """The whiplash specifically came from QP topping up an EXISTING
        holding while buy_blocked. Verify Δw>0 on held names is also
        suppressed (not just new-entry buys)."""
        ctx = _Ctx(config=_qp_on())
        ctx.buy_blocked = True
        # Already holding "H", QP would normally top it up given +mu
        ctx.holdings = {"H": _Hold(shares=10, mu=0.05, sigma=0.10)}
        ctx.prices = {"H": 100.0}
        ctx.cash = 9000.0
        ctx.portfolio_value = 10000.0
        JointPortfolioQPTask().run(ctx)
        # No top-up BUY orders
        assert ctx.orders == [], (
            f"buy_blocked must suppress QP top-ups too; got "
            f"{len(ctx.orders)} order(s) — bug 3 reopened"
        )

    def test_skip_buys_also_suppresses_qp_top_up(self):
        """Persistent drawdown halt (ctx.skip_buys=True from
        DrawdownCircuitTask) must also suppress QP buy emissions.
        skip_buys is the longer-lived sibling of buy_blocked — when
        portfolio drawdown crosses halt_pct, skip_buys stays True until
        recovery below resume_pct. QP must respect both flags."""
        ctx = _Ctx(config=_qp_on())
        ctx.skip_buys = True   # drawdown halted
        ctx.candidates = [_Cand("A", mu=0.05, sigma=0.10)]
        ctx.prices = {"A": 100.0}
        ctx.cash = ctx.portfolio_value = 10000.0
        JointPortfolioQPTask().run(ctx)
        assert ctx.orders == [], (
            f"skip_buys must suppress all QP_BUY emissions; got "
            f"{len(ctx.orders)} — drawdown circuit was bypassed"
        )

    def test_skip_buys_allows_sells(self):
        """Sells must still fire when skip_buys=True so drawdown halt
        can de-risk the portfolio."""
        ctx = _Ctx(config=_qp_on())
        ctx.skip_buys = True
        ctx.holdings = {"H": _Hold(shares=20, mu=-0.05, sigma=0.10)}
        ctx.prices = {"H": 100.0}
        ctx.cash = 8000.0
        ctx.portfolio_value = 10000.0
        JointPortfolioQPTask().run(ctx)
        assert len(ctx.exits) >= 1, (
            "skip_buys must NOT block sells — drawdown halt must be "
            "able to de-risk"
        )


class TestNonFiniteDeltaWGuard:
    """Bug 9 fix (2026-05-05): solver returns sol.status='optimal' but
    sol.delta_w can occasionally contain NaN/inf on numerically-degenerate
    inputs (zero-vol asset, near-singular Σ). Pre-fix, `int(abs(NaN) ×
    nav / px)` raised ValueError mid-loop → uncaught → entire bar's
    Phase 3 unwound, no signal recorded.
    Post-fix: skip non-finite entries with a WARN log, rest of the bar
    proceeds normally."""

    def test_nan_dw_skipped_not_crashed(self):
        """Construct a synthetic _qp_solution with NaN dw — emit must
        not crash and must skip the bad entry."""
        from kernel.portfolio_qp.tasks import EmitOrdersFromQPSolutionTask
        from kernel.portfolio_qp.qp_solver import QPSolution
        import numpy as np

        ctx = _Ctx(config=_qp_on())
        # 2 tickers; NaN dw on ticker B
        ctx._qp_tickers = ["A", "B"]
        ctx.candidates = [_Cand("A", mu=0.05, sigma=0.10),
                          _Cand("B", mu=0.05, sigma=0.10)]
        ctx.prices = {"A": 100.0, "B": 100.0}
        ctx.cash = ctx.portfolio_value = 10000.0
        ctx._qp_solution = QPSolution(
            delta_w=np.array([0.10, float("nan")]),
            target_w=np.array([0.10, 0.10]),
            objective=0.001,
            n_iter=5,
            status="optimal",
            diagnostics={},
        )
        # Should NOT crash:
        EmitOrdersFromQPSolutionTask().run(ctx)
        # A still emits a BUY; B gets skipped.
        a_orders = [o for o in ctx.orders
                     if (o.get("ticker") if isinstance(o, dict)
                         else getattr(o, "ticker", None)) == "A"]
        b_orders = [o for o in ctx.orders
                     if (o.get("ticker") if isinstance(o, dict)
                         else getattr(o, "ticker", None)) == "B"]
        assert len(a_orders) == 1, "A's finite Δw must still produce a BUY"
        assert len(b_orders) == 0, "B's NaN Δw must be skipped, not crash"

    def test_inf_dw_skipped_not_crashed(self):
        """Same guard for +inf delta_w."""
        from kernel.portfolio_qp.tasks import EmitOrdersFromQPSolutionTask
        from kernel.portfolio_qp.qp_solver import QPSolution
        import numpy as np

        ctx = _Ctx(config=_qp_on())
        ctx._qp_tickers = ["A"]
        ctx.candidates = [_Cand("A", mu=0.05, sigma=0.10)]
        ctx.prices = {"A": 100.0}
        ctx.cash = ctx.portfolio_value = 10000.0
        ctx._qp_solution = QPSolution(
            delta_w=np.array([float("inf")]),
            target_w=np.array([0.0]),
            objective=0.0,
            n_iter=5,
            status="optimal",
            diagnostics={},
        )
        # Should NOT crash:
        EmitOrdersFromQPSolutionTask().run(ctx)
        assert ctx.orders == [], "inf Δw must be skipped, not produce an order"


class TestEarningsBlackoutQPTopUp:
    """2026-05-05 wl183 incident bug 4: QP had no earnings awareness.
    The buy-side EarningsFilterTask blocks new entries within ±N days
    of earnings, but QP could top-up an existing holding right into
    the print. Same gap-risk rationale that justifies blocking entries
    applies to top-ups (5–15% gap on a miss far exceeds typical Δw).
    Fix: per-ticker is_earnings_blocked check on dw>0 emissions."""

    def _cfg_with_earnings(self, buf=3):
        cfg = _qp_on()
        cfg["regime"] = {"earnings_buffer_days": buf}
        return cfg

    def test_blocks_top_up_within_buffer(self):
        """Held name with earnings tomorrow → QP top-up suppressed."""
        ctx = _Ctx(config=self._cfg_with_earnings(buf=3))
        # Earnings 2 days from "today" (2026-04-26) → within ±3-day buffer
        ctx.earnings_calendar = {"H": ["2026-04-28"]}
        ctx.holdings = {"H": _Hold(shares=10, mu=0.05, sigma=0.10)}
        ctx.prices = {"H": 100.0}
        ctx.cash = 9000.0
        ctx.portfolio_value = 10000.0
        JointPortfolioQPTask().run(ctx)
        assert ctx.orders == [], (
            f"earnings within buffer must suppress QP top-up; got "
            f"{len(ctx.orders)} order(s) — bug 4 reopened"
        )

    def test_allows_top_up_outside_buffer(self):
        """Held name with earnings 30 days out → QP top-up allowed."""
        ctx = _Ctx(config=self._cfg_with_earnings(buf=3))
        ctx.earnings_calendar = {"H": ["2026-05-26"]}  # +30d
        ctx.holdings = {"H": _Hold(shares=10, mu=0.10, sigma=0.10)}
        ctx.prices = {"H": 100.0}
        ctx.cash = 9000.0
        ctx.portfolio_value = 10000.0
        JointPortfolioQPTask().run(ctx)
        assert len(ctx.orders) >= 1, (
            "earnings outside buffer must NOT block top-up; got 0 orders"
        )

    def test_allows_sell_within_buffer(self):
        """Earnings blackout MUST NOT block sells — risk reduction is
        always allowed regardless of earnings proximity."""
        ctx = _Ctx(config=self._cfg_with_earnings(buf=3))
        ctx.earnings_calendar = {"H": ["2026-04-28"]}  # within buffer
        ctx.holdings = {"H": _Hold(shares=20, mu=-0.05, sigma=0.10)}
        ctx.prices = {"H": 100.0}
        ctx.cash = 8000.0
        ctx.portfolio_value = 10000.0
        JointPortfolioQPTask().run(ctx)
        assert len(ctx.exits) >= 1, (
            "earnings buffer must NOT block sells — sell remains valid "
            "for risk-reduction"
        )

    def test_no_earnings_calendar_no_op(self):
        """Empty earnings_calendar (e.g. data outage) must not silently
        block all top-ups — the task degrades gracefully."""
        ctx = _Ctx(config=self._cfg_with_earnings(buf=3))
        ctx.earnings_calendar = {}     # empty
        ctx.holdings = {"H": _Hold(shares=10, mu=0.10, sigma=0.10)}
        ctx.prices = {"H": 100.0}
        ctx.cash = 9000.0
        ctx.portfolio_value = 10000.0
        JointPortfolioQPTask().run(ctx)
        assert len(ctx.orders) >= 1, (
            "empty earnings calendar must not silently suppress top-ups"
        )


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
