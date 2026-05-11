"""L3 experiment — drawdown-triggered portfolio rebalance.

Implements Grossman & Zhou 1993, "Optimal Investment Strategies for
Controlling Drawdowns," Journal of Economic Dynamics and Control 19(2):
241-276. Their key equation (Eq. 8 in the original paper, reformulated
in Section 3):

    f*(DD_t) = f_Kelly × max(0, 1 - DD_t / DD_max)

where ``DD_t = (HWM - PV_t) / HWM`` is the current drawdown and
``DD_max`` is the investor's drawdown tolerance.

In a multi-name portfolio, applying this position-sizing rule means
scaling DOWN gross exposure as drawdown approaches the limit. Practically,
this maps to selling the weakest holdings (lowest cross-sectional panel
score) to free risk budget. This is the "trim the bottom of the book"
operationalization familiar from cvxportfolio's risk-aware optimizer
(Boyd et al, Cambridge 2024).

The new :class:`DrawdownRebalanceTask` runs AFTER :class:`SellJob` so it
sees per-ticker exits already emitted, and skips tickers already exiting.
It appends portfolio-level rebalance exits to ``ctx.exits`` for the
remainder of the chain (LimitSellsTask, JointActionsTask, ExecutionPipeline)
to process.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from kernel.exits import HoldingState
from kernel.pipeline.context import InferenceContext
from kernel.pipeline.task_drawdown_rebalance import DrawdownRebalanceTask


def _hs(panel_score, entry_price=100.0, entry_date=datetime.date(2024, 1, 1)):
    h = HoldingState(
        entry_price=entry_price,
        entry_date=entry_date,
        high_watermark=entry_price,
    )
    h.panel_score = panel_score
    return h


def _ctx(hwm=100_000.0, pv=100_000.0,
         holdings=None, dd_max=0.30, threshold=0.20,
         exits=None):
    cfg = {
        "risk": {
            "drawdown_rebalance": {
                "enabled": True,
                "trigger_drawdown": threshold,
                "max_drawdown": dd_max,
            }
        }
    }
    ctx = InferenceContext(config=cfg, today=datetime.date(2024, 6, 1))
    ctx.hwm = hwm
    ctx.portfolio_value = pv
    ctx.holdings = holdings or {}
    ctx.exits = exits or []
    return ctx


class TestDrawdownRebalanceNoFire:
    """DD below trigger → task is a no-op."""

    def test_dd_below_trigger_emits_nothing(self):
        ctx = _ctx(hwm=100_000.0, pv=90_000.0,  # 10% DD
                   holdings={"AAPL": _hs(0.5), "MSFT": _hs(0.4)})
        DrawdownRebalanceTask().run(ctx)
        assert ctx.exits == []

    def test_disabled_in_config_skips(self):
        ctx = _ctx(hwm=100_000.0, pv=70_000.0,
                   holdings={"AAPL": _hs(0.5)})
        ctx.config["risk"]["drawdown_rebalance"]["enabled"] = False
        DrawdownRebalanceTask().run(ctx)
        assert ctx.exits == []

    def test_zero_hwm_or_pv_safe_noop(self):
        """A pre-warm-up ctx (hwm=0) must not trigger NaN division."""
        ctx = _ctx(hwm=0.0, pv=0.0, holdings={"AAPL": _hs(0.5)})
        DrawdownRebalanceTask().run(ctx)
        assert ctx.exits == []

    def test_no_holdings_safe_noop(self):
        ctx = _ctx(hwm=100_000.0, pv=70_000.0, holdings={})
        DrawdownRebalanceTask().run(ctx)
        assert ctx.exits == []


class TestDrawdownRebalanceFires:
    """DD above trigger → liquidate weakest by panel_score."""

    def test_dd_30pct_kelly_scaling_down(self):
        # DD = 30%, dd_max = 0.30, trigger = 0.20 → f_kelly = 1 - 30/30 = 0
        # → liquidate ALL holdings (Kelly fraction collapsed to 0)
        ctx = _ctx(
            hwm=100_000.0, pv=70_000.0,  # 30% DD
            holdings={
                "AAPL": _hs(0.8),
                "MSFT": _hs(0.6),
                "GOOG": _hs(0.4),
                "AMZN": _hs(0.2),
            },
            dd_max=0.30,
            threshold=0.20,
        )
        DrawdownRebalanceTask().run(ctx)
        # f_kelly = max(0, 1 - 0.30/0.30) = 0 → target_count = 0 → exit all 4
        assert len(ctx.exits) == 4
        exited_tickers = {t for t, _ in ctx.exits}
        assert exited_tickers == {"AAPL", "MSFT", "GOOG", "AMZN"}

    def test_dd_25pct_partial_liquidation_weakest_first(self):
        # DD = 25%, dd_max = 0.30 → f_kelly = 1 - 25/30 ≈ 0.167
        # 4 holdings × 0.167 = target ≈ 0 → liquidate 4 (rounded down)
        # Use dd_max=0.50 to get a meaningful partial cut:
        # f_kelly = 1 - 0.25/0.50 = 0.50 → target = 4 × 0.50 = 2 → liquidate 2 weakest
        ctx = _ctx(
            hwm=100_000.0, pv=75_000.0,  # 25% DD
            holdings={
                "AAPL": _hs(0.8),  # strongest
                "MSFT": _hs(0.6),
                "GOOG": _hs(0.4),
                "AMZN": _hs(0.2),  # weakest
            },
            dd_max=0.50,
            threshold=0.20,
        )
        DrawdownRebalanceTask().run(ctx)
        # target = 2 → liquidate 2 weakest (AMZN=0.2, GOOG=0.4)
        assert len(ctx.exits) == 2
        exited_tickers = {t for t, _ in ctx.exits}
        assert exited_tickers == {"AMZN", "GOOG"}, \
            "Liquidation should target weakest panel_score names"

    def test_exit_signal_metadata_is_correct(self):
        ctx = _ctx(
            hwm=100_000.0, pv=70_000.0,
            holdings={"AAPL": _hs(0.3)},
            dd_max=0.30,
            threshold=0.20,
        )
        DrawdownRebalanceTask().run(ctx)
        assert len(ctx.exits) == 1
        ticker, sig = ctx.exits[0]
        assert ticker == "AAPL"
        assert sig.should_exit is True
        assert sig.exit_type == "drawdown_rebalance"
        assert sig.quantity is None   # full liquidate
        # Reason should reference the drawdown ratio
        assert "DD=" in sig.reason or "drawdown" in sig.reason.lower()

    def test_already_exiting_ticker_not_re_emitted(self):
        from kernel.exits import ExitSignal
        existing = ExitSignal(
            should_exit=True, reason="stop_loss fired",
            exit_type="stop_loss", quantity=None,
        )
        ctx = _ctx(
            hwm=100_000.0, pv=70_000.0,
            holdings={
                "AAPL": _hs(0.8),
                "MSFT": _hs(0.2),  # weakest, but already exiting
            },
            dd_max=0.50,
            threshold=0.20,
            exits=[("MSFT", existing)],
        )
        DrawdownRebalanceTask().run(ctx)
        # f_kelly = 0.5 → target = 1, need to liquidate 1
        # MSFT is weakest but already in exits → AAPL gets liquidated instead?
        # Actually no — MSFT is already counted as "exiting" so we treat it as
        # already removed; remaining open positions = {AAPL}. target=0.5 of
        # 1 ≈ 0 → liquidate AAPL too. But user expectation: don't double-emit
        # MSFT.
        msft_emissions = [t for t, _ in ctx.exits if t == "MSFT"]
        assert len(msft_emissions) == 1, "MSFT not re-emitted"

    def test_holdings_without_panel_score_treated_as_weakest(self):
        """Defensive: a HoldingState with panel_score=None is treated as
        the weakest (priority for liquidation). Otherwise we'd silently
        keep junk positions during a drawdown."""
        ctx = _ctx(
            hwm=100_000.0, pv=75_000.0,
            holdings={
                "AAPL": _hs(0.8),     # strong
                "MSFT": _hs(None),    # unknown → treated weakest
                "GOOG": _hs(0.5),
            },
            dd_max=0.50,
            threshold=0.20,
        )
        DrawdownRebalanceTask().run(ctx)
        # f_kelly = 0.5 → target = int(3 × 0.5) = 1 → liquidate 2
        # MSFT (None) + GOOG (0.5) should be liquidated; AAPL (0.8) kept
        exited = {t for t, _ in ctx.exits}
        assert "MSFT" in exited
        assert "GOOG" in exited
        assert "AAPL" not in exited


class TestDrawdownRebalanceRegressionGuards:
    """AUDIT REGRESSION GUARDS: pin the Grossman-Zhou behavioural invariants."""

    def test_grossman_zhou_eq8_kelly_scaling(self):
        """For DD ∈ [trigger, dd_max], the fraction f_kelly should
        decrease LINEARLY from 1 toward 0. This pins the Eq. 8 mapping."""
        from kernel.pipeline.task_drawdown_rebalance import (
            compute_kelly_scaling,
        )
        assert compute_kelly_scaling(0.0, dd_max=0.30) == 1.0
        assert compute_kelly_scaling(0.15, dd_max=0.30) == 0.5
        assert compute_kelly_scaling(0.30, dd_max=0.30) == 0.0
        assert compute_kelly_scaling(0.45, dd_max=0.30) == 0.0  # clipped
        assert compute_kelly_scaling(-0.10, dd_max=0.30) == 1.0  # negative DD → no scale

    def test_no_emit_on_nonfinite_pv(self):
        """NaN / inf portfolio_value must NOT trigger a flood of exits.
        §5.13.11 NaN-guard discipline."""
        ctx = _ctx(hwm=100_000.0, pv=float("nan"),
                   holdings={"AAPL": _hs(0.5)})
        DrawdownRebalanceTask().run(ctx)
        assert ctx.exits == []
