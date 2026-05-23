"""TopUpHeldTask — Kelly-driven add-to-existing-position.

Plan C + AB (2026-04-23 evening):

The selection loop + rotation only handle NEW buys and 1:1 swaps.
Neither can *add* to an already-held position whose Kelly target
exceeds its current weight. Without this Task, a held ticker whose
calibrated score spikes (stronger edge) stays stuck at the weight we
entered at.

This Task runs after SelectionJob. For each held ticker not already
in the current bar's orders or exits:

  current_pct  = shares * price / portfolio_value
  kelly_target = HoldingState.kelly_target_pct       (set by
                  PanelScoringJob::ApplyScoresTask)
  delta        = kelly_target - current_pct

If delta > `ranking.kelly_sizing.top_up_threshold` (default 0.05), emit
an extra BUY order of floor(delta * portfolio / price) shares into
`ctx.orders` so adapter.commit ships it to the broker.

This is additive and non-destructive: never sells, only tops up. Trim
(reduce over-weight positions) is a separate Task (TrimHeldTask)
which requires the partial-sell path and is scoped for the next
session.
"""
from __future__ import annotations

import logging
import math

from .context import InferenceContext
from .order_attribution import stamp_order_attribution
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.topup")


def _pending_buy_invest(orders: list | None) -> float:
    """Cash already reserved by buy orders emitted earlier this bar."""
    total = 0.0
    for order in orders or []:
        if not isinstance(order, dict):
            continue
        side = str(order.get("side") or order.get("action") or "BUY").upper()
        if side == "SELL":
            continue
        invest = order.get("invest")
        try:
            invest_f = float(invest) if invest is not None else float("nan")
        except (TypeError, ValueError):
            invest_f = float("nan")
        if math.isfinite(invest_f) and invest_f > 0:
            total += invest_f
            continue
        try:
            shares = float(order.get("shares", 0.0))
            price = float(order.get("price", 0.0))
        except (TypeError, ValueError):
            continue
        notional = shares * price
        if shares > 0 and math.isfinite(notional) and notional > 0:
            total += notional
    return total


class TopUpHeldTask(Task):
    """Emit additional BUY orders for held positions whose Kelly target
    exceeds their current weight by `top_up_threshold`."""

    def run(self, ctx: InferenceContext) -> bool | None:
        kelly_cfg = ctx.config.get("ranking", {}).get("kelly_sizing", {})
        if not kelly_cfg.get("enabled", False):
            return
        top_up_thresh = float(kelly_cfg.get("top_up_threshold", 0.05))
        if top_up_thresh <= 0:
            return
        # 2026-05-04 audit Issue 39 fix: TopUp must also respect
        # ctx.buy_blocked — set by macro gates (EMA50Gate, VelocityCrash,
        # DrawdownGate). Pre-fix, when SPY went below its 50-day EMA,
        # NEW buys were correctly blocked but TopUps kept adding to
        # existing positions — violating the macro gate's intent. TopUp
        # is a buy (cash debit + share increment), should respect every
        # buy-side macro gate.
        if ctx.bear_only or ctx.skip_buys or getattr(ctx, "buy_blocked", False):
            return   # don't add during BEAR / halt / macro-block

        portfolio = float(getattr(ctx, "portfolio_value", 0.0))
        if portfolio <= 0:
            return

        # Tickers already touched this bar — don't add on top of them.
        # Production ctx.exits is list[(ticker, ExitSignal)]; some tests
        # still pass list[SimpleNamespace] / list[ExitSignal-like]. Be
        # tolerant of both shapes.
        already_buying = {o.get("ticker") for o in getattr(ctx, "orders", [])
                          if isinstance(o, dict)}
        already_selling: set = set()
        for e in (getattr(ctx, "exits", []) or []):
            if isinstance(e, tuple) and len(e) == 2:
                already_selling.add(e[0])
            else:
                t = getattr(e, "ticker", None)
                if t is not None:
                    already_selling.add(t)
        rotation_sells = {p.sell_ticker for p in (getattr(ctx, "rotations", []) or [])}

        added = 0
        # Shared buy budget: Selection/QP/rotation may already have queued
        # buy orders this bar. TopUp must consume only the unreserved cash,
        # then decrement it after each emitted top-up so live never relies
        # on broker rejects as the budget check.
        cash = float(getattr(ctx, "cash", 0.0))
        if not math.isfinite(cash):
            return
        try:
            from kernel.regime import confidence_to_size_multiplier  # noqa: PLC0415
            conf_mult = confidence_to_size_multiplier(ctx.confidence)
        except Exception:  # pragma: no cover - defensive for legacy test doubles
            conf_mult = 1.0
        regime_p = (
            ctx.config.get("regime_params", {}).get(ctx.regime, {})
            if isinstance(getattr(ctx, "config", None), dict) else {}
        )
        reserve_pct = float(regime_p.get("cash_reserve_pct", 0.0)) * conf_mult
        reserve_cash = max(portfolio * reserve_pct, 0.0)
        pending_buy_cash = _pending_buy_invest(getattr(ctx, "orders", []))
        available_cash = max(cash - pending_buy_cash - reserve_cash, 0.0)

        # 2026-05-01 trade-audit fix: TopUp must respect the same earnings
        # blackout the buy-side EarningsFilterTask enforces. Pre-fix, FTNT
        # was topped up on 2026-04-29 — one day before its 2026-04-30
        # earnings print — because TopUp ran on the held set, not the
        # candidate pipeline. Symmetric (±buffer) — adding to a position is
        # entering, and entry must respect event windows.
        # Guarded: if ctx lacks today / earnings_calendar / config, fall
        # through silently (legacy SimpleNamespace tests that don't model
        # earnings inputs still get baseline TopUp behavior).
        from kernel.selection import is_earnings_blocked  # noqa: PLC0415
        earnings_calendar = getattr(ctx, "earnings_calendar", None) or {}
        today = getattr(ctx, "today", None)
        cfg_for_buf = getattr(ctx, "config", None) or {}
        earnings_buf = int(
            (cfg_for_buf.get("regime", {}) if isinstance(cfg_for_buf, dict) else {})
            .get("earnings_buffer_days", 3)
        )
        earnings_check_active = bool(earnings_calendar) and today is not None

        # Conviction floor on TopUp (2026-05-01 trade-audit response):
        # Pre-fix, TopUp added shares to held positions whenever Kelly
        # target > current weight, regardless of whether the panel ranker
        # still liked the holding. Result: 4 of 7 buys 2026-04-29 → 05-01
        # had rank_score=0.0 — TopUp was blindly Kelly-maintaining held
        # positions while the panel had no current opinion on them.
        # Invariant: TopUp only adds to a holding whose latest calibrated
        # rank_score is at or above `topup_conviction_floor` (default 0.20
        # — same level as the panel-conviction sell floor). When the
        # panel hasn't scored the holding yet (None / NaN), fail-CLOSED:
        # don't add. This gate sits BEFORE the Kelly-delta math so
        # TopUp can only act when both the model AND Kelly agree.
        topup_floor = float(kelly_cfg.get("topup_conviction_floor", 0.20))

        for ticker, hs in ctx.holdings.items():
            if ticker in already_buying or ticker in already_selling \
               or ticker in rotation_sells:
                continue
            if earnings_check_active and is_earnings_blocked(
                    ticker, today, earnings_calendar, earnings_buf):
                log.info(
                    "TopUpHeldTask [%s]: SKIPPED — within ±%d days of earnings",
                    ticker, earnings_buf,
                )
                continue
            # Conviction floor — fail-closed on missing/low rank.
            if topup_floor > 0:
                hs_rank = getattr(hs, "rank_score", None)
                if hs_rank is None or not math.isfinite(hs_rank) \
                   or hs_rank < topup_floor:
                    log.info(
                        "TopUpHeldTask [%s]: SKIPPED — conviction floor "
                        "(rank_score=%s < floor=%.2f)",
                        ticker, hs_rank, topup_floor,
                    )
                    continue
            kelly_target = getattr(hs, "kelly_target_pct", None)
            if kelly_target is None or not math.isfinite(kelly_target) or kelly_target <= 0:
                continue

            price = ctx.prices.get(ticker)
            if price is None or not math.isfinite(price) or price <= 0:
                continue

            if not math.isfinite(portfolio) or portfolio <= 0:
                continue   # zero/NaN portfolio → nothing to top up against
            current_shares = float(getattr(hs, "shares", 0.0))
            current_pct    = (current_shares * price) / portfolio
            delta          = float(kelly_target) - current_pct
            if delta < top_up_thresh:
                continue

            # Multi-entry accumulation — cap top-up delta at per_session_buy_cap.
            per_session_cap = kelly_cfg.get("per_session_buy_cap")
            bought_delta = delta
            if per_session_cap is not None:
                cap = float(per_session_cap)
                if cap > 0 and bought_delta > cap:
                    bought_delta = cap

            extra_shares = int(bought_delta * portfolio / price)
            if extra_shares < 1:
                continue

            invest     = extra_shares * price
            if invest > available_cash:
                # Re-size down to available cash (whole shares only)
                affordable_shares = int(available_cash // price)
                if affordable_shares < 1:
                    continue
                extra_shares = affordable_shares
                invest = extra_shares * price
            # Audit fix (2026-04-24): use actual bought delta, not the
            # uncapped Kelly delta. When per_session_buy_cap or cash
            # constraint trims the order, the recorded target_pct must
            # reflect the post-fill weight, not the abstract Kelly target.
            actual_delta = (extra_shares * price) / portfolio if portfolio > 0 else 0.0
            target_pct = (current_pct + actual_delta)
            ctx.orders.append(stamp_order_attribution({
                "ticker":      ticker,
                "shares":      extra_shares,
                "price":       price,
                "invest":      invest,
                "target_pct":  target_pct,
                "regime":      ctx.regime,
                "confidence":  ctx.confidence,
                "conviction":  1.0,
                "sigma_mult":  1.0,
                "rank_score":  float(getattr(hs, "rank_score",  0.0) or 0.0),
                "rs_score":    0.0,
                "panel_score": getattr(hs, "panel_score", None),
                "sigma":       getattr(hs, "sigma", None),
                "mu":          getattr(hs, "mu",    None),
                "detail":      "top_up_kelly",
                "order_type":  "TOP_UP",
            }, ctx=ctx, source_job="TopUpJob",
                source_task="TopUpHeldTask",
                acceptance_reason="kelly_target_above_current_weight",
                source_obj=hs,
                decision_inputs={
                    "current_pct": current_pct,
                    "kelly_target_pct": float(kelly_target),
                    "delta_pct": delta,
                    "top_up_threshold": top_up_thresh,
                    "topup_conviction_floor": topup_floor,
                    "cash_before": cash,
                    "pending_buy_cash": pending_buy_cash,
                    "reserve_cash": reserve_cash,
                    "available_cash_before": available_cash,
                }))
            available_cash -= invest
            added += 1
            log.info(
                "TopUpHeldTask: %s +%d shares (current=%.1f%% target=%.1f%% "
                "delta=%.1f%% remaining_cash=$%.0f)",
                ticker, extra_shares, current_pct * 100,
                kelly_target * 100, delta * 100, available_cash,
            )

        if added:
            log.info("TopUpHeldTask: emitted %d top-up order(s)", added)
