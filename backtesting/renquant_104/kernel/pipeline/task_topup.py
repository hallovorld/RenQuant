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

from .context import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.topup")


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
        if ctx.bear_only or ctx.skip_buys:
            return   # don't add during BEAR / halt

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
        # Audit fix TU-1..TU-4 (Round 2 deep audit, 2026-04-25): pre-fix,
        # NaN kelly_target / price / portfolio slipped past guards
        # (`<= 0` is False on NaN), then propagated through delta calc,
        # producing `delta < threshold = False` → silent skip with NO
        # log signal. Operator couldn't see WHY a holding wasn't being
        # topped up.
        import math
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

            # Bug 26 fix (2026-04-24): cap by available cash. Without this
            # check, TopUp emits orders that the adapter's _apply_buy then
            # rejects with "insufficient cash" warnings — wastes ctx.orders
            # space and pollutes audit logs. The panel value is a notional
            # weight target; actual buy must come from real cash.
            cash = float(getattr(ctx, "cash", 0.0))
            # TU-4 guard: NaN cash → treat as 0 to be safe (would cause
            # the down-sizing branch to fail otherwise).
            if not math.isfinite(cash):
                continue
            invest     = extra_shares * price
            if invest > cash:
                # Re-size down to available cash (whole shares only)
                affordable_shares = int(cash // price)
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
            ctx.orders.append({
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
            })
            added += 1
            log.info(
                "TopUpHeldTask: %s +%d shares (current=%.1f%% target=%.1f%% delta=%.1f%%)",
                ticker, extra_shares, current_pct * 100,
                kelly_target * 100, delta * 100,
            )

        if added:
            log.info("TopUpHeldTask: emitted %d top-up order(s)", added)
