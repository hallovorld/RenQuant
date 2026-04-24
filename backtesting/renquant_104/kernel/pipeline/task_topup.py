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
        already_buying = {o.get("ticker") for o in getattr(ctx, "orders", [])
                          if isinstance(o, dict)}
        already_selling = {getattr(e, "ticker", None)
                            for e in getattr(ctx, "exits", [])}
        rotation_sells = {p.sell_ticker for p in (getattr(ctx, "rotations", []) or [])}

        added = 0
        for ticker, hs in ctx.holdings.items():
            if ticker in already_buying or ticker in already_selling \
               or ticker in rotation_sells:
                continue
            kelly_target = getattr(hs, "kelly_target_pct", None)
            if kelly_target is None or kelly_target <= 0:
                continue

            price = ctx.prices.get(ticker)
            if price is None or price <= 0:
                continue

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
            target_pct = (current_pct + delta)
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
