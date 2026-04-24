"""TrimHeldTask — Kelly-driven partial sell on over-weight positions.

Sibling of TopUpHeldTask (adds to under-weight). The selection loop
caps NEW buys at `kelly_target_pct`, and TopUpHeldTask scales held
positions UP toward target. Neither trims a position DOWN when price
rallies drift current_pct above target, or when a retrain lowers
kelly_target_pct beneath the current weight.

Without this Task, over-weight positions stay heavy until a
stop/trail/model exit fully liquidates them — we never gently rebalance.

This Task runs after SelectionJob (alongside TopUpHeldTask). For each
held ticker not already touched this bar:

  current_pct  = shares * price / portfolio_value
  kelly_target = HoldingState.kelly_target_pct       (set by
                  PanelScoringJob::ApplyKellySizingTask)
  delta        = current_pct - kelly_target       # positive = over-weight

If delta > `ranking.kelly_sizing.trim_threshold` (default 0.10 hysteresis
to avoid daily churn), emit an ExitSignal(kelly_trim, quantity=Δshares)
into `ctx.exits`. The partial-sell infra (ExitSignal.quantity) shipped
earlier means the adapter's commit path sells exactly those shares and
leaves the remaining position open with cost basis + tenure preserved.

Aggressive mode: set `trim_threshold: 0.0` to trim to exact target every
bar.  Hysteresis (0.10 default) only trims when drift > 10 pct pts, which
the user's A/B feedback suggested was worth testing empirically.

Skipped in BEAR and during drawdown halt — rebalancing in risk-off is
counter-productive.
"""
from __future__ import annotations

import logging

from .context import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.trim")


class TrimHeldTask(Task):
    """Emit partial-sell ExitSignals for held positions whose current
    weight exceeds their Kelly target by `trim_threshold`."""

    def run(self, ctx: InferenceContext) -> bool | None:
        kelly_cfg = ctx.config.get("ranking", {}).get("kelly_sizing", {})
        if not kelly_cfg.get("enabled", False):
            return
        # AB-trim A/B (2026-04-24, 27-mo OOS):
        #   A_GOLDEN (trim_threshold=0.10 default)  25.09% APY
        #   B_hyst (explicit 0.10)                   25.09% APY
        #   C_tight (trim_threshold=0.0)             34.89% APY (+9.80 vs A)
        #   Pre-trim v4 golden                       37.82% APY
        # All three trim settings REGRESS vs pre-trim. The 0.10 default was
        # especially bad (-12.7 pts) — it trims AFTER big rallies and misses
        # continuation moves. Default switched to OFF via `trim_enabled=False`;
        # callers must opt in explicitly.
        if not bool(kelly_cfg.get("trim_enabled", False)):
            return
        trim_thresh = float(kelly_cfg.get("trim_threshold", 0.10))
        if trim_thresh < 0:
            return
        if ctx.bear_only or ctx.skip_buys:
            return   # no rebalancing during BEAR / drawdown halt

        portfolio = float(getattr(ctx, "portfolio_value", 0.0))
        if portfolio <= 0:
            return

        # Tickers already exiting or rotating this bar — don't trim on top.
        already_exiting = {getattr(e, "ticker", None)
                            for e in getattr(ctx, "exits", [])}
        rotation_sells  = {p.sell_ticker for p in (getattr(ctx, "rotations", []) or [])}
        already_buying  = {o.get("ticker") for o in getattr(ctx, "orders", [])
                            if isinstance(o, dict)}

        trimmed = 0
        for ticker, hs in ctx.holdings.items():
            if ticker in already_exiting or ticker in rotation_sells \
               or ticker in already_buying:
                continue
            kelly_target = getattr(hs, "kelly_target_pct", None)
            if kelly_target is None or kelly_target <= 0:
                continue

            price = ctx.prices.get(ticker)
            if price is None or price <= 0:
                continue

            current_shares = float(getattr(hs, "shares", 0.0))
            if current_shares <= 0:
                continue
            current_pct    = (current_shares * price) / portfolio
            delta          = current_pct - float(kelly_target)   # + = over-weight
            if delta <= trim_thresh:
                continue

            # Shares to sell: bring position back to kelly_target exactly.
            target_value  = float(kelly_target) * portfolio
            current_value = current_shares * price
            delta_value   = current_value - target_value
            trim_shares   = int(delta_value / price)
            if trim_shares < 1:
                continue
            # Never trim more than we hold (safety).
            trim_shares = min(trim_shares, int(current_shares))
            # If this would empty the position, it's a full exit — not a trim.
            # Skip and let sell paths handle it.
            if trim_shares >= int(current_shares):
                continue

            # Import lazily to mirror the rest of the pipeline.
            from kernel.exits import ExitSignal  # noqa: PLC0415
            sig = ExitSignal(
                should_exit = True,
                reason      = (f"kelly trim current={current_pct:.1%} "
                                f"target={kelly_target:.1%} delta={delta:.1%}"),
                exit_type   = "kelly_trim",
                quantity    = float(trim_shares),
            )
            ctx.exits.append((ticker, sig))
            trimmed += 1
            log.info(
                "TrimHeldTask: %s -%d shares (current=%.1f%% target=%.1f%% delta=%.1f%%)",
                ticker, trim_shares, current_pct * 100,
                float(kelly_target) * 100, delta * 100,
            )

        if trimmed:
            log.info("TrimHeldTask: emitted %d trim exit(s)", trimmed)
