"""Phase 2D — Short cover stop-loss + IRC §1233 ST tax marker.

NOT YET WIRED into InferencePipeline. Ships as ready-to-use code. The
2026-05-15 long-short Phase 2A/2B + paper-account broker isolation are
in place; this module unlocks Phase 2D (operator decision required to
wire into inference pipeline + flip shorts ON).

Two tasks:

  ShortCoverStopLossTask
    Symmetric counterpart to existing long stop-loss. A SHORT position
    loses money when the underlying RISES. Trigger: cover when
    realized loss on the short exceeds `cover_stop_pct` (default 15%
    of entry price, matching long stop_loss_pct semantics).

    Loss math for shorts:
        short_pnl = (entry_price - current_price) × qty_short
        loss_pct  = (current_price - entry_price) / entry_price
                    (positive = LOSS for the short)

    Triggers a `buy_to_close` order with reason="short_cover_stop".

  IRC1233TaxMarkerTask
    IRC §1233 ST: short-sale gains/losses are SHORT-TERM regardless of
    holding period (no LT-cap-gains preferential rate ever applies to
    a short). Stamps `cover_taxlot.holding_period = "ST_FORCED_§1233"`
    on every short-cover trade so the realized-PnL ledger reports
    correctly. Reporting-only — no algorithmic effect.

Both tasks are PURE — no I/O, no broker calls. They emit ExitSignal
records that the standard execution path (ExecuteExitsTask) processes.

Tests: tests/test_short_cover_stop_phase_2d.py.

When wired into InferencePipeline (next session), insert AFTER
TickerSellJob (long sells already settled) and BEFORE JointActionJob
(so cover orders compete for capital alongside new buys).

References:
  IRC §1233(a) — character of gain on short sale (always ST)
  Hong-Stein 2003 — short squeeze risk in mean-reversion regimes
"""
from __future__ import annotations

import logging
import math
from typing import Any

from kernel.pipeline.pipeline import Task

log = logging.getLogger("kernel.pipeline.short_cover")


# ── Phase 2D-1: cover stop-loss ─────────────────────────────────────────────


class ShortCoverStopLossTask(Task):
    """Trigger buy_to_close on short positions whose mark-to-market
    loss exceeds `cover_stop_pct` of entry price.

    Reads:
      ctx.short_holdings: dict[ticker, ShortHoldingState] — keys: qty (negative),
        entry_price, regime_at_entry, days_held
      ctx.config["risk"]["short_cover_stop_pct"] (default 0.15)
      ctx.ohlcv[ticker] — current price for MTM
    Writes:
      ctx.exits.append((ticker, ExitSignal(reason="short_cover_stop", ...)))
      ctx.counters["short_cover_stop_triggered"]
    """
    name = "ShortCoverStopLossTask"

    def run(self, ctx) -> bool | None:
        cfg = (ctx.config or {}).get("risk", {})
        if not cfg.get("short_cover_stop_enabled", True):
            return
        cover_pct = float(cfg.get("short_cover_stop_pct", 0.15))

        shorts = getattr(ctx, "short_holdings", None) or {}
        if not shorts:
            return
        ohlcv = getattr(ctx, "ohlcv", None) or {}

        triggered = []
        for ticker, holding in shorts.items():
            qty = float(getattr(holding, "qty", 0))
            if qty >= 0:  # not a short
                continue
            entry = float(getattr(holding, "entry_price", 0))
            if entry <= 0 or not math.isfinite(entry):
                continue
            df = ohlcv.get(ticker)
            if df is None or "close" not in getattr(df, "columns", []):
                continue
            try:
                current = float(df["close"].iloc[-1])
            except (IndexError, ValueError, TypeError):
                continue
            if not math.isfinite(current) or current <= 0:
                continue
            # Loss for short = (current - entry) / entry > 0 means LOSING
            loss_pct = (current - entry) / entry
            if loss_pct >= cover_pct:
                triggered.append({
                    "ticker": ticker,
                    "qty": -qty,  # buy_to_close needs POSITIVE qty
                    "entry": entry,
                    "current": current,
                    "loss_pct": loss_pct,
                })

        if not triggered:
            return

        # Emit cover orders. The exit_type "short_cover_stop" is novel —
        # downstream ExecuteExitsTask must route to buy_to_close (not
        # the long sell path).
        from collections import namedtuple
        ExitSignal = namedtuple("ExitSignal", ["reason", "qty", "details"])
        exits = list(getattr(ctx, "exits", None) or [])
        for t in triggered:
            sig = ExitSignal(
                reason="short_cover_stop",
                qty=t["qty"],
                details={
                    "side": "buy_to_close",
                    "loss_pct": t["loss_pct"],
                    "trigger": cover_pct,
                    "entry_price": t["entry"],
                    "current_price": t["current"],
                    "tax_holding_period": "ST_FORCED_§1233",  # always ST for shorts
                },
            )
            exits.append((t["ticker"], sig))
            log.warning(
                "ShortCoverStopLoss: %s loss=%.2f%% (entry=$%.2f cur=$%.2f) "
                "≥ trigger=%.0f%% → buy_to_close %.0f shares (§1233 ST tax)",
                t["ticker"], t["loss_pct"] * 100, t["entry"], t["current"],
                cover_pct * 100, t["qty"],
            )
        ctx.exits = exits
        ctx.counters = getattr(ctx, "counters", None) or {}
        ctx.counters["short_cover_stop_triggered"] = (
            ctx.counters.get("short_cover_stop_triggered", 0) + len(triggered)
        )


# ── Phase 2D-2: IRC §1233 tax marker ────────────────────────────────────────


class IRC1233TaxMarkerTask(Task):
    """Stamp `tax_holding_period = "ST_FORCED_§1233"` on every short-cover
    fill in the realized-PnL ledger.

    IRC §1233(a) requires that gain/loss on closing a short sale is
    ALWAYS short-term, regardless of how long the position was held.
    No long-term capital-gains preferential rate ever applies to a
    short. This is reporting-only.

    Reads:
      ctx.realized_trades: list of {ticker, side, ...} — emitted by
        ExecuteExitsTask post-fill
    Writes:
      ctx.realized_trades — adds tax_holding_period field on shorts
      ctx.counters["irc_1233_marker_applied"]
    """
    name = "IRC1233TaxMarkerTask"

    def run(self, ctx) -> bool | None:
        if not (ctx.config or {}).get("tax", {}).get("irc_1233_marker_enabled", True):
            return
        trades = getattr(ctx, "realized_trades", None) or []
        if not trades:
            return
        n = 0
        for t in trades:
            # Identify a short cover: side=buy AND position_intent contains
            # 'close' AND the underlying realized_pnl was on a negative qty.
            side = (t.get("side") or "").lower()
            intent = (t.get("position_intent") or "").lower()
            if side == "buy" and "close" in intent:
                t["tax_holding_period"] = "ST_FORCED_§1233"
                n += 1
        if n:
            ctx.counters = getattr(ctx, "counters", None) or {}
            ctx.counters["irc_1233_marker_applied"] = (
                ctx.counters.get("irc_1233_marker_applied", 0) + n
            )
            log.info("IRC1233TaxMarker: stamped %d short-cover trades as ST", n)


__all__ = ["ShortCoverStopLossTask", "IRC1233TaxMarkerTask"]
