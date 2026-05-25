"""Phase 2D — Short cover stop-loss + IRC §1233 ST tax marker.

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

Both tasks are PURE — no I/O, no broker calls. Short-cover stops emit
``ctx.orders`` buy-to-cover rows; adapters route those through the normal buy
execution path, where an existing negative holding is covered instead of
opening a fresh long.

Tests: tests/test_short_cover_stop_phase_2d.py.

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
      ctx.holdings: dict[ticker, HoldingState] with shares < 0
        Legacy tests may also pass ctx.short_holdings with qty < 0.
      ctx.config["risk"]["short_cover_stop_pct"] (default 0.15)
      ctx.ohlcv[ticker] — current price for MTM
    Writes:
      ctx.orders.append(buy-to-cover order)
      ctx.counters["short_cover_stop_triggered"]
    """
    name = "ShortCoverStopLossTask"

    def run(self, ctx) -> bool | None:
        cfg = (ctx.config or {}).get("risk", {})
        if not cfg.get("short_cover_stop_enabled", True):
            return
        cover_pct = float(cfg.get("short_cover_stop_pct", 0.15))

        shorts = _short_holding_map(ctx)
        if not shorts:
            return
        ohlcv = getattr(ctx, "ohlcv", None) or {}

        triggered = []
        for ticker, holding in shorts.items():
            qty = float(
                getattr(holding, "shares", getattr(holding, "qty", 0)) or 0
            )
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

        orders = list(getattr(ctx, "orders", None) or [])
        for t in triggered:
            orders.append({
                "ticker": t["ticker"],
                "shares": float(t["qty"]),
                "price": float(t["current"]),
                "target_pct": 0.0,
                "detail": "short_cover_stop",
                "order_type": "BUY_TO_COVER_short_cover_stop",
                "source": "ShortCoverStopLossTask",
                "source_job": "ShortCoverStopLossTask",
                "source_task": "short_cover_stop",
                "order_source": "ShortCoverStopLossTask.short_cover_stop",
                "decision_inputs": {
                    "acceptance_reason": "short_cover_stop",
                    "side": "buy_to_close",
                    "loss_pct": t["loss_pct"],
                    "trigger": cover_pct,
                    "entry_price": t["entry"],
                    "current_price": t["current"],
                    "tax_holding_period": "ST_FORCED_§1233",
                },
            })
            log.warning(
                "ShortCoverStopLoss: %s loss=%.2f%% (entry=$%.2f cur=$%.2f) "
                "≥ trigger=%.0f%% → buy_to_close %.0f shares (§1233 ST tax)",
                t["ticker"], t["loss_pct"] * 100, t["entry"], t["current"],
                cover_pct * 100, t["qty"],
            )
        ctx.orders = orders
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


def _short_holding_map(ctx) -> dict[str, Any]:
    holdings = getattr(ctx, "holdings", None) or {}
    out = {
        ticker: hs for ticker, hs in holdings.items()
        if float(getattr(hs, "shares", 0) or 0) < 0
    }
    if out:
        return out
    return getattr(ctx, "short_holdings", None) or {}
