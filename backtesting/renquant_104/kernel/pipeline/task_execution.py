"""Tasks composing the :class:`ExecutionPipeline`.

Per CLAUDE.md §1c each Task is ≤50 line body, single-responsibility, and
mutates well-documented ctx fields. The set:

============================  =====================================================
Task                          Responsibility
============================  =====================================================
:class:`PrepareExecutionTask` clear ctx.fills (transient per-bar accumulator)
:class:`DedupeExitsTask`      ctx.exits → de-duped by ticker (full liquidate wins)
:class:`DedupeBuysTask`       ctx.orders → de-duped by ticker (first-write-wins)
:class:`ExecuteExitsTask`     for each (ticker, sig), call backend, append Fill
:class:`StampWashSaleTask`    SELL fills → ctx.last_sell_dates / last_stop_exit_dates
:class:`PruneFullExitsTask`   full-liquidate tickers popped from ctx.holdings
:class:`ExecuteBuysTask`      for each order, call backend, append Fill
:class:`UpsertHoldingsTask`   BUY fills → new HoldingState OR volume-weighted topup
============================  =====================================================

Together they replace the 150+-line bodies of ``adapters/sim.py:commit``,
``adapters/runner.py:commit``, and ``adapters/lean.py:commit``. Each Task
documents the **invariant it pins** so a future ``--no-verify`` skip
during a hot patch can't silently delete a defensive guard.
"""
from __future__ import annotations

import logging
import math

import pandas as pd

from kernel.execution import OrderIntent, OrderSide
from kernel.exits import HoldingState
from kernel.pipeline.pipeline import Task
from kernel.pipeline.task_post_stop_cooldown import DEFAULT_STOP_EXIT_TYPES

log = logging.getLogger("kernel.pipeline.execution")


def is_full_liquidate_signal(sig, held_qty: float | None = None) -> bool:
    """ExitSignal.quantity semantics:

    - None  → full
    - ≤ 0   → full (defensive: caller bug)
    - NaN   → full (§5.13.11)
    - ≥ held quantity → full when held_qty is supplied
    - > 0   → partial (caller must guarantee qty < current shares)
    """
    q = getattr(sig, "quantity", None)
    if q is None:
        return True
    try:
        qf = float(q)
    except (TypeError, ValueError):
        return True
    if not math.isfinite(qf) or qf <= 0:
        return True
    if held_qty is not None:
        try:
            held = float(held_qty)
        except (TypeError, ValueError):
            held = 0.0
        if math.isfinite(held) and held > 0 and qf >= held:
            return True
    return False


def _is_full_liquidate(sig) -> bool:
    """Back-compat wrapper for older tests/imports."""
    return is_full_liquidate_signal(sig)


def _require_backend(ctx):
    if ctx.execution_backend is None:
        raise ValueError(
            "ExecutionPipeline.run requires ctx.execution_backend to be set; "
            "got None. Adapters must attach an ExecutionBackend before calling."
        )
    return ctx.execution_backend


# ─── Prep ──────────────────────────────────────────────────────────────────


class PrepareExecutionTask(Task):
    """Clear transient per-bar accumulators.

    Pins: a stale ``ctx.fills`` from the prior bar must not survive into
    this one. Without this, an adapter that forgets to reset ctx between
    bars would double-count fills.
    """

    def run(self, ctx) -> "bool | None":
        _require_backend(ctx)
        ctx.fills = []
        return True


# ─── Exit-side dedupe + execute ────────────────────────────────────────────


class DedupeExitsTask(Task):
    """Collapse duplicate ``ctx.exits`` rows per ticker (full liquidate wins).

    Pins: the partial-vs-full priority that ``sim.commit():628-642`` already
    encodes. A misbehaving upstream Job emitting both a ``kelly_trim`` (partial)
    and a ``stop_loss`` (full) on the same ticker must collapse to the full
    liquidate — otherwise we'd cash out partial then fail to close the rest.
    """

    def run(self, ctx) -> "bool | None":
        seen: dict[str, tuple] = {}
        for ticker, sig in (ctx.exits or []):
            existing = seen.get(ticker)
            if existing is None:
                seen[ticker] = (ticker, sig)
                continue
            ex_full = _is_full_liquidate(existing[1])
            new_full = _is_full_liquidate(sig)
            if new_full and not ex_full:
                seen[ticker] = (ticker, sig)
            # otherwise first-write-wins
        ctx.exits = list(seen.values())
        return True


class ExecuteExitsTask(Task):
    """Place exit orders via the backend; record Fills to ctx.fills.

    Pins: SELL intents for unheld tickers MUST be dropped (not raised) so
    a stale rotation pair pointing at a ticker the broker has already
    closed (post-reconciliation race) doesn't crash the entire bar.
    Matches ``sim._apply_sell:781`` guard.
    """

    def run(self, ctx) -> "bool | None":
        backend = _require_backend(ctx)
        today = pd.Timestamp(ctx.today)
        for ticker, sig in (ctx.exits or []):
            held_qty = backend.get_position_quantity(ticker)
            if held_qty <= 0:
                log.warning(
                    "ExecuteExitsTask: SELL for %s but backend reports no "
                    "position; dropping intent (reason=%s)",
                    ticker, getattr(sig, "reason", "?"),
                )
                continue
            full = is_full_liquidate_signal(sig, held_qty)
            shares = None if full else int(float(sig.quantity))
            intent = OrderIntent(
                ticker=ticker, side=OrderSide.SELL,
                shares=shares, target_pct=0.0,
                today=today,
                reason=getattr(sig, "reason", "") or "exit",
                exit_type=getattr(sig, "exit_type", None) or "model_sell",
            )
            fill = backend.place_market_order(intent)
            ctx.fills.append(fill)
        return True


class StampWashSaleTask(Task):
    """Stamp ctx.last_sell_dates + ctx.last_stop_exit_dates from SELL fills.

    Pins: only **full liquidates** stamp ``last_sell_dates`` (matches the
    2026-04-24 partial-trim wash-sale exemption); path-rule exits stamp
    ``last_stop_exit_dates`` regardless of partial-vs-full (G8 invariant).
    """

    def run(self, ctx) -> "bool | None":
        if not ctx.fills:
            return True
        today = pd.Timestamp(ctx.today).date()
        # Reconstruct which fills came from which ExitSignal by ticker.
        exit_lookup: dict[str, str] = {}
        for ticker, sig in (ctx.exits or []):
            et = getattr(sig, "exit_type", None) or ""
            exit_lookup[ticker] = et
        for fill in ctx.fills:
            if fill.side != OrderSide.SELL:
                continue
            t = fill.ticker
            held = ctx.execution_backend.get_position_quantity(t)
            # Full liquidate: backend now reports 0 shares (post-fill).
            if held <= 0:
                ctx.last_sell_dates[t] = today
            # G8: path-rule exits stamp cooldown date regardless of partial
            et = exit_lookup.get(t, "")
            if et in DEFAULT_STOP_EXIT_TYPES:
                ctx.last_stop_exit_dates[t] = today
        return True


class PruneFullExitsTask(Task):
    """Pop fully-liquidated tickers from ctx.holdings.

    Pins: partial trims keep the position open (entry_date / entry_price
    preserved). Without this, a closed position would linger in holdings
    and trigger spurious stop-loss / trailing-stop logic next bar.
    """

    def run(self, ctx) -> "bool | None":
        backend = _require_backend(ctx)
        to_pop: list[str] = []
        for ticker in list(ctx.holdings.keys()):
            if backend.get_position_quantity(ticker) <= 0:
                to_pop.append(ticker)
        for ticker in to_pop:
            ctx.holdings.pop(ticker, None)
        return True


# ─── Buy-side dedupe + execute ─────────────────────────────────────────────


class DedupeBuysTask(Task):
    """Collapse duplicate ``ctx.orders`` rows per ticker (first-write-wins).

    Pins: matches sim BUY-DEDUPE guard (2026-04-25 audit). Two independent
    Jobs nominating the same ticker on the same bar would otherwise debit
    cash twice + double the share count.
    """

    def run(self, ctx) -> "bool | None":
        seen: set[str] = set()
        deduped = []
        for order in (ctx.orders or []):
            t = order.get("ticker") if isinstance(order, dict) else None
            if t is None or t in seen:
                continue
            seen.add(t)
            deduped.append(order)
        ctx.orders = deduped
        return True


def _order_fields_finite(order: dict) -> bool:
    """§5.13.11: NaN/inf/zero price/shares/target_pct silently skip the order.

    Matches ``lean.commit():371-378`` + sim defensive guard.
    """
    try:
        p = float(order["price"])
        s = float(order["shares"])
        t = float(order["target_pct"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isfinite(p) and p > 0
        and math.isfinite(s) and s > 0
        and math.isfinite(t) and t > 0
    )


class ExecuteBuysTask(Task):
    """Place buy orders via the backend; record Fills to ctx.fills."""

    def run(self, ctx) -> "bool | None":
        backend = _require_backend(ctx)
        today = pd.Timestamp(ctx.today)
        for order in (ctx.orders or []):
            if not _order_fields_finite(order):
                log.warning(
                    "ExecuteBuysTask: dropping order with non-finite fields: %r",
                    order,
                )
                continue
            intent = OrderIntent(
                ticker=order["ticker"], side=OrderSide.BUY,
                shares=int(order["shares"]),
                target_pct=float(order["target_pct"]),
                today=today,
                reason=str(order.get("detail") or "buy"),
                exit_type=None,
            )
            fill = backend.place_market_order(intent)
            ctx.fills.append(fill)
        return True


class UpsertHoldingsTask(Task):
    """Create / update :class:`HoldingState` for each BUY Fill.

    Pins: top-ups preserve ``entry_date`` (tax LT/ST clock) and apply
    volume-weighted average ``entry_price`` (matches ``sim._apply_buy`` +
    ``lean.commit:389-407``). New entries stamp thesis-baseline scores
    from the originating order.
    """

    def run(self, ctx) -> "bool | None":
        if not ctx.fills:
            return True
        # Build a per-ticker order lookup so we can read thesis scores
        # for new HoldingStates. ``ctx.orders`` is the same list the buys
        # came from (DedupeBuysTask already collapsed it).
        order_by_ticker = {
            o["ticker"]: o for o in (ctx.orders or []) if isinstance(o, dict)
        }
        today_date = pd.Timestamp(ctx.today).date()
        for fill in ctx.fills:
            if fill.side != OrderSide.BUY:
                continue
            ticker = fill.ticker
            order = order_by_ticker.get(ticker, {})
            hs = ctx.holdings.get(ticker)
            if hs is None:
                ctx.holdings[ticker] = HoldingState(
                    entry_price=fill.price,
                    entry_date=today_date,
                    high_watermark=fill.price,
                    entry_rank_score=order.get("rank_score"),
                    entry_panel_score=order.get("panel_score"),
                    entry_kelly_target_pct=order.get("kelly_target_pct"),
                    entry_regime=order.get("regime"),
                )
            else:
                # Top-up: vol-weighted avg cost over prior basis.
                # backend now reports post-fill share count.
                new_qty = ctx.execution_backend.get_position_quantity(ticker)
                old_qty = new_qty - fill.shares
                if old_qty > 0 and new_qty > 0:
                    hs.entry_price = (
                        hs.entry_price * old_qty + fill.price * fill.shares
                    ) / new_qty
                hs.high_watermark = max(hs.high_watermark, fill.price)
        return True


__all__ = [
    "PrepareExecutionTask",
    "DedupeExitsTask",
    "ExecuteExitsTask",
    "StampWashSaleTask",
    "PruneFullExitsTask",
    "DedupeBuysTask",
    "ExecuteBuysTask",
    "UpsertHoldingsTask",
    "is_full_liquidate_signal",
]
