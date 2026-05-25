"""LEAN-side :class:`ExecutionBackend` — thin proxy over ``QCAlgorithm``.

LEAN's brokerage layer is the source of truth for cash + position state
during a backtest (or paper-trade via QuantConnect Cloud). Per §5.13.5
the adapter MUST NOT maintain a parallel mirror; every read delegates
to ``algo.Portfolio`` / ``algo.Securities`` and every write delegates
to ``algo.MarketOrder`` / ``algo.Liquidate``.

Order placement semantics (matches ``adapters/lean.py:202`` legacy
commit body):

* BUY  → ``algo.MarketOrder(sym, shares)`` (the pipeline is the sizing
  owner; LEAN must not recompute a different quantity from target_pct).
* SELL full     → ``algo.Liquidate(sym)`` (closes entire position).
* SELL partial  → ``algo.MarketOrder(sym, -shares)`` (negative qty).

The synchronous :class:`Fill` we hand back carries ``fees=0`` because
LEAN tracks fees on its brokerage model and reports them via
``algo.Portfolio.TotalFees`` after the bar. Strategy-level fee accounting
(if any) reads that value in the adapter's post-pipeline hook, NOT here.
"""
from __future__ import annotations

import math
from typing import Any

from .backend import ExecutionBackend
from .types import Fill, OrderIntent, OrderSide


def _ticket_status_text(ticket: Any) -> str:
    status = getattr(ticket, "Status", None)
    return str(status or "").lower()


def _ticket_float(ticket: Any, *names: str) -> float | None:
    for name in names:
        value = getattr(ticket, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(out):
            return out
    return None


def _confirmed_fill(
    ticket: Any,
    *,
    ticker: str,
    requested_qty: float,
    fallback_price: float,
) -> tuple[float, float]:
    """Return confirmed (abs_qty, avg_price) or raise on missing evidence."""
    requested_abs = abs(float(requested_qty))
    fallback = float(fallback_price)
    if ticket is None:
        raise RuntimeError(f"LeanBackend {ticker}: missing order ticket")
    if isinstance(ticket, (list, tuple)):
        total_qty = 0.0
        total_value = 0.0
        for item in ticket:
            qty, price = _confirmed_fill(
                item,
                ticker=ticker,
                requested_qty=requested_abs,
                fallback_price=fallback,
            )
            total_qty += qty
            total_value += qty * price
        if total_qty <= 0.0:
            raise RuntimeError(f"LeanBackend {ticker}: no filled quantity")
        return total_qty, total_value / total_qty

    status = _ticket_status_text(ticket)
    if any(token in status for token in ("reject", "cancel", "invalid", "error")):
        raise RuntimeError(f"LeanBackend {ticker}: order not filled ({status})")
    qty = _ticket_float(
        ticket,
        "QuantityFilled",
        "AbsoluteQuantityFilled",
        "FilledQuantity",
    )
    price = _ticket_float(
        ticket,
        "AverageFillPrice",
        "AvgFillPrice",
        "FillPrice",
        "Price",
    )
    if qty is not None and abs(qty) > 0.0:
        return abs(qty), price or fallback
    if "filled" in status:
        return requested_abs, price or fallback
    raise RuntimeError(f"LeanBackend {ticker}: order not filled ({status or 'unknown'})")


class LeanBackend(ExecutionBackend):
    """Proxy over ``QCAlgorithm`` for the LEAN backtest / paper path.

    Constructor takes the live ``algo`` reference; the backend keeps no
    private state. All mutations land directly on the algo's broker
    bookkeeping via the QC API.
    """

    def __init__(self, algo: Any) -> None:
        self._algo = algo

    # ── ABC implementation ─────────────────────────────────────────────

    def place_market_order(self, intent: OrderIntent) -> Fill:
        algo = self._algo
        sym = algo.symbols.get(intent.ticker)
        if sym is None:
            raise ValueError(
                f"LeanBackend: no LEAN symbol mapping for {intent.ticker!r}"
            )
        # Snapshot last price BEFORE the order so the Fill carries the
        # bar-close price both sim and LEAN report.
        price = self.get_last_price(intent.ticker)

        if intent.side == OrderSide.BUY:
            shares = int(intent.shares)  # type: ignore[arg-type]
            ticket = algo.MarketOrder(sym, shares)
            filled_qty, fill_price = _confirmed_fill(
                ticket,
                ticker=intent.ticker,
                requested_qty=shares,
                fallback_price=price,
            )
            return Fill(
                ticker=intent.ticker, side=OrderSide.BUY,
                shares=int(filled_qty), price=fill_price, fees=0.0,
                today=intent.today,
            )

        # SELL — full liquidate vs partial trim.
        current = float(algo.Portfolio[sym].Quantity)
        if current <= 0:
            raise ValueError(
                f"LeanBackend SELL {intent.ticker}: no position to close "
                f"(LEAN reports quantity={current})"
            )
        if intent.is_full_liquidate:
            shares = int(current)
            ticket = algo.Liquidate(sym)
        else:
            requested = int(intent.shares)  # type: ignore[arg-type]
            if requested > current:
                raise ValueError(
                    f"LeanBackend SELL {intent.ticker}: requested {requested} "
                    f"> held {current}"
                )
            shares = requested
            ticket = algo.MarketOrder(sym, -shares)
        filled_qty, fill_price = _confirmed_fill(
            ticket,
            ticker=intent.ticker,
            requested_qty=shares,
            fallback_price=price,
        )
        return Fill(
            ticker=intent.ticker, side=OrderSide.SELL,
            shares=int(filled_qty), price=fill_price, fees=0.0,
            today=intent.today,
        )

    def get_position_quantity(self, ticker: str) -> float:
        algo = self._algo
        sym = algo.symbols.get(ticker)
        if sym is None:
            return 0.0
        try:
            return float(algo.Portfolio[sym].Quantity)
        except (KeyError, AttributeError):
            return 0.0

    def get_unrealized_pnl(self, ticker: str) -> float:
        algo = self._algo
        sym = algo.symbols.get(ticker)
        if sym is None:
            return 0.0
        try:
            v = float(algo.Portfolio[sym].UnrealizedProfit)
        except (KeyError, AttributeError):
            return 0.0
        return v if math.isfinite(v) else 0.0

    def get_cash(self) -> float:
        return float(self._algo.Portfolio.Cash)

    def get_portfolio_value(self) -> float:
        return float(self._algo.Portfolio.TotalPortfolioValue)

    def get_last_price(self, ticker: str) -> float:
        algo = self._algo
        sym = algo.symbols.get(ticker)
        if sym is None:
            raise KeyError(
                f"LeanBackend: no LEAN symbol mapping for {ticker!r}"
            )
        try:
            p = float(algo.Securities[sym].Price)
        except (KeyError, AttributeError) as exc:
            raise KeyError(
                f"LeanBackend: LEAN Securities[{ticker!r}] has no Price"
            ) from exc
        if not math.isfinite(p) or p <= 0:
            raise ValueError(
                f"LeanBackend: LEAN Securities[{ticker!r}].Price is not "
                f"finite/positive (got {p!r})"
            )
        return p


__all__ = ["LeanBackend"]
