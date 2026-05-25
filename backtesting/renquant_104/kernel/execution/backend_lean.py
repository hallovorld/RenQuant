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
            algo.MarketOrder(sym, shares)
            return Fill(
                ticker=intent.ticker, side=OrderSide.BUY,
                shares=shares, price=price, fees=0.0,
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
            algo.Liquidate(sym)
        else:
            requested = int(intent.shares)  # type: ignore[arg-type]
            if requested > current:
                raise ValueError(
                    f"LeanBackend SELL {intent.ticker}: requested {requested} "
                    f"> held {current}"
                )
            shares = requested
            algo.MarketOrder(sym, -shares)
        return Fill(
            ticker=intent.ticker, side=OrderSide.SELL,
            shares=shares, price=price, fees=0.0,
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
