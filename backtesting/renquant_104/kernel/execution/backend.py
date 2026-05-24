"""Execution backend abstraction.

The pipeline-facing contract sim / Alpaca / LEAN all implement. Per
CLAUDE.md §5.13.5 the *business* logic (tax accounting, wash-sale
stamps, holding-state mutation) lives in pipeline Tasks (slice 2 of
this refactor); this module covers ONLY broker-side I/O — placing
orders, reading positions, computing fees from the canonical fee
schedule in :mod:`kernel.execution.fees`.

:class:`FakeBackend` is the in-memory reference used by the pipeline
tests. It MUST NOT be imported from production code (sim, runner, LEAN
adapters); a CI grep — ``grep -rn 'FakeBackend' adapters/`` — should
report zero hits.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from .fees import FeeConfig, compute_buy_fees, compute_sell_fees
from .types import Fill, OrderIntent, OrderSide


# ─── ExecutionBackend ABC ──────────────────────────────────────────────────


class ExecutionBackend(ABC):
    """Translates :class:`OrderIntent` into broker calls and reports state.

    Concrete subclasses MUST implement every abstract method. The
    pipeline never special-cases on backend type — every consumer
    routes through the same interface. Adapters that need additional
    state (e.g. Alpaca's account number, LEAN's algorithm handle)
    accept it in ``__init__`` and keep it private.
    """

    @abstractmethod
    def place_market_order(self, intent: OrderIntent) -> Fill:
        """Execute ``intent`` at the bar's market price and return a
        confirmed :class:`Fill`. Must raise :class:`ValueError` for
        SELL intents on tickers with no current position (do NOT silently
        no-op — that masks pipeline bugs)."""

    @abstractmethod
    def get_position_quantity(self, ticker: str) -> float:
        """Current long-only share count; zero if not held."""

    @abstractmethod
    def get_unrealized_pnl(self, ticker: str) -> float:
        """Mark-to-market P&L on the open position. Zero if no position."""

    @abstractmethod
    def get_cash(self) -> float:
        """Available cash/buying power as defined by the backend."""

    @abstractmethod
    def get_portfolio_value(self) -> float:
        """Total liquidation value = cash + mark-to-market positions."""

    @abstractmethod
    def get_last_price(self, ticker: str) -> float:
        """Last observed close price for ``ticker``. Raises
        :class:`KeyError` if no price has ever been observed."""


# ─── FakeBackend (test-only reference impl) ────────────────────────────────


@dataclass
class _FakePosition:
    quantity: int = 0
    avg_cost: float = 0.0  # volume-weighted average cost basis


class FakeBackend(ExecutionBackend):
    """In-memory :class:`ExecutionBackend` for pipeline tests.

    Models cash, per-ticker share count + avg cost, and last-price cache.
    Uses the **same** :func:`kernel.execution.fees.compute_*_fees` schedule
    sim and live use — so a buy/sell round-trip on FakeBackend should
    match SimBackend / AlpacaBackend to the penny on identical inputs.

    Test-only: do NOT import from production adapters.
    """

    def __init__(
        self,
        starting_cash: float = 100_000.0,
        fee_config: FeeConfig | None = None,
    ) -> None:
        if not math.isfinite(starting_cash) or starting_cash < 0:
            raise ValueError(
                f"starting_cash must be finite and non-negative, "
                f"got {starting_cash!r}"
            )
        self._cash: float = float(starting_cash)
        self._positions: dict[str, _FakePosition] = {}
        self._last_prices: dict[str, float] = {}
        self._intents: list[OrderIntent] = []
        self._fills: list[Fill] = []
        self._fee_cfg = fee_config or FeeConfig()

    # ── Test helpers ────────────────────────────────────────────────────

    def seed_price(self, ticker: str, price: float, today: pd.Timestamp) -> None:
        """Update the last-known price for ``ticker``. Marks-to-market.

        ``today`` is accepted for API symmetry with real backends (which
        carry per-bar prices); FakeBackend stores only the latest value.
        """
        if not math.isfinite(price) or price <= 0:
            raise ValueError(
                f"seed_price requires finite positive price, got {price!r}"
            )
        # `today` is currently informational on FakeBackend; real backends
        # use it to index their per-bar price cache.
        _ = today
        self._last_prices[ticker] = float(price)

    @property
    def intents(self) -> tuple[OrderIntent, ...]:
        """Immutable view of every intent processed, in order."""
        return tuple(self._intents)

    @property
    def fills(self) -> tuple[Fill, ...]:
        return tuple(self._fills)

    # ── ABC implementations ─────────────────────────────────────────────

    def place_market_order(self, intent: OrderIntent) -> Fill:
        self._intents.append(intent)
        price = self.get_last_price(intent.ticker)  # raises KeyError on unseeded

        if intent.side == OrderSide.BUY:
            shares = int(intent.shares)  # type: ignore[arg-type]  # guarded by __post_init__
            fees_dict = compute_buy_fees(shares, price, self._fee_cfg)
            fees = fees_dict["total"]
            self._cash -= shares * price + fees
            pos = self._positions.setdefault(intent.ticker, _FakePosition())
            new_qty = pos.quantity + shares
            # Volume-weighted average cost basis (parity with
            # SimAdapter._apply_buy top-up math).
            if new_qty > 0 and pos.quantity >= 0:
                pos.avg_cost = (
                    pos.avg_cost * pos.quantity + price * shares
                ) / new_qty
            pos.quantity = new_qty
            fill = Fill(
                ticker=intent.ticker,
                side=OrderSide.BUY,
                shares=shares,
                price=price,
                fees=fees,
                today=intent.today,
            )

        else:  # SELL
            pos = self._positions.get(intent.ticker)
            if pos is None or pos.quantity <= 0:
                raise ValueError(
                    f"SELL OrderIntent for {intent.ticker!r}: no position to close"
                )
            if intent.is_full_liquidate:
                shares = int(pos.quantity)
            else:
                requested = int(intent.shares)  # type: ignore[arg-type]
                if requested > pos.quantity:
                    raise ValueError(
                        f"SELL OrderIntent for {intent.ticker!r}: "
                        f"requested {requested} > held {pos.quantity}"
                    )
                shares = requested
            fees_dict = compute_sell_fees(shares, price, self._fee_cfg)
            fees = fees_dict["total"]
            revenue = shares * price - fees
            self._cash += revenue
            pos.quantity -= shares
            if pos.quantity == 0:
                pos.avg_cost = 0.0  # release cost basis on full close
            fill = Fill(
                ticker=intent.ticker,
                side=OrderSide.SELL,
                shares=shares,
                price=price,
                fees=fees,
                today=intent.today,
            )

        self._fills.append(fill)
        return fill

    def get_position_quantity(self, ticker: str) -> float:
        pos = self._positions.get(ticker)
        return float(pos.quantity) if pos is not None else 0.0

    def get_unrealized_pnl(self, ticker: str) -> float:
        pos = self._positions.get(ticker)
        if pos is None or pos.quantity == 0:
            return 0.0
        try:
            price = self.get_last_price(ticker)
        except KeyError:
            return 0.0
        return float((price - pos.avg_cost) * pos.quantity)

    def get_cash(self) -> float:
        return self._cash

    def get_portfolio_value(self) -> float:
        total = self._cash
        for ticker, pos in self._positions.items():
            if pos.quantity == 0:
                continue
            try:
                price = self.get_last_price(ticker)
            except KeyError:
                # No mark — treat at cost (defensive; in practice every
                # held ticker has a seeded price).
                price = pos.avg_cost
            total += price * pos.quantity
        return total

    def get_last_price(self, ticker: str) -> float:
        if ticker not in self._last_prices:
            raise KeyError(
                f"FakeBackend has no seeded price for {ticker!r}; "
                "call seed_price(ticker, price, today) first."
            )
        return self._last_prices[ticker]


__all__ = ["ExecutionBackend", "FakeBackend"]
