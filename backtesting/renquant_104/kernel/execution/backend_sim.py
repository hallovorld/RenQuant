"""Sim-side :class:`ExecutionBackend` implementation.

Wraps:

* cash + position quantities (long-only)
* per-ticker volume-weighted average cost (for unrealized P&L)
* per-bar last-price cache (mark + fill price source)
* :class:`kernel.execution.fees.FeeConfig` schedule
* :class:`kernel.execution.slippage.SlippageConfig` adjustment
* :class:`kernel.execution.t2_settlement.T2CashQueue` for sell proceeds

Does **not** own: tax accounting, lot disposal (FIFO/HIFO), trade log,
equity curve, wash-sale stamping. Those are strategy-level and stay in
:class:`SimAdapter`'s post-pipeline hook (slice 3b). The backend ONLY
models broker-side state.

Two execution modes:

* ``exec_enabled=False`` (default) — slippage off, fees zero,
  proceeds credited T+0. Byte-identical to the pre-2026-05-10 sim
  cash math used by every legacy fixture.
* ``exec_enabled=True`` — slippage on (half-spread + impact), full
  fee schedule, sell proceeds queued T+N.

The flag matches ``SimAdapter._exec_enabled`` (set by
:func:`SimAdapter.__init__`); slice 3b just propagates it.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import pandas as pd

from .backend import ExecutionBackend
from .fees import FeeConfig, compute_buy_fees, compute_sell_fees
from .slippage import SlippageConfig, slip_fill_price
from .t2_settlement import T2CashQueue
from .types import Fill, OrderIntent, OrderSide

log = logging.getLogger("kernel.execution.backend_sim")


@dataclass
class _SimPosition:
    quantity: int = 0
    avg_cost: float = 0.0   # volume-weighted average cost basis


class SimBackend(ExecutionBackend):
    """Self-contained simulator broker."""

    def __init__(
        self,
        starting_cash: float = 100_000.0,
        *,
        fee_config: FeeConfig | None = None,
        slip_config: SlippageConfig | None = None,
        exec_enabled: bool = False,
        t2_days: int = 0,
    ) -> None:
        if not math.isfinite(starting_cash) or starting_cash < 0:
            raise ValueError(
                f"SimBackend.starting_cash must be finite and non-negative, "
                f"got {starting_cash!r}"
            )
        self._cash: float = float(starting_cash)
        self._positions: dict[str, _SimPosition] = {}
        self._last_prices: dict[str, float] = {}
        self._fills: list[Fill] = []
        self._fee_cfg = fee_config or FeeConfig()
        self._slip_cfg = slip_config or SlippageConfig()
        self._exec_enabled = bool(exec_enabled)
        # T+N only when execution model on AND t2_days > 0.
        effective_t2 = t2_days if exec_enabled and t2_days > 0 else 0
        self._t2_queue: T2CashQueue | None = (
            T2CashQueue(settlement_days=effective_t2) if effective_t2 > 0 else None
        )

    # ── Bar-level lifecycle ─────────────────────────────────────────────

    def update_bar_prices(
        self,
        prices: dict[str, float],
        today: pd.Timestamp,
    ) -> None:
        """Refresh the per-ticker last-price cache for the current bar.

        §5.13.11: non-finite or non-positive prices are silently dropped
        (matches ``lean.commit:332-338`` ``algo._prev_closes`` filter).
        Callers MAY blanket-pass a ``ctx.prices`` dict; bad rows are
        skipped without raising.
        """
        for ticker, raw_price in prices.items():
            try:
                p = float(raw_price)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(p) or p <= 0:
                continue
            self._last_prices[ticker] = p
        # ``today`` kept on the API surface for future per-bar invariants
        # (e.g. delisting-handling or NYSE holiday checks).
        _ = today

    def drain_settled(self, today: pd.Timestamp) -> float:
        """Credit any T+N proceeds that have settled by ``today``.

        Returns the amount credited (0 if T+N disabled or nothing due).
        Caller MUST invoke once per bar at the top of the loop; otherwise
        T+N proceeds linger in the queue forever.
        """
        if self._t2_queue is None:
            return 0.0
        amount = self._t2_queue.drain(today)
        if amount > 0:
            self._cash += amount
        return amount

    # ── ExecutionBackend ABC ────────────────────────────────────────────

    def place_market_order(self, intent: OrderIntent) -> Fill:
        # KeyError if no price seeded — matches FakeBackend contract.
        market_price = self.get_last_price(intent.ticker)

        if intent.side == OrderSide.BUY:
            return self._execute_buy(intent, market_price)
        return self._execute_sell(intent, market_price)

    # ── Internal: BUY / SELL helpers (≤50 lines each per §1c) ───────────

    def _execute_buy(self, intent: OrderIntent, market_price: float) -> Fill:
        shares = int(intent.shares)  # type: ignore[arg-type]
        fill_price = self._fill_price_for_buy(market_price, shares)
        # §5.13.5 parity with sim._apply_buy:1066-1077 — fees gated by
        # the same exec_enabled flag the legacy adapter uses, so a
        # __new__-constructed test fixture bypassing __init__ stays
        # byte-identical to the pre-execution-model behaviour.
        fees = (
            compute_buy_fees(shares, fill_price, self._fee_cfg)["total"]
            if self._exec_enabled else 0.0
        )
        invest = shares * fill_price + fees
        if not math.isfinite(invest):
            raise ValueError(
                f"SimBackend BUY {intent.ticker}: invest non-finite "
                f"(shares={shares} fill_price={fill_price} fees={fees})"
            )
        # §5.13.5 parity with sim._apply_buy:1083 — tight epsilon for
        # floating-point insufficient-cash guard.
        if invest > self._cash + 1e-6:
            raise ValueError(
                f"SimBackend BUY {intent.ticker}: insufficient cash "
                f"(need {invest:.2f}, have {self._cash:.2f})"
            )
        self._cash -= invest
        pos = self._positions.setdefault(intent.ticker, _SimPosition())
        new_qty = pos.quantity + shares
        if new_qty > 0 and pos.quantity >= 0:
            pos.avg_cost = (
                pos.avg_cost * pos.quantity + fill_price * shares
            ) / new_qty
        pos.quantity = new_qty
        fill = Fill(
            ticker=intent.ticker, side=OrderSide.BUY,
            shares=shares, price=fill_price, fees=fees,
            today=intent.today,
        )
        self._fills.append(fill)
        return fill

    def _execute_sell(self, intent: OrderIntent, market_price: float) -> Fill:
        pos = self._positions.get(intent.ticker)
        if pos is None or pos.quantity <= 0:
            raise ValueError(
                f"SimBackend SELL {intent.ticker}: no position to close"
            )
        if intent.is_full_liquidate:
            shares = int(pos.quantity)
        else:
            requested = int(intent.shares)  # type: ignore[arg-type]
            if requested > pos.quantity:
                raise ValueError(
                    f"SimBackend SELL {intent.ticker}: requested {requested} "
                    f"> held {pos.quantity}"
                )
            shares = requested
        fill_price = self._fill_price_for_sell(market_price, shares)
        # Mirror sim._apply_sell:894-897 — fees only when exec_enabled.
        fees = (
            compute_sell_fees(shares, fill_price, self._fee_cfg)["total"]
            if self._exec_enabled else 0.0
        )
        notional = shares * fill_price
        net_proceeds = notional - fees
        if not math.isfinite(net_proceeds):
            raise ValueError(
                f"SimBackend SELL {intent.ticker}: net_proceeds non-finite "
                f"(shares={shares} fill_price={fill_price} fees={fees})"
            )
        if self._t2_queue is not None:
            self._t2_queue.add_pending(intent.today, net_proceeds)
        else:
            self._cash += net_proceeds
        pos.quantity -= shares
        if pos.quantity == 0:
            pos.avg_cost = 0.0
        fill = Fill(
            ticker=intent.ticker, side=OrderSide.SELL,
            shares=shares, price=fill_price, fees=fees,
            today=intent.today,
        )
        self._fills.append(fill)
        return fill

    # ── Slippage adapters ────────────────────────────────────────────────

    def _fill_price_for_buy(self, market_price: float, shares: int) -> float:
        if not self._exec_enabled:
            return market_price
        slipped = slip_fill_price(
            market_price=market_price, side="buy",
            shares=shares, adv_shares=None, cfg=self._slip_cfg,
        )
        return slipped if (math.isfinite(slipped) and slipped > 0) else market_price

    def _fill_price_for_sell(self, market_price: float, shares: int) -> float:
        if not self._exec_enabled:
            return market_price
        slipped = slip_fill_price(
            market_price=market_price, side="sell",
            shares=shares, adv_shares=None, cfg=self._slip_cfg,
        )
        return slipped if (math.isfinite(slipped) and slipped > 0) else market_price

    # ── Read-only state ─────────────────────────────────────────────────

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
        if self._t2_queue is not None:
            total += self._t2_queue.pending_total()
        for ticker, pos in self._positions.items():
            if pos.quantity == 0:
                continue
            try:
                price = self.get_last_price(ticker)
            except KeyError:
                price = pos.avg_cost
            total += price * pos.quantity
        return total

    def get_last_price(self, ticker: str) -> float:
        if ticker not in self._last_prices:
            raise KeyError(
                f"SimBackend has no seeded price for {ticker!r} "
                f"(call update_bar_prices first)"
            )
        return self._last_prices[ticker]

    @property
    def fills(self) -> tuple[Fill, ...]:
        return tuple(self._fills)


__all__ = ["SimBackend"]
