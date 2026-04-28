"""Paper (simulated) broker for testing the live runner without IBKR."""

from __future__ import annotations

import logging

from .broker import BaseBroker

log = logging.getLogger(__name__)


class PaperBroker(BaseBroker):
    broker_name = "paper"  # state-file isolation tag (see kernel.state_paths)

    """Simulates order fills locally.  No real money is involved.

    Round-2 audit (#R2-7..10): now tracks cash, average cost basis, and
    last-fill price for sensible mark-to-market. Pre-fix, place_order
    was a pure position-counter that ignored cash + price entirely, so
    the strategy in `--broker paper` mode saw infinite cash regardless
    of trades.
    """

    def __init__(self, initial_cash: float = 100_000):
        self._cash:    float = float(initial_cash)
        self._initial_cash: float = float(initial_cash)
        self._positions:  dict[str, float] = {}
        self._avg_cost:   dict[str, float] = {}
        self._last_price: dict[str, float] = {}
        self._order_counter = 0
        self._connected = False

    def connect(self) -> None:
        self._connected = True
        log.info("PaperBroker connected (cash=$%.2f)", self._cash)

    def disconnect(self) -> None:
        self._connected = False
        log.info("PaperBroker disconnected")

    def get_position(self, symbol: str) -> float:
        return self._positions.get(symbol, 0.0)

    def get_account_value(self) -> float:
        # Mark-to-market: cash + Σ qty × last_price
        total = self._cash
        for sym, qty in self._positions.items():
            if qty <= 0:
                continue
            total += qty * self._last_price.get(sym, self._avg_cost.get(sym, 0.0))
        return total

    def get_cash(self) -> float:
        return self._cash

    def get_avg_cost(self, symbol: str) -> float:
        return self._avg_cost.get(symbol, 0.0)

    def get_all_positions(self) -> list[dict]:
        rows: list[dict] = []
        for sym, qty in self._positions.items():
            if qty <= 0:
                continue
            cost  = self._avg_cost.get(sym, 0.0)
            price = self._last_price.get(sym, cost)
            rows.append({
                "symbol":          sym,
                "qty":             qty,
                "avg_entry_price": cost,
                "market_value":    qty * price,
                "unrealized_pl":   qty * (price - cost),
            })
        return rows

    def set_price(self, symbol: str, price: float) -> None:
        """Test/runner hook: stamp the last-known price for mark-to-market.

        The runner doesn't currently feed prices in (it pulls from parquet),
        so callers can call this between place_order calls to keep
        get_account_value in sync with reality during paper-mode dry-runs.
        """
        if price > 0:
            self._last_price[symbol] = float(price)

    def place_order(
        self, symbol: str, action: str, quantity: float, price: "float | None" = None,
    ) -> dict:
        self._order_counter += 1
        oid = f"PAPER-{self._order_counter:04d}"
        action_u = action.upper()
        # Use supplied price; fallback to last known; final fallback to avg
        # cost (for closing trades). If we have no price reference, the
        # cash impact is undefined — but that's a configuration issue we
        # surface rather than silently mis-compute.
        if price is None:
            price = self._last_price.get(symbol)
        if price is None:
            log.warning(
                "PaperBroker.place_order(%s, %s, %s): no price — cash NOT updated",
                action, symbol, quantity,
            )
            invest = 0.0
        else:
            price  = float(price)
            invest = float(quantity) * price
            self._last_price[symbol] = price

        if action_u == "BUY":
            if invest > self._cash + 1e-6 and price is not None:
                log.warning(
                    "PaperBroker: insufficient cash for %s (need $%.2f, have $%.2f) — "
                    "executing anyway, going to negative cash",
                    symbol, invest, self._cash,
                )
            old_qty   = self._positions.get(symbol, 0.0)
            old_cost  = self._avg_cost.get(symbol, 0.0)
            new_qty   = old_qty + quantity
            if new_qty > 0 and price is not None:
                self._avg_cost[symbol] = (
                    old_cost * old_qty + price * quantity
                ) / new_qty
            self._positions[symbol] = new_qty
            self._cash -= invest
        elif action_u == "SELL":
            held = self._positions.get(symbol, 0.0)
            if quantity > held + 1e-9:
                log.warning(
                    "PaperBroker: SELL %s qty=%s exceeds held=%s — clipping",
                    symbol, quantity, held,
                )
                quantity = held
                if price is not None:
                    invest = quantity * price
            new_qty = held - quantity
            self._positions[symbol] = max(0.0, new_qty)
            if self._positions[symbol] == 0:
                self._avg_cost.pop(symbol, None)
            self._cash += invest

        log.info("Order %s: %s %s %.0f shares @ $%s",
                 oid, action_u, symbol, quantity,
                 f"{price:.2f}" if price is not None else "?")
        return {"order_id": oid, "status": "filled", "action": action_u,
                "symbol": symbol, "quantity": quantity, "price": price}
