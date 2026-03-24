"""Paper (simulated) broker for testing the live runner without IBKR."""

from __future__ import annotations

import logging

from .broker import BaseBroker

log = logging.getLogger(__name__)


class PaperBroker(BaseBroker):
    """Simulates order fills locally.  No real money is involved."""

    def __init__(self, initial_cash: float = 100_000):
        self._cash = initial_cash
        self._positions: dict[str, float] = {}
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
        # In paper mode we only track cash (no mark-to-market)
        return self._cash

    def place_order(self, symbol: str, action: str, quantity: float) -> dict:
        self._order_counter += 1
        oid = f"PAPER-{self._order_counter:04d}"

        if action.upper() == "BUY":
            self._positions[symbol] = self._positions.get(symbol, 0) + quantity
        elif action.upper() == "SELL":
            self._positions[symbol] = self._positions.get(symbol, 0) - quantity

        log.info("Order %s: %s %s %.0f shares", oid, action, symbol, quantity)
        return {"order_id": oid, "status": "filled", "action": action,
                "symbol": symbol, "quantity": quantity}
