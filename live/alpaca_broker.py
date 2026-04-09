"""Alpaca broker for live and paper trading.

Requires ``alpaca-py`` (``pip install alpaca-py``) and API credentials
set via environment variables or passed at construction:

    ALPACA_API_KEY
    ALPACA_SECRET_KEY

By default connects to the paper trading endpoint.  Pass ``paper=False``
for live trading.
"""

from __future__ import annotations

import logging
import os

from .broker import BaseBroker

log = logging.getLogger(__name__)


class AlpacaBroker(BaseBroker):
    """Execute orders via Alpaca Markets API."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
    ):
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self._paper = paper
        self._trading_client = None
        self._order_counter = 0

    def connect(self) -> None:
        try:
            from alpaca.trading.client import TradingClient
        except ImportError:
            raise ImportError(
                "alpaca-py is required for Alpaca broker. "
                "Install it with: pip install alpaca-py"
            )

        if not self._api_key or not self._secret_key:
            raise ValueError(
                "Alpaca API credentials not found. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables, or pass them to "
                "AlpacaBroker(api_key=..., secret_key=...)."
            )

        self._trading_client = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=self._paper,
        )

        account = self._trading_client.get_account()
        mode = "paper" if self._paper else "LIVE"
        log.info(
            "Alpaca connected (%s) — equity=$%s, cash=$%s, status=%s",
            mode, account.equity, account.cash, account.status,
        )

        if account.status != "ACTIVE":
            log.warning("Account status is %s — trading may be restricted", account.status)

    def disconnect(self) -> None:
        self._trading_client = None
        log.info("Alpaca disconnected")

    def get_position(self, symbol: str) -> float:
        from alpaca.common.exceptions import APIError

        try:
            position = self._trading_client.get_open_position(symbol)
            return float(position.qty)
        except APIError:
            return 0.0

    def get_account_value(self) -> float:
        account = self._trading_client.get_account()
        return float(account.equity)

    def get_cash(self) -> float:
        """Return available cash (non-margin)."""
        account = self._trading_client.get_account()
        return float(account.cash)

    def place_order(self, symbol: str, action: str, quantity: float) -> dict:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        side = OrderSide.BUY if action.upper() == "BUY" else OrderSide.SELL

        request = MarketOrderRequest(
            symbol=symbol,
            qty=int(quantity),
            side=side,
            time_in_force=TimeInForce.DAY,
        )

        order = self._trading_client.submit_order(request)
        self._order_counter += 1

        log.info(
            "Order %s: %s %s %d shares — status=%s",
            order.id, action, symbol, int(quantity), order.status,
        )

        return {
            "order_id": str(order.id),
            "status": str(order.status),
            "action": action,
            "symbol": symbol,
            "quantity": int(quantity),
        }

    def get_avg_cost(self, symbol: str) -> float:
        from alpaca.common.exceptions import APIError
        try:
            position = self._trading_client.get_open_position(symbol)
            return float(position.avg_entry_price)
        except APIError:
            return 0.0

    def get_all_positions(self) -> list[dict]:
        """Return all open positions as a list of dicts."""
        positions = self._trading_client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "avg_entry_price": float(p.avg_entry_price),
            }
            for p in positions
        ]

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        clock = self._trading_client.get_clock()
        return clock.is_open
