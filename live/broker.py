"""Broker abstraction for live trading."""

from abc import ABC, abstractmethod


class BaseBroker(ABC):
    """Interface for order execution backends."""

    # Broker tag for state-file isolation. Each subclass overrides via
    # class attribute or @property. Used by adapters/runner.py to compute
    # broker-specific paths for live_state.json and runs.db so a paper
    # smoke can never contaminate alpaca live state. See
    # kernel/state_paths.py for the path convention.
    broker_name: str = "unknown"

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_position(self, symbol: str) -> float:
        """Return current share count for *symbol* (0 if flat)."""
        ...

    @abstractmethod
    def get_account_value(self) -> float:
        """Return total account liquidation value."""
        ...

    def get_avg_cost(self, symbol: str) -> float:
        """Return average cost basis per share for *symbol* (0 if not held). Override for accuracy."""
        return 0.0

    def get_cash(self) -> float:
        """Return available cash.  Defaults to account value; override for accuracy."""
        return self.get_account_value()

    def get_all_positions(self) -> list[dict]:
        """Return all open positions as a list of dicts with keys:
        symbol, qty, avg_entry_price, market_value, unrealized_pl.
        Override for batch-efficient broker implementations.
        """
        return []

    def get_filled_orders(self, after: str | None = None) -> list[dict]:
        """Return filled orders since *after* ('YYYY-MM-DD').  Returns [] if not supported."""
        return []

    def get_open_orders(self) -> set[str]:
        """Return the set of symbols that have a pending/open order right now.
        Used to avoid placing duplicate orders when a catch-up run fires after a
        prior run already submitted orders that haven't filled yet.
        Returns an empty set if not supported by this broker.
        """
        return set()

    @abstractmethod
    def place_order(self, symbol: str, action: str, quantity: float) -> dict:
        """Place a market order.

        Args:
            symbol:   Ticker.
            action:   ``"BUY"`` or ``"SELL"``.
            quantity: Number of shares (positive).

        Returns:
            Order confirmation dict with at least ``{"order_id", "status"}``.
        """
        ...
