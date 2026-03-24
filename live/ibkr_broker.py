"""Interactive Brokers broker (stub).

Requires ``ib_insync`` and a running TWS or IB Gateway instance.
"""

from __future__ import annotations

from .broker import BaseBroker


class IBKRBroker(BaseBroker):
    """Execute orders via Interactive Brokers TWS / Gateway.

    This is a stub — the full implementation will use ``ib_insync``.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
    ):
        self._host = host
        self._port = port
        self._client_id = client_id

    def connect(self) -> None:
        raise NotImplementedError(
            "IBKR broker not configured. Install ib_insync and "
            "ensure TWS/Gateway is running."
        )

    def disconnect(self) -> None:
        raise NotImplementedError("IBKR broker not configured.")

    def get_position(self, symbol: str) -> float:
        raise NotImplementedError("IBKR broker not configured.")

    def get_account_value(self) -> float:
        raise NotImplementedError("IBKR broker not configured.")

    def place_order(self, symbol: str, action: str, quantity: float) -> dict:
        raise NotImplementedError("IBKR broker not configured.")
