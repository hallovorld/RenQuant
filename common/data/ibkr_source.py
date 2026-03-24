"""Interactive Brokers data source (stub).

This module defines the interface for IBKR historical data retrieval.
Implementation will use ``ib_insync`` once the IBKR gateway is configured.
"""

import pandas as pd

from .base import DataSource


class IBKRSource(DataSource):
    """Fetch historical OHLCV from Interactive Brokers.

    Requires a running TWS or IB Gateway instance.
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

    @property
    def name(self) -> str:
        return "ibkr"

    def fetch(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "IBKR data source is not configured yet. "
            "Install ib_insync and ensure TWS/Gateway is running."
        )
