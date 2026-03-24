"""Abstract base class for market data providers."""

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """Interface for market data providers.

    Every implementation must return a DataFrame with:
      - DatetimeIndex
      - Columns: open, high, low, close, volume
    """

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data for *symbol* over the given date range."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name used in config and logging."""
        ...
