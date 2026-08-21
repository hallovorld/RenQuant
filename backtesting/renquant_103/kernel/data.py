"""OHLCV data fetching with local Parquet cache.

Self-contained — no common/ imports.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class LocalStore:
    """Read/write OHLCV data as Parquet files.

    Layout::

        {data_dir}/{SYMBOL}/{timeframe}.parquet
    """

    def __init__(self, data_dir: Path | str = "data/ohlcv"):
        self.data_dir = Path(data_dir)

    def _path(self, symbol: str, timeframe: str = "1d") -> Path:
        return self.data_dir / symbol.upper() / f"{timeframe}.parquet"

    def load(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame | None:
        """Load from local Parquet. Returns None if the file is missing."""
        path = self._path(symbol, timeframe)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if start:
            df = df.loc[start:]
        if end:
            df = df.loc[:end]
        return df if not df.empty else None

    def save(self, df: pd.DataFrame, symbol: str, timeframe: str = "1d") -> Path:
        """Save (or append) OHLCV data. Deduplicates by index."""
        path = self._path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        if path.exists():
            existing = pd.read_parquet(path)
            if not isinstance(existing.index, pd.DatetimeIndex):
                existing.index = pd.to_datetime(existing.index)
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

        df.to_parquet(path)
        return path

    def has_range(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: str | None = None,
        end: str | None = None,
        tolerance_days: int = 5,
    ) -> bool:
        """Check whether the local cache covers [start, end]."""
        path = self._path(symbol, timeframe)
        if not path.exists():
            return False
        df = pd.read_parquet(path)
        if df.empty:
            return False
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if start and df.index.min() > pd.Timestamp(start):
            return False
        if end and df.index.max() < pd.Timestamp(end) - pd.Timedelta(days=tolerance_days):
            return False
        return True


_default_store = LocalStore()


def fetch_ohlcv(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    provider: str = "yfinance",
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data, using a local Parquet cache when possible."""
    store = _default_store

    if cache and store.has_range(symbol, start=start, end=end):
        cached = store.load(symbol, start=start, end=end)
        if cached is not None:
            return cached

    if provider == "yfinance":
        from openbb import obb  # lazy import — OpenBB init is slow
        kwargs: dict = {"symbol": symbol, "provider": "yfinance"}
        if start:
            kwargs["start_date"] = start
        if end:
            kwargs["end_date"] = end
        df = obb.equity.price.historical(**kwargs).to_df()
    else:
        raise ValueError(f"Unknown provider {provider!r}. Supported: ['yfinance']")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if cache:
        store.save(df, symbol)

    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]

    return df


__all__ = ["LocalStore", "fetch_ohlcv"]
