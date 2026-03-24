"""Local Parquet cache for OHLCV data."""

from pathlib import Path

import pandas as pd


class LocalStore:
    """Read/write OHLCV data as Parquet files.

    Layout::

        {data_dir}/{SYMBOL}/{timeframe}.parquet

    The store deduplicates rows by date index on save and supports
    incremental appends.
    """

    def __init__(self, data_dir: Path | str = "data/ohlcv"):
        self.data_dir = Path(data_dir)

    # ── paths ──────────────────────────────────────────────────────────

    def _path(self, symbol: str, timeframe: str = "1d") -> Path:
        return self.data_dir / symbol.upper() / f"{timeframe}.parquet"

    # ── public API ─────────────────────────────────────────────────────

    def load(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame | None:
        """Load from local Parquet.  Returns ``None`` if the file is missing."""
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

    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "1d",
    ) -> Path:
        """Save (or append) OHLCV data.  Deduplicates by index."""
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
    ) -> bool:
        """Check whether the local cache fully covers ``[start, end]``."""
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
        if end and df.index.max() < pd.Timestamp(end):
            return False
        return True
