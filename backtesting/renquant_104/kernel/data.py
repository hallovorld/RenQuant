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
        df = df[~df.index.duplicated(keep="last")].sort_index()
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

    df = df[~df.index.duplicated(keep="last")].sort_index()

    if cache:
        store.save(df, symbol)

    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]

    return df


def fetch_intraday_bars(
    symbols: list[str] | str,
    *,
    timeframe: str = "5Min",
    start: "datetime.datetime | None" = None,
    end: "datetime.datetime | None" = None,
    limit: int = 10_000,
) -> dict[str, pd.DataFrame]:
    """Fetch intraday bars via Alpaca's IEX feed (free tier).

    `timeframe` is an Alpaca string: "1Min", "5Min", "15Min", "1Hour", "1Day".
    `start`/`end` are datetime objects (UTC or naive — Alpaca treats naive as UTC).
    Returns `{symbol: DataFrame}` with columns [open, high, low, close, volume, ...].

    Credentials are read from the ALPACA_API_KEY / ALPACA_SECRET_KEY env vars
    (populate via .env before calling).
    """
    import datetime as _dt
    import os

    if isinstance(symbols, str):
        symbols = [symbols]
    if not symbols:
        return {}

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import DataFeed
    except ImportError as exc:
        raise RuntimeError("alpaca-py not installed") from exc

    key    = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "fetch_intraday_bars: ALPACA_API_KEY + ALPACA_SECRET_KEY must be set "
            "(source .env before running)",
        )

    # Parse timeframe
    tf_map = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
    }
    if timeframe not in tf_map:
        raise ValueError(f"Unknown Alpaca timeframe {timeframe!r}. "
                          f"Supported: {list(tf_map.keys())}")

    now = _dt.datetime.utcnow()
    if end is None:
        end = now
    if start is None:
        # Default: last 5 market days
        start = end - _dt.timedelta(days=7)

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    # Force IEX feed — free tier can't query current-day SIP data
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=tf_map[timeframe],
        start=start,
        end=end,
        limit=limit,
        feed=DataFeed.IEX,
    )
    bars = client.get_stock_bars(req)
    df_all = bars.df

    out: dict[str, pd.DataFrame] = {}
    if df_all is None or df_all.empty:
        return out
    # Alpaca returns a MultiIndex DataFrame (symbol, timestamp)
    for sym in symbols:
        if sym in df_all.index.get_level_values(0):
            sub = df_all.xs(sym, level=0).copy()
            out[sym] = sub
    return out


__all__ = ["LocalStore", "fetch_ohlcv", "fetch_intraday_bars"]
