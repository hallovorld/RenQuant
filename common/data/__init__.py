"""Data ingestion with local Parquet caching and pluggable providers.

Typical usage::

    from common.data import fetch_ohlcv
    df = fetch_ohlcv("NVDA", "2022-01-01", "2023-01-01")
"""

from __future__ import annotations

import pandas as pd

from .base import DataSource
from .ibkr_source import IBKRSource
from .store import LocalStore
from .yfinance_source import YFinanceSource

_SOURCES: dict[str, type[DataSource]] = {
    "yfinance": YFinanceSource,
    "ibkr": IBKRSource,
}

_default_store = LocalStore()


def fetch_ohlcv(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    provider: str = "yfinance",
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data, using a local Parquet cache when possible.

    With ``cache=True`` (default) the local store is checked first.
    If the requested range is fully covered, no network call is made.
    Otherwise the provider is queried and the result is saved locally.

    Args:
        symbol:   Ticker symbol (e.g. ``"NVDA"``).
        start:    Start date ``"YYYY-MM-DD"`` (inclusive).
        end:      End date ``"YYYY-MM-DD"`` (inclusive).
        provider: ``"yfinance"`` (default) or ``"ibkr"``.
        cache:    If ``True``, read/write the local Parquet cache.

    Returns:
        OHLCV DataFrame with a DatetimeIndex.
    """
    store = _default_store

    # Try cache first
    if cache and store.has_range(symbol, start=start, end=end):
        cached = store.load(symbol, start=start, end=end)
        if cached is not None:
            return cached

    # Fetch from provider
    source_cls = _SOURCES.get(provider)
    if source_cls is None:
        raise ValueError(f"Unknown provider {provider!r}. Available: {list(_SOURCES)}")
    source = source_cls()
    df = source.fetch(symbol, start=start, end=end)

    # Normalise index to DatetimeIndex so string-based .loc slicing always works
    # (OpenBB/yfinance may return datetime.date objects instead of Timestamps)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Save to cache
    if cache:
        store.save(df, symbol)

    # Slice to requested range (provider may return extra rows)
    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]

    return df


__all__ = [
    "DataSource",
    "IBKRSource",
    "LocalStore",
    "YFinanceSource",
    "fetch_ohlcv",
]
