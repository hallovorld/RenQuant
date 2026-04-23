"""Cached intraday (hourly) OHLCV bars for panel aggregation.

Cache layout mirrors `LocalStore` + `FundamentalsStore`:

  data/intraday/{SYMBOL}/1h.parquet

Rows are indexed by a timezone-naive `pd.DatetimeIndex` (US/Eastern wall-clock
after tz-strip), columns `[open, high, low, close, volume]`. Multiple sessions
per file; callers de-duplicate by timestamp on save.

The Alpaca fetcher lives in `kernel/data.py::fetch_intraday_bars`. Keep that
module narrow — this one only owns the cache shape so tests can inject stub
data without network calls.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger("kernel.intraday")


@dataclass
class HourlyBarStore:
    """Parquet-backed cache at `data/intraday/{SYMBOL}/1h.parquet`."""
    data_dir: Path = Path("data/intraday")

    def __post_init__(self):
        if not isinstance(self.data_dir, Path):
            self.data_dir = Path(self.data_dir)

    def _path(self, symbol: str) -> Path:
        return self.data_dir / symbol.upper() / "1h.parquet"

    def load(self, symbol: str) -> pd.DataFrame | None:
        p = self._path(symbol)
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def save(self, df: pd.DataFrame, symbol: str) -> Path:
        p = self._path(symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        existing = self.load(symbol)
        if existing is not None and not existing.empty:
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_parquet(p)
        return p


__all__ = ["HourlyBarStore"]
