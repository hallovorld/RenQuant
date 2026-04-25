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
class _TimeframedBarStore:
    """Common parquet cache keyed by symbol + a fixed filename suffix.

    Subclasses set `_filename` (e.g. "1h.parquet", "10min.parquet") so
    every intraday timeframe gets the same dedup/merge semantics.
    """
    data_dir: Path = Path("data/intraday")
    _filename: str = "bars.parquet"

    def __post_init__(self):
        if not isinstance(self.data_dir, Path):
            self.data_dir = Path(self.data_dir)

    def _path(self, symbol: str) -> Path:
        return self.data_dir / symbol.upper() / self._filename

    def load(self, symbol: str) -> pd.DataFrame | None:
        # Audit fix INT-READ-RACE (Round 2 deep audit, 2026-04-25):
        # other parquet loaders (FundamentalsStore — FU-4) wrap
        # `pd.read_parquet` in try/except so a corrupt file (truncated
        # by Ctrl-C mid-write, partial flush after disk-full, or
        # cross-version pyarrow incompat) is treated as cache-miss
        # rather than crashing the panel pipeline. This loader was
        # missing that guard. Now mirror FU-4 + ES-READ-RACE.
        p = self._path(symbol)
        if not p.exists():
            return None
        try:
            df = pd.read_parquet(p)
        except Exception as exc:
            log.warning(
                "%s.load(%s): corrupt parquet — %s; treating as cache-miss",
                type(self).__name__, symbol, exc,
            )
            return None
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def save(self, df: pd.DataFrame, symbol: str) -> Path:
        # Audit fix INT-ATOM (Round 2 deep audit, 2026-04-25): atomic
        # write via .tmp + rename. Same as DC-2-CACHE / FU-1.
        p = self._path(symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        existing = self.load(symbol)
        if existing is not None and not existing.empty:
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        tmp = p.with_suffix(p.suffix + ".tmp")
        df.to_parquet(tmp)
        tmp.replace(p)
        return p


@dataclass
class HourlyBarStore(_TimeframedBarStore):
    """Parquet-backed cache at `data/intraday/{SYMBOL}/1h.parquet`."""
    _filename: str = "1h.parquet"


@dataclass
class MinuteBarStore(_TimeframedBarStore):
    """Parquet-backed cache for 10-minute bars at `data/intraday/{SYMBOL}/10min.parquet`.

    Added 2026-04-24 to support finer-grained panel features. Expected
    ~39 bars per session × ~250 trading days × N years ≈ 10k+ bars/ticker,
    so cache hygiene matters — parquet compression + dedup on save.
    """
    _filename: str = "10min.parquet"


__all__ = ["HourlyBarStore", "MinuteBarStore"]
