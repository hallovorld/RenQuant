"""Data coverage measurement — single source of truth for what data each
ticker has, across all data sources the model consumes.

2026-05-04 P0: surfaced after the arm A NaN-leaf collapse audit. Half the
universe (~100/183 tickers) had no hourly/minute bars historically; the
model trained on a panel where 50%+ of rows had multiple NaN intraday
features, and at inference XGB collapsed those rows to a single
terminal-leaf score. The structural fix (row-coverage gate, task #13)
filters low-coverage rows. This module is the *measurement* layer that
tells callers WHICH rows are low-coverage and WHY.

Used by:
  * scripts/snapshot_data_coverage.py — daily coverage report (L3)
  * tests/test_data_coverage.py — contract that pins current coverage so
    future regressions fail loud (L4)
  * future cron drift detection — alert when coverage worsens

Public API
----------
  ``compute_coverage(watchlist, repo_root) -> dict``
      Returns ``{ticker: TickerCoverage}`` where TickerCoverage is a
      dataclass holding bool flags + freshness deltas for each source.

  ``coverage_summary(coverage) -> dict``
      Aggregates per-source counts (n_with / n_without / pct_covered).
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class TickerCoverage:
    ticker:               str
    has_ohlcv_daily:      bool = False
    ohlcv_max_date:       _dt.date | None = None
    ohlcv_age_days:       int | None = None
    has_hourly_bars:      bool = False
    hourly_max_date:      _dt.date | None = None
    hourly_n_rows:        int = 0
    has_minute_bars:      bool = False
    minute_max_date:      _dt.date | None = None
    minute_n_rows:        int = 0
    has_fundamentals:     bool = False
    has_earnings_surprise: bool = False
    has_insider:          bool = False

    def to_dict(self) -> dict:
        return {
            "ticker":                self.ticker,
            "has_ohlcv_daily":       self.has_ohlcv_daily,
            "ohlcv_max_date":        str(self.ohlcv_max_date) if self.ohlcv_max_date else None,
            "ohlcv_age_days":        self.ohlcv_age_days,
            "has_hourly_bars":       self.has_hourly_bars,
            "hourly_max_date":       str(self.hourly_max_date) if self.hourly_max_date else None,
            "hourly_n_rows":         self.hourly_n_rows,
            "has_minute_bars":       self.has_minute_bars,
            "minute_max_date":       str(self.minute_max_date) if self.minute_max_date else None,
            "minute_n_rows":         self.minute_n_rows,
            "has_fundamentals":      self.has_fundamentals,
            "has_earnings_surprise": self.has_earnings_surprise,
            "has_insider":           self.has_insider,
        }


def _safe_max_date(path: Path, date_col: str | None = None) -> tuple[_dt.date | None, int]:
    """Read a parquet/csv at ``path``; return (max_date, n_rows). Defensive."""
    if not path.exists():
        return None, 0
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    except Exception:
        return None, 0
    if df.empty:
        return None, 0
    if date_col and date_col in df.columns:
        s = pd.to_datetime(df[date_col], errors="coerce").dropna()
        return (s.max().date() if not s.empty else None, len(df))
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.max().date(), len(df)
    # Try a reasonable fallback
    for cand in ("date", "timestamp", "Date"):
        if cand in df.columns:
            s = pd.to_datetime(df[cand], errors="coerce").dropna()
            return (s.max().date() if not s.empty else None, len(df))
    return None, len(df)


def compute_coverage(
    watchlist: list[str],
    repo_root: Path,
    *,
    today: _dt.date | None = None,
) -> dict[str, TickerCoverage]:
    """Measure per-ticker × per-source coverage on disk.

    Paths checked (relative to repo_root):
      * data/ohlcv/{TICKER}/1d.parquet
      * data/intraday/{TICKER}/1h.parquet
      * data/intraday/{TICKER}/10m.parquet
      * data/fundamentals/{TICKER}.parquet
      * data/earnings_surprise/{TICKER}.parquet OR .csv
      * data/insider_trades/{TICKER}.parquet OR .csv
    """
    if today is None:
        today = _dt.date.today()
    repo_root = Path(repo_root)

    out: dict[str, TickerCoverage] = {}
    for tic in watchlist:
        cov = TickerCoverage(ticker=tic)

        ohlcv_path = repo_root / "data" / "ohlcv" / tic / "1d.parquet"
        d, n = _safe_max_date(ohlcv_path)
        cov.has_ohlcv_daily = d is not None and n > 0
        cov.ohlcv_max_date = d
        if d is not None:
            cov.ohlcv_age_days = (today - d).days

        hourly_path = repo_root / "data" / "intraday" / tic / "1h.parquet"
        d, n = _safe_max_date(hourly_path)
        cov.has_hourly_bars = d is not None and n > 0
        cov.hourly_max_date = d
        cov.hourly_n_rows = n

        # Production filename is 10min.parquet (not 10m.parquet); accept both.
        minute_path = repo_root / "data" / "intraday" / tic / "10min.parquet"
        if not minute_path.exists():
            minute_path = repo_root / "data" / "intraday" / tic / "10m.parquet"
        d, n = _safe_max_date(minute_path)
        cov.has_minute_bars = d is not None and n > 0
        cov.minute_max_date = d
        cov.minute_n_rows = n

        fund_path = repo_root / "data" / "fundamentals" / f"{tic}.parquet"
        cov.has_fundamentals = fund_path.exists()

        earn_pq = repo_root / "data" / "earnings_surprise" / f"{tic}.parquet"
        earn_csv = repo_root / "data" / "earnings_surprise" / f"{tic}.csv"
        cov.has_earnings_surprise = earn_pq.exists() or earn_csv.exists()

        ins_pq = repo_root / "data" / "insider_trades" / f"{tic}.parquet"
        ins_csv = repo_root / "data" / "insider_trades" / f"{tic}.csv"
        cov.has_insider = ins_pq.exists() or ins_csv.exists()

        out[tic] = cov
    return out


def coverage_summary(coverage: dict[str, TickerCoverage]) -> dict:
    """Aggregate counts + percentages per source.

    Returns flat dict suitable for JSON snapshot or log line.
    """
    n = max(1, len(coverage))
    counts: dict[str, int] = {}
    sources = (
        "has_ohlcv_daily",
        "has_hourly_bars",
        "has_minute_bars",
        "has_fundamentals",
        "has_earnings_surprise",
        "has_insider",
    )
    for src in sources:
        counts[src] = sum(1 for c in coverage.values() if getattr(c, src))

    return {
        "n_tickers": len(coverage),
        "ohlcv_daily_n":      counts["has_ohlcv_daily"],
        "ohlcv_daily_pct":    counts["has_ohlcv_daily"] / n,
        "hourly_n":           counts["has_hourly_bars"],
        "hourly_pct":         counts["has_hourly_bars"] / n,
        "minute_n":           counts["has_minute_bars"],
        "minute_pct":         counts["has_minute_bars"] / n,
        "fundamentals_n":     counts["has_fundamentals"],
        "fundamentals_pct":   counts["has_fundamentals"] / n,
        "earnings_surprise_n":   counts["has_earnings_surprise"],
        "earnings_surprise_pct": counts["has_earnings_surprise"] / n,
        "insider_n":          counts["has_insider"],
        "insider_pct":        counts["has_insider"] / n,
    }
