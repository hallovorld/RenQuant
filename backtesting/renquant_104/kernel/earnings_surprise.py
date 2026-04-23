"""Earnings-surprise cache (yfinance .earnings_dates backed).

Populates `data/earnings_surprise/{SYMBOL}.parquet` with one row per past
earnings announcement:

    index: pd.Timestamp (announcement date)
    columns:
        eps_actual       float — reported EPS
        eps_estimate     float — consensus estimate immediately prior
        surprise_abs     float — eps_actual - eps_estimate
        surprise_pct     float — (eps_actual - eps_estimate) / |eps_estimate|

The cross-sectional factor computed downstream is the **trailing-4-quarter
cumulative surprise %**, daily-forward-filled so it has a value on every
trading day (the value updates step-wise at each new announcement).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger("kernel.earnings_surprise")


SURPRISE_COLS: list[str] = [
    "eps_actual", "eps_estimate", "surprise_abs", "surprise_pct",
]


@dataclass
class EarningsSurpriseStore:
    """Parquet cache at `data/earnings_surprise/{SYMBOL}.parquet`."""
    data_dir: Path = Path("data/earnings_surprise")

    def __post_init__(self):
        if not isinstance(self.data_dir, Path):
            self.data_dir = Path(self.data_dir)

    def _path(self, symbol: str) -> Path:
        return self.data_dir / f"{symbol.upper()}.parquet"

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
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_parquet(p)
        return p


# ── Provider ──────────────────────────────────────────────────────────────────

def _fetch_from_yfinance(symbol: str) -> pd.DataFrame:
    """Fetch past earnings surprises via yfinance `.earnings_dates`.

    Returns an empty DataFrame on any failure (offline, rate-limited,
    unsupported ticker). Caller is expected to tolerate missing values —
    the z-score step sector-median-fills nulls.
    """
    try:
        import yfinance as yf  # noqa: PLC0415
        ed = yf.Ticker(symbol).earnings_dates
    except Exception as exc:
        log.warning("yfinance.earnings_dates(%s) failed — %s", symbol, exc)
        return pd.DataFrame(columns=SURPRISE_COLS)

    if ed is None or ed.empty:
        return pd.DataFrame(columns=SURPRISE_COLS)

    # Normalize: yfinance returns with tz-aware index + columns
    # ["EPS Estimate", "Reported EPS", "Surprise(%)"]. Keep only rows with
    # a reported actual (drop upcoming estimates).
    df = ed.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.rename(columns={
        "EPS Estimate": "eps_estimate",
        "Reported EPS": "eps_actual",
        "Surprise(%)":  "surprise_pct_yf",
    })
    # Keep only rows with a reported actual
    df = df[df["eps_actual"].notna()].copy()
    df["surprise_abs"] = df["eps_actual"] - df["eps_estimate"]
    # Compute surprise_pct ourselves — yfinance's Surprise(%) is in percent,
    # we want a fraction. Guard against zero denominators.
    denom = df["eps_estimate"].abs().replace(0, np.nan)
    df["surprise_pct"] = df["surprise_abs"] / denom
    return df[SURPRISE_COLS].sort_index()


def fetch_earnings_surprise(
    symbol: str,
    *,
    cache: bool = True,
    store: EarningsSurpriseStore | None = None,
    provider_fn: Callable[[str], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Load or fetch earnings-surprise history for `symbol`.

    Returns the cached DataFrame if available (fast path), else fetches via
    provider_fn (defaults to yfinance) and writes to cache.
    """
    store = store or EarningsSurpriseStore()
    if cache:
        cached = store.load(symbol)
        if cached is not None and not cached.empty:
            return cached

    fetch = provider_fn or _fetch_from_yfinance
    df = fetch(symbol)
    if cache and not df.empty:
        store.save(df, symbol)
    return df


def fetch_earnings_surprise_watchlist(
    watchlist: list[str],
    *,
    cache: bool = True,
    provider_fn: Callable[[str], pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch earnings surprises for every ticker in `watchlist`. Returns
    a dict — empty frames signal "no data" and do not raise."""
    out: dict[str, pd.DataFrame] = {}
    for t in watchlist:
        try:
            out[t] = fetch_earnings_surprise(t, cache=cache, provider_fn=provider_fn)
        except Exception as exc:
            log.warning("fetch_earnings_surprise(%s) failed — %s", t, exc)
            out[t] = pd.DataFrame(columns=SURPRISE_COLS)
    return out


# ── Factor computation ────────────────────────────────────────────────────────

def compute_earnings_surprise_cum(
    surprises: dict[str, pd.DataFrame],
    ohlcv: dict[str, pd.DataFrame],
    *,
    trailing_quarters: int = 4,
) -> dict[str, pd.Series]:
    """Trailing-N-quarter cumulative surprise %, aligned to each ticker's
    daily OHLCV index via forward-fill.

    On each trading day, the value is sum(surprise_pct) over the most
    recent `trailing_quarters` announcements at or before that date.
    Tickers with no earnings data get an all-NaN series.
    """
    out: dict[str, pd.Series] = {}
    for ticker, df_ohlcv in ohlcv.items():
        surprise_df = surprises.get(ticker)
        idx = df_ohlcv.index
        if surprise_df is None or surprise_df.empty or "surprise_pct" not in surprise_df.columns:
            out[ticker] = pd.Series(np.nan, index=idx)
            continue
        sp = surprise_df["surprise_pct"].sort_index()
        # Trailing-N rolling sum of the last N announcements. Operates on
        # announcement-sampled index first, then reindexed to daily + ffilled.
        trailing = sp.rolling(trailing_quarters, min_periods=1).sum()
        daily = trailing.reindex(idx, method="ffill")
        out[ticker] = daily
    return out
