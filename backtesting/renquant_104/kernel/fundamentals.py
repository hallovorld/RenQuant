"""Cached equity fundamentals — cross-sectional factor inputs.

Produces four factor columns per ticker:

  earnings_yield        trailing TTM EPS / last close
  roe                   return on equity (trailing)
  gross_profitability   gross profit / total assets  (Novy-Marx)
  book_to_price         book value per share / last close

Cache layout mirrors `LocalStore`:

  data/fundamentals/{SYMBOL}.parquet   # one row per snapshot

Each row is indexed by the UTC date of the fetch; callers forward-fill into
a daily panel. The snapshot model is deliberately simple — extending to
full time-series via `obb.equity.fundamental.*` is a future change.

The OpenBB import is **lazy** so importing this module (e.g. during
training) doesn't pay the OpenBB init cost until a fetch is requested.
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("kernel.fundamentals")


FACTOR_COLS: list[str] = [
    "earnings_yield",
    "roe",
    "gross_profitability",
    "book_to_price",
    # Sentiment / positioning (yfinance .info)
    "short_pct_float",
]


@dataclass
class FundamentalsStore:
    """Parquet-backed cache at `data/fundamentals/{SYMBOL}.parquet`."""
    data_dir: Path = Path("data/fundamentals")

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
        existing = self.load(symbol)
        if existing is not None:
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_parquet(p)
        return p

    def latest(self, symbol: str) -> dict[str, float] | None:
        df = self.load(symbol)
        if df is None or df.empty:
            return None
        row = df.iloc[-1]
        return {c: (float(row[c]) if c in row and pd.notna(row[c]) else float("nan"))
                for c in FACTOR_COLS}


# ── Provider: OpenBB ──────────────────────────────────────────────────────────

def _fetch_from_openbb(symbol: str) -> dict[str, float]:
    """Single-snapshot fetch via OpenBB. Falls back to NaN on any missing field.

    Kept in its own function so tests can monkey-patch it without touching OpenBB.
    """
    try:
        from openbb import obb  # lazy; OpenBB init is slow
    except Exception as exc:
        raise RuntimeError("openbb is not installed") from exc

    out: dict[str, float] = {c: float("nan") for c in FACTOR_COLS}

    def _latest_non_nan(df: pd.DataFrame, col: str) -> float | None:
        """Return the first non-NaN value of `col` scanning top-down (most recent first)."""
        if df is None or df.empty or col not in df.columns:
            return None
        s = df[col]
        for v in s:
            if pd.notna(v):
                return float(v)
        return None

    # Metrics endpoint: trailing-12m snapshot covering EY / ROE / B/P.
    # OpenBB yfinance returns rows in reverse chronological order (iloc[0] = newest).
    try:
        m = obb.equity.fundamental.metrics(symbol=symbol, provider="yfinance").to_df()
        if m is not None and not m.empty:
            pe  = _latest_non_nan(m, "pe_ratio")    or _latest_non_nan(m, "peRatio")
            roe = _latest_non_nan(m, "return_on_equity") or _latest_non_nan(m, "returnOnEquity")
            bp  = _latest_non_nan(m, "price_to_book") or _latest_non_nan(m, "priceToBook")
            if pe is not None and pe > 0:
                out["earnings_yield"] = 1.0 / pe
            if roe is not None:
                out["roe"] = roe
            if bp is not None and bp > 0:
                out["book_to_price"] = 1.0 / bp
    except Exception as exc:
        log.warning("fundamentals.metrics(%s) failed — %s", symbol, exc)

    # Novy-Marx gross profitability = gross_profit / total_assets
    try:
        bs = obb.equity.fundamental.balance(symbol=symbol, period="annual",
                                            provider="yfinance").to_df()
        incs = obb.equity.fundamental.income(symbol=symbol, period="annual",
                                             provider="yfinance").to_df()
        ta = _latest_non_nan(bs,   "total_assets")
        gp = _latest_non_nan(incs, "gross_profit")
        if ta is not None and gp is not None and ta > 0:
            out["gross_profitability"] = gp / ta
    except Exception as exc:
        log.warning("fundamentals.balance/income(%s) failed — %s", symbol, exc)

    # Short interest (yfinance .info). ETFs and some tickers don't report —
    # missing values left unset; z-score step sector-median-fills.
    try:
        import yfinance as yf  # noqa: PLC0415
        info = yf.Ticker(symbol).info or {}
        sp = info.get("shortPercentOfFloat")
        if sp is not None and pd.notna(sp):
            out["short_pct_float"] = float(sp)
    except Exception as exc:
        log.debug("yfinance short interest(%s) failed — %s", symbol, exc)

    return out


def fetch_fundamentals(
    symbol: str,
    *,
    cache: bool = True,
    store: FundamentalsStore | None = None,
    provider_fn=None,
) -> dict[str, float]:
    """Fetch a single snapshot of fundamentals for `symbol` and cache it.

    provider_fn: injected for testing; defaults to OpenBB.
    """
    store = store or FundamentalsStore()
    if cache:
        cached = store.latest(symbol)
        if cached is not None:
            return cached

    fetch = provider_fn or _fetch_from_openbb
    fundamentals = fetch(symbol)

    if cache and fundamentals:
        row = pd.DataFrame(
            [fundamentals],
            index=pd.DatetimeIndex(
                [pd.Timestamp(datetime.datetime.utcnow().date())],
                name="date",
            ),
        )
        store.save(row, symbol)
    return fundamentals


def fetch_fundamentals_watchlist(
    watchlist: list[str],
    *,
    cache: bool = True,
    provider_fn=None,
    store: FundamentalsStore | None = None,
) -> dict[str, dict[str, float]]:
    """Fetch + cache fundamentals for every ticker. Returns a plain dict."""
    out: dict[str, dict[str, float]] = {}
    for sym in watchlist:
        try:
            out[sym] = fetch_fundamentals(sym, cache=cache,
                                          store=store, provider_fn=provider_fn)
        except Exception as exc:
            log.warning("  %-6s fundamentals fetch failed: %s", sym, exc)
    return out


__all__ = [
    "FACTOR_COLS",
    "FundamentalsStore",
    "fetch_fundamentals",
    "fetch_fundamentals_watchlist",
]
