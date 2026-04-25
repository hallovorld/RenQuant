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
        # Audit fix FU-4 (Round 2 deep audit, 2026-04-25): catch corrupt
        # parquet (truncated mid-write, truncated by disk full, malformed
        # by older format). Pre-fix this raised → callers like
        # FundamentalsStore.latest() crashed → LoadFundamentalsTask
        # could fail the entire panel pipeline. Now: log + return None
        # so the caller treats it as cache-miss and refetches.
        p = self._path(symbol)
        if not p.exists():
            return None
        try:
            df = pd.read_parquet(p)
        except Exception as exc:
            import logging  # noqa: PLC0415
            logging.getLogger("kernel.fundamentals").warning(
                "FundamentalsStore.load(%s): corrupt parquet — %s; "
                "treating as cache-miss", symbol, exc,
            )
            return None
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def save(self, df: pd.DataFrame, symbol: str) -> Path:
        # Audit fix FU-1 (Round 2 deep audit, 2026-04-25): atomic write
        # via .tmp + rename. Same pattern as DC-2-CACHE — prevent
        # process-kill mid-write from leaving a truncated parquet.
        p = self._path(symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        existing = self.load(symbol)
        if existing is not None:
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        tmp = p.with_suffix(p.suffix + ".tmp")
        df.to_parquet(tmp)
        tmp.replace(p)
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
        """Return the first non-NaN value of `col` scanning top-down.

        Round-3 audit (#R3-42): if df.index is a DatetimeIndex, sort
        descending first so we deterministically return the MOST-RECENT
        non-NaN value regardless of the provider's row order.
        """
        if df is None or df.empty or col not in df.columns:
            return None
        s = df[col]
        if isinstance(df.index, pd.DatetimeIndex):
            # Sort by index descending so the first non-NaN is the most recent.
            s = s.sort_index(ascending=False)
        for v in s:
            if pd.notna(v):
                return float(v)
        return None

    # All network calls below are timeout-wrapped via
    # `kernel.net_safety.call_with_timeout`. The 2026-04-23 incident
    # was a yfinance hang that blocked PanelDataJob for 10+ minutes;
    # every external call now abandons after 20 s and returns None,
    # letting the caller continue with whatever metadata it has.
    from .net_safety import call_with_timeout  # noqa: PLC0415

    # Metrics endpoint: trailing-12m snapshot covering EY / ROE / B/P.
    m = call_with_timeout(
        lambda: obb.equity.fundamental.metrics(
            symbol=symbol, provider="yfinance").to_df(),
        timeout_sec = 20.0,
        label       = f"obb.metrics({symbol})",
    )
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

    # Novy-Marx gross profitability = gross_profit / total_assets
    bs = call_with_timeout(
        lambda: obb.equity.fundamental.balance(
            symbol=symbol, period="annual", provider="yfinance").to_df(),
        timeout_sec = 20.0,
        label       = f"obb.balance({symbol})",
    )
    incs = call_with_timeout(
        lambda: obb.equity.fundamental.income(
            symbol=symbol, period="annual", provider="yfinance").to_df(),
        timeout_sec = 20.0,
        label       = f"obb.income({symbol})",
    )
    if bs is not None and incs is not None and not bs.empty and not incs.empty:
        ta = _latest_non_nan(bs,   "total_assets")
        gp = _latest_non_nan(incs, "gross_profit")
        if ta is not None and gp is not None and ta > 0:
            out["gross_profitability"] = gp / ta

    # Short interest (yfinance .info). ETFs and some tickers don't report —
    # missing values left unset; z-score step sector-median-fills.
    def _fetch_info():
        import yfinance as yf  # noqa: PLC0415
        return yf.Ticker(symbol).info or {}
    info = call_with_timeout(
        _fetch_info, timeout_sec=15.0, label=f"yf.info({symbol})",
    )
    if info:
        sp = info.get("shortPercentOfFloat")
        if sp is not None and pd.notna(sp):
            out["short_pct_float"] = float(sp)

    return out


def fetch_fundamentals(
    symbol: str,
    *,
    cache: bool = True,
    store: FundamentalsStore | None = None,
    provider_fn=None,
    refresh_after_days: float = 90.0,
) -> dict[str, float]:
    """Fetch a single snapshot of fundamentals for `symbol` and cache it.

    provider_fn: injected for testing; defaults to OpenBB.

    Round-3 audit (#R3-37): cache previously NEVER refreshed once written.
    Quarterly fundamentals stayed stale forever. Now: when the most-recent
    cached snapshot is older than `refresh_after_days` (default 90d —
    quarterly cadence), refetch and append a fresh snapshot.
    """
    store = store or FundamentalsStore()
    if cache:
        cached_df = store.load(symbol)
        cached    = store.latest(symbol)
        if cached is not None:
            latest_idx = (cached_df.index.max()
                          if cached_df is not None and not cached_df.empty
                          else None)
            if latest_idx is not None and isinstance(latest_idx, pd.Timestamp):
                age_days = (pd.Timestamp.now().normalize() - latest_idx).days
                if age_days <= refresh_after_days:
                    return cached
            else:
                # Index unparsable — return what we have rather than re-fetch
                return cached

    fetch = provider_fn or _fetch_from_openbb
    fundamentals = fetch(symbol)

    if cache and fundamentals:
        # Use UTC-date but anchored to the trader's local-day for index sanity:
        # consume modern timezone-aware utcnow.
        row = pd.DataFrame(
            [fundamentals],
            index=pd.DatetimeIndex(
                [pd.Timestamp(
                    datetime.datetime.now(datetime.timezone.utc).date()
                )],
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
    total_budget_sec: float = 180.0,
) -> dict[str, dict[str, float]]:
    """Fetch + cache fundamentals for every ticker. Returns a plain dict.

    Wraps the loop in a `FetchBudget` so a chain of slow yfinance
    responses can't eat more than `total_budget_sec` seconds of wall
    time. Once the budget is exhausted, remaining tickers silently
    skip (logged). Per-call timeouts still apply within
    `fetch_fundamentals`.
    """
    from .net_safety import FetchBudget, call_with_timeout  # noqa: PLC0415
    budget = FetchBudget(total_sec=total_budget_sec,
                          label="fetch_fundamentals_watchlist")
    out: dict[str, dict[str, float]] = {}
    per_ticker_sec = 90.0   # sum of 4 inner calls (each ≤ 20 s) + buffer
    for sym in watchlist:
        if budget.exhausted():
            log.warning("  %-6s — skipping (fundamentals budget exhausted)", sym)
            continue
        result = call_with_timeout(
            fetch_fundamentals, sym,
            timeout_sec = per_ticker_sec,
            label       = f"fundamentals.fetch({sym})",
            budget      = budget,
            cache       = cache,
            store       = store,
            provider_fn = provider_fn,
        )
        if result is not None:
            out[sym] = result
    return out


__all__ = [
    "FACTOR_COLS",
    "FundamentalsStore",
    "fetch_fundamentals",
    "fetch_fundamentals_watchlist",
]
