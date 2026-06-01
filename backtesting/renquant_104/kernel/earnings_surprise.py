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
        # Audit fix ES-READ-RACE (Round 2 deep audit, 2026-04-25):
        # mirror FU-4 / INT-READ-RACE — corrupt parquet (truncated mid-
        # write, disk-full partial flush) was raising and crashing the
        # caller; now treated as cache-miss so the next refetch refills
        # cleanly.
        p = self._path(symbol)
        if not p.exists():
            return None
        try:
            df = pd.read_parquet(p)
        except Exception as exc:
            log.warning(
                "EarningsSurpriseStore.load(%s): corrupt parquet — %s; "
                "treating as cache-miss", symbol, exc,
            )
            return None
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def save(self, df: pd.DataFrame, symbol: str) -> Path:
        # Audit fix ES-ATOM (Round 2 deep audit, 2026-04-25): atomic
        # write via .tmp + rename. Same as DC-2-CACHE / FU-1.
        p = self._path(symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        tmp = p.with_suffix(p.suffix + ".tmp")
        df.to_parquet(tmp)
        tmp.replace(p)
        return p


# ── Provider ──────────────────────────────────────────────────────────────────

def _fetch_from_yfinance(symbol: str) -> pd.DataFrame:
    """Fetch past earnings surprises via yfinance `.earnings_dates`.

    Returns an empty DataFrame on any failure (offline, rate-limited,
    unsupported ticker, or Yahoo slow-drip). Caller is expected to
    tolerate missing values — the z-score step sector-median-fills
    nulls. The 2026-04-23 incident was a yfinance hang on
    `.earnings_dates` with no timeout; now wrapped in a 20 s hard
    timeout via `renquant_common.net_safety.call_with_timeout`.
    """
    from renquant_common.net_safety import call_with_timeout  # noqa: PLC0415

    def _fetch():
        import yfinance as yf  # noqa: PLC0415
        return yf.Ticker(symbol).earnings_dates

    ed = call_with_timeout(
        _fetch, timeout_sec=20.0, label=f"yf.earnings_dates({symbol})",
    )
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
    refresh_after_days: float = 30.0,
) -> pd.DataFrame:
    """Load or fetch earnings-surprise history for `symbol`.

    Returns the cached DataFrame if available (fast path), else fetches via
    provider_fn (defaults to yfinance) and writes to cache.

    Round-3 audit (#R3-36): cache previously NEVER refreshed once written.
    New earnings announcements posted after the first cache write were
    invisible until manual deletion. Now: when the cache's most-recent
    announcement is older than `refresh_after_days` (default 30d — earnings
    are quarterly), incremental-fetch fresh history and merge in.
    """
    store = store or EarningsSurpriseStore()
    cached = None
    if cache:
        cached = store.load(symbol)
        if cached is not None and not cached.empty:
            latest = cached.index.max() if isinstance(cached.index, pd.DatetimeIndex) else None
            if latest is not None:
                age_days = (pd.Timestamp.now().normalize() - latest).days
                if age_days <= refresh_after_days:
                    return cached
            else:
                return cached

    fetch = provider_fn or _fetch_from_yfinance
    new_df = fetch(symbol)

    if cached is not None and not cached.empty and new_df is not None and not new_df.empty:
        merged = pd.concat([cached, new_df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = new_df if (new_df is not None and not new_df.empty) else (
            cached if cached is not None else new_df
        )

    if cache and merged is not None and not merged.empty:
        store.save(merged, symbol)
    return merged if merged is not None else pd.DataFrame(columns=SURPRISE_COLS)


def fetch_earnings_surprise_watchlist(
    watchlist: list[str],
    *,
    cache: bool = True,
    provider_fn: Callable[[str], pd.DataFrame] | None = None,
    total_budget_sec: float = 120.0,
    per_ticker_sec: float = 25.0,
) -> dict[str, pd.DataFrame]:
    """Per-ticker hard timeout + batch budget. Each ticker's
    `fetch_earnings_surprise` is wrapped in `call_with_timeout` so a
    single stalled yf.earnings_dates call can't block the loop."""
    from renquant_common.net_safety import FetchBudget, call_with_timeout
    budget = FetchBudget(total_sec=total_budget_sec,
                          label="fetch_earnings_surprise_watchlist")
    out: dict[str, pd.DataFrame] = {}
    for t in watchlist:
        if budget.exhausted():
            log.warning("  %-6s — skipping (earnings budget exhausted)", t)
            out[t] = pd.DataFrame(columns=SURPRISE_COLS)
            continue
        result = call_with_timeout(
            fetch_earnings_surprise, t,
            timeout_sec = per_ticker_sec,
            label       = f"earnings.fetch({t})",
            budget      = budget,
            cache       = cache,
            provider_fn = provider_fn,
        )
        out[t] = result if result is not None else pd.DataFrame(columns=SURPRISE_COLS)
    return out


# ── Factor computation ────────────────────────────────────────────────────────

# ── PEAD enrichment (Track B, Bernard-Thomas 1989, Chan-Jegadeesh-Lakonishok 1996)

def compute_pead_features(
    surprises: dict[str, pd.DataFrame],
    ohlcv: dict[str, pd.DataFrame],
    *,
    decay_window_days: int = 60,
    max_window_days: int = 90,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, pd.Series]]:
    """Three PEAD feature columns per ticker, ffilled to daily OHLCV index:

    Returns
    -------
    days_since_earnings : dict[ticker, Series]
        Calendar days since most-recent announcement, clamped to
        [0, max_window_days]. NaN before first announcement in window.

    pead_decay_weight : dict[ticker, Series]
        Linear ramp 1.0 at day 0 → 0.0 at decay_window_days. 0.0 past
        decay_window_days. NaN before first announcement.

    pead_signal : dict[ticker, Series]
        Most-recent surprise_pct × decay_weight. The canonical PEAD
        alpha — captures sign + magnitude + recency in one column.
        NaN before first announcement.

    No-lookahead: announcement INDEX is shifted +1 calendar day before
    reindex+ffill, mirroring `compute_earnings_surprise_cum` (#R3-35).
    Earnings releases are typically after-market — the post-release
    state first becomes available on the bar AFTER the announcement.

    References
    ----------
    Bernard & Thomas 1989: post-earnings drift strongest in days 1-30,
    decays mostly by day 60, ~zero past 90.
    Chan-Jegadeesh-Lakonishok 1996: surprise quintile sign+magnitude
    scales drift size.
    """
    days_out: dict[str, pd.Series] = {}
    decay_out: dict[str, pd.Series] = {}
    signal_out: dict[str, pd.Series] = {}

    for ticker, df_ohlcv in ohlcv.items():
        surprise_df = surprises.get(ticker)
        idx = df_ohlcv.index
        if surprise_df is None or surprise_df.empty or "surprise_pct" not in surprise_df.columns:
            days_out[ticker]   = pd.Series(np.nan, index=idx)
            decay_out[ticker]  = pd.Series(np.nan, index=idx)
            signal_out[ticker] = pd.Series(np.nan, index=idx)
            continue

        sp = surprise_df["surprise_pct"].sort_index()
        # Shift announcement index +1 day (lookahead-safe; same convention
        # as compute_earnings_surprise_cum).
        ann_index_shifted = sp.index + pd.Timedelta(days=1)

        # For each daily bar, find most-recent announcement at-or-before
        # that bar. Use a per-announcement Series that holds the
        # announcement date itself; reindex+ffill gives us the most-recent.
        ann_dates_sr = pd.Series(ann_index_shifted, index=ann_index_shifted)
        most_recent_ann = ann_dates_sr.reindex(idx, method="ffill")

        # Most-recent surprise value (forward-filled).
        sp_shifted_idx = sp.copy()
        sp_shifted_idx.index = ann_index_shifted
        most_recent_surprise = sp_shifted_idx.reindex(idx, method="ffill")

        # days_since = (idx_date - most_recent_ann_date).days
        # Pandas vectorisation: subtract two datetime Series → Timedelta Series → .dt.days
        days_since_raw = (pd.Series(idx, index=idx) - most_recent_ann).dt.days
        # Clamp + propagate NaN where most_recent_ann was NaT (pre-first-announcement)
        days_since = days_since_raw.where(~most_recent_ann.isna(), np.nan)
        days_since = days_since.clip(lower=0, upper=max_window_days)

        # decay_weight = max(0, 1 - days/decay_window_days)
        decay = (1.0 - days_since / float(decay_window_days)).clip(lower=0.0)
        decay = decay.where(~days_since.isna(), np.nan)

        # signal = most_recent_surprise × decay
        signal = most_recent_surprise * decay
        signal = signal.where(~days_since.isna(), np.nan)

        days_out[ticker]   = days_since
        decay_out[ticker]  = decay
        signal_out[ticker] = signal

    return days_out, decay_out, signal_out


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
        # Round-3 audit (#R3-35): shift announcement INDEX by +1 calendar day
        # so the cumulative value first becomes available on the bar AFTER
        # the announcement. Earnings releases are typically after-market —
        # using the post-release cumulative on the announcement day itself
        # would be lookahead. Shift the index (not .shift(1) on values which
        # produces NaN on the 1-announcement edge case) so the same value
        # is now associated with the next calendar day. The OHLCV reindex+
        # ffill below picks up that value on whichever trading day comes next.
        trailing.index = trailing.index + pd.Timedelta(days=1)
        daily = trailing.reindex(idx, method="ffill")
        out[ticker] = daily
    return out
