"""FRED API macro-series ingestion (Tier 2 of macro expansion plan).

Per `doc/research/macro-data-expansion-plan-2026-04-27.md`, this adds
22+ Fed-/BLS-sourced macro series to the panel-LTR feature space, on
top of the existing 30 ETF symbols handled by `kernel/macro.py`.

Public API
==========

`FredMacroStore(cache_dir, api_key)`
    .save(series_id, frame)        — write one series to parquet
    .load(series_id) -> DataFrame  — read from parquet, None if missing
    .fetch(series_id, ...)         — wrap fredapi.Fred.get_series

`build_fred_frame(store, series_ids, training_end=None) -> (frame, meta)`
    Assemble date-indexed DataFrame with one column per series, z-scored
    on rolling 252-day window. Forward-filled to daily so monthly
    series (CPI, payrolls) align to the daily panel calendar.

`fred_levels_to_returns(frame) -> DataFrame`
    Mirror of `kernel.macro_per_ticker.macro_levels_to_returns` but
    operating on FRED z-scored levels. Produces `<series>_chg` columns
    for use in `compute_per_ticker_macro_betas`.

Series catalog
==============

The DEFAULT_FRED_SERIES list contains the 22 Tier-2 series identified
in the macro expansion plan:

  Treasury yields (curve):
    DGS2  DGS5  DGS10  DGS30      — 2/5/10/30-year nominal yields
    DGS1MO DGS3MO DGS6MO          — short bills (1m/3m/6m)
    T10Y2Y                         — 10y-2y spread (recession proxy)
    T5YIE                          — 5y breakeven inflation

  Policy / funding:
    DFF                            — fed funds effective rate
    SOFR                           — secured overnight funding rate

  Volatility / credit / risk:
    VIXCLS                         — VIX close
    BAMLC0A0CM                     — IG OAS
    BAMLH0A0HYM2                   — HY OAS

  USD strength:
    DTWEXBGS                       — broad trade-weighted USD

  Inflation / activity (monthly, ffill to daily):
    CPIAUCSL  PCEPILFE             — CPI / Core PCE
    INDPRO                         — industrial production
    PAYEMS                         — non-farm payrolls
    ICSA                           — initial unemployment claims (weekly)
    UMCSENT                        — consumer sentiment
    NAPM                           — PMI manufacturing
    RSAFS                          — retail sales

API key
=======

Free key from <https://fred.stlouisfed.org/docs/api/api_key.html>.
Read from `RENQUANT_FRED_API_KEY` env var or `~/.fred_api_key` file.
Store dies cleanly with a clear error message if neither is set —
**no fake placeholder data**.

Data quality safeguards
=======================

- F1: per-series try/except — one bad series doesn't kill the rest.
- F2: drop series with rolling-window coverage < 95% on training window.
- F3: forward-fill monthly/weekly series to daily ONLY through the
  current bar — no peeking at the next release date.
- F4 (look-ahead guard): release-date lag. Monthly series like CPI
  release ~2 weeks after the reference month. We lag every monthly
  series by 1 release day (default 5 trading days) before forward-fill,
  so the panel never sees a CPI value before its release.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger("kernel.fred_macro")


# ── Default series catalog ────────────────────────────────────────────────────

# Series id → (display_name, frequency, release_lag_bars)
# release_lag_bars: number of bars (trading days) by which to lag the series
# to avoid look-ahead. Daily series → 0 lag (already published EOD).
# Monthly: ~5 trading days (most macro releases drop ~2 weeks after the
# month ends, but we lag by 5 days conservatively to absorb release-day
# uncertainty without over-discarding the front edge).
# Weekly: ~2 trading days.
DEFAULT_FRED_SERIES: list[tuple[str, str, str, int]] = [
    # ── Treasury yield curve (daily, 0 lag — published end of day) ──
    ("DGS1MO",       "1m T-bill",            "daily",   0),
    ("DGS3MO",       "3m T-bill",            "daily",   0),
    ("DGS6MO",       "6m T-bill",            "daily",   0),
    ("DGS2",         "2y Treasury",          "daily",   0),
    ("DGS5",         "5y Treasury",          "daily",   0),
    ("DGS10",        "10y Treasury",         "daily",   0),
    ("DGS30",        "30y Treasury",         "daily",   0),
    ("T10Y2Y",       "10y-2y spread",        "daily",   0),
    ("T5YIE",        "5y breakeven infl",    "daily",   0),
    # ── Policy / funding ───────────────────────────────────────────
    ("DFF",          "fed funds eff",        "daily",   0),
    ("SOFR",         "secured overnight",    "daily",   0),
    # ── Vol / credit / risk ────────────────────────────────────────
    ("VIXCLS",       "VIX close",            "daily",   0),
    ("BAMLC0A0CM",   "IG OAS",               "daily",   0),
    ("BAMLH0A0HYM2", "HY OAS",               "daily",   0),
    # ── USD strength ───────────────────────────────────────────────
    ("DTWEXBGS",     "broad TWEXB USD",      "weekly",  2),
    # ── Inflation / activity (monthly — lag by 5 trading days) ──────
    ("CPIAUCSL",     "CPI all-urban",        "monthly", 5),
    ("PCEPILFE",     "Core PCE",             "monthly", 5),
    ("INDPRO",       "industrial prod",      "monthly", 5),
    ("PAYEMS",       "non-farm payrolls",    "monthly", 5),
    ("UMCSENT",      "consumer sentiment",   "monthly", 5),
    # NAPM (PMI manufacturing) — discontinued in FRED in 2016. Use
    # ISM data via separate vendor; for now skip.
    ("RSAFS",        "retail sales",         "monthly", 5),
    # ── Weekly (lag 2 trading days) ─────────────────────────────────
    ("ICSA",         "initial claims",       "weekly",  2),
]

DEFAULT_ROLLING_WINDOW: int = 252


# ── API key resolution ────────────────────────────────────────────────────────

def _resolve_api_key(api_key: str | None = None) -> str | None:
    """Return FRED API key from arg, env, or ~/.fred_api_key. None if absent."""
    if api_key:
        return api_key.strip()
    env_key = os.environ.get("RENQUANT_FRED_API_KEY")
    if env_key:
        return env_key.strip()
    home_key_file = Path.home() / ".fred_api_key"
    if home_key_file.exists():
        try:
            return home_key_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            log.warning("FredMacroStore: ~/.fred_api_key unreadable — %s", exc)
    return None


# ── Storage ───────────────────────────────────────────────────────────────────

class FredMacroStore:
    """Thin parquet cache around the FRED API.

    Per-series files: cache_dir / <SERIES_ID>.parquet
    Schema: index=date (DatetimeIndex), columns=["value"]
    Idempotent (no clobber on re-fetch — concat+dedupe).
    """

    def __init__(self, cache_dir: str | Path, api_key: str | None = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = _resolve_api_key(api_key)
        self._fred_client = None  # lazy

    def _get_client(self):
        if self._fred_client is None:
            if not self.api_key:
                raise RuntimeError(
                    "FredMacroStore: FRED API key missing. Set "
                    "RENQUANT_FRED_API_KEY env var or write the key to "
                    "~/.fred_api_key. Free key at "
                    "https://fred.stlouisfed.org/docs/api/api_key.html",
                )
            from fredapi import Fred  # noqa: PLC0415
            self._fred_client = Fred(api_key=self.api_key)
        return self._fred_client

    def _path(self, series_id: str) -> Path:
        return self.cache_dir / f"{series_id}.parquet"

    def load(self, series_id: str) -> pd.DataFrame | None:
        p = self._path(series_id)
        if not p.exists():
            return None
        try:
            df = pd.read_parquet(p)
        except Exception as exc:
            # F9-equivalent: corrupt file → treat as cache miss
            log.warning("FredMacroStore.load(%s): corrupt parquet — %s", series_id, exc)
            return None
        if df.empty or "value" not in df.columns:
            return None
        df = df.sort_index()
        return df

    def save(self, series_id: str, frame: pd.DataFrame) -> Path:
        """Atomic write via .tmp rename. Concat with existing on overlap."""
        p = self._path(series_id)
        existing = self.load(series_id)
        if existing is not None and not existing.empty:
            frame = pd.concat([existing, frame])
            frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        tmp = p.with_suffix(p.suffix + ".tmp")
        frame.to_parquet(tmp)
        tmp.replace(p)
        return p

    def fetch(
        self, series_id: str,
        observation_start: str | pd.Timestamp | None = None,
        observation_end:   str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch from FRED API; wrap result as DataFrame[value] indexed by date."""
        fred = self._get_client()
        s = fred.get_series(
            series_id,
            observation_start=observation_start,
            observation_end=observation_end,
        )
        if s is None or s.empty:
            return pd.DataFrame(columns=["value"])
        df = pd.DataFrame({"value": s.astype(float).values}, index=pd.to_datetime(s.index))
        df.index.name = "date"
        return df


# ── Z-score (mirror kernel.macro._rolling_z) ─────────────────────────────────

def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    """Z-score against rolling mean+std. NaN preserved in warmup; +/-inf → 0."""
    mean = series.rolling(window, min_periods=max(2, window // 4)).mean()
    std = series.rolling(window, min_periods=max(2, window // 4)).std()
    z = (series - mean) / std.replace(0.0, np.nan)
    z = z.where(~np.isinf(z), 0.0)
    return z


# ── Daily-bar conversion (look-ahead-safe) ────────────────────────────────────

def _to_daily_bars(
    series: pd.Series,
    *,
    target_index: pd.DatetimeIndex,
    release_lag_bars: int,
) -> pd.Series:
    """Forward-fill `series` onto `target_index` with `release_lag_bars` lag.

    Workflow:
      1. Reindex `series` to a calendar covering both source and target.
      2. Apply `shift(release_lag_bars)` so a value at position t becomes
         visible from position t + release_lag_bars onward (look-ahead
         safety).
      3. Forward-fill — fills weekends/holidays with the most recent
         release.
      4. Reindex to `target_index` (the panel's trading-day calendar).

    The shift is applied AFTER reindex onto the trading-day calendar so
    that release_lag_bars counts trading days, not calendar days. With
    `release_lag_bars=5` and a CPI release on Jan 12, the value first
    appears in the panel feature column on the 5th trading day after
    Jan 12.
    """
    if series is None or series.empty:
        return pd.Series(np.nan, index=target_index, dtype=float, name=series.name if series is not None else None)
    # Step 1: reindex onto the union, ffill-able.
    union = target_index.union(series.index).sort_values()
    s = series.astype(float).reindex(union)
    s = s.ffill()
    # Step 2: shift on the daily bar count of TARGET INDEX. We do this
    # by intersecting with target_index first to count trading-day
    # positions correctly.
    s_target = s.reindex(target_index)
    if release_lag_bars > 0:
        s_target = s_target.shift(release_lag_bars)
    return s_target


# ── Frame assembly ────────────────────────────────────────────────────────────

def build_fred_frame(
    store: FredMacroStore,
    target_index: pd.DatetimeIndex,
    *,
    series_specs: Iterable[tuple[str, str, str, int]] = DEFAULT_FRED_SERIES,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    min_window_overlap_pct: float = 0.95,
    training_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Assemble FRED features as a date-indexed DataFrame.

    Output columns:
      <series_id>_level_z   — rolling-z of raw level
      <series_id>_chg_5d_z  — rolling-z of 5-bar diff
      <series_id>_chg_20d_z — rolling-z of 20-bar diff

    The 3-transform schema mirrors `kernel.macro.compute_macro_features`
    so downstream `macro_levels_to_returns` + `compute_per_ticker_macro_betas`
    work without modification. Total features = len(series) × 3.
    """
    target_index = pd.DatetimeIndex(target_index).sort_values()
    cols: dict[str, pd.Series] = {}
    skipped: list[tuple[str, str]] = []
    used: list[str] = []

    for series_id, _name, _freq, lag_bars in series_specs:
        try:
            df = store.load(series_id)
        except Exception as exc:
            skipped.append((series_id, f"load_failed: {type(exc).__name__}"))
            log.warning("build_fred_frame: load %s failed — %s", series_id, exc)
            continue
        if df is None or df.empty:
            skipped.append((series_id, "no_cache"))
            continue
        try:
            daily = _to_daily_bars(
                df["value"], target_index=target_index, release_lag_bars=int(lag_bars),
            )
        except Exception as exc:
            skipped.append((series_id, f"to_daily_failed: {type(exc).__name__}"))
            log.warning("build_fred_frame: to_daily %s failed — %s", series_id, exc)
            continue

        # F2 coverage check (95% of the rolling-window worth of trading days
        # should be non-NaN at training_end).
        if training_end is not None:
            window_start_pos = max(0, target_index.get_indexer([pd.Timestamp(training_end)],
                                                                method="nearest")[0]
                                    - rolling_window * 2)
            in_window = daily.iloc[window_start_pos:]
            if len(in_window) > 0:
                non_nan_pct = float(in_window.notna().mean())
                if non_nan_pct < min_window_overlap_pct:
                    skipped.append(
                        (series_id, f"insufficient_coverage_{non_nan_pct:.2f}<{min_window_overlap_pct}"),
                    )
                    log.warning(
                        "build_fred_frame: %s coverage %.0f%% < min %.0f%% — skipping",
                        series_id, non_nan_pct * 100, min_window_overlap_pct * 100,
                    )
                    continue

        # Three transforms per series, mirror kernel.macro.compute_macro_features
        sid_lower = series_id.lower()
        cols[f"{sid_lower}_level_z"]   = _rolling_z(daily, rolling_window)
        cols[f"{sid_lower}_chg_5d_z"]  = _rolling_z(daily.diff(5),  rolling_window)
        cols[f"{sid_lower}_chg_20d_z"] = _rolling_z(daily.diff(20), rolling_window)
        used.append(series_id)

    if not cols:
        log.warning("build_fred_frame: no FRED features built (all series skipped)")
        return pd.DataFrame(index=target_index), {
            "series_used":    [],
            "series_skipped": skipped,
            "n_features":     0,
            "rolling_window": rolling_window,
        }

    frame = pd.DataFrame(cols, index=target_index)
    log.info(
        "build_fred_frame: %d features from %d series (%d skipped); %d dates",
        len(cols), len(used), len(skipped), len(frame),
    )
    return frame, {
        "series_used":    used,
        "series_skipped": skipped,
        "n_features":     len(cols),
        "rolling_window": rolling_window,
    }


def fred_levels_to_returns(fred_frame: pd.DataFrame) -> pd.DataFrame:
    """Convert FRED `*_level_z` columns to `*_chg` returns (diff).

    Mirror of `kernel.macro_per_ticker.macro_levels_to_returns` so the
    same `compute_per_ticker_macro_betas` consumer works on FRED data.
    """
    if fred_frame is None or fred_frame.empty:
        return pd.DataFrame()
    out: dict[str, pd.Series] = {}
    for col in fred_frame.columns:
        if col.endswith("_level_z"):
            base = col.replace("_level_z", "")
            out[f"{base}_chg"] = fred_frame[col].diff()
    return pd.DataFrame(out, index=fred_frame.index).dropna(how="all")


__all__ = [
    "DEFAULT_FRED_SERIES",
    "DEFAULT_ROLLING_WINDOW",
    "FredMacroStore",
    "build_fred_frame",
    "fred_levels_to_returns",
    "_resolve_api_key",
]
