"""Layered NaN / newly-listed-ticker handling for Stage-1 LTR.

Four composable pieces applied in order during panel assembly:

  1. `apply_min_history_gate`   — drop the first N bars per ticker so that
                                   indicator warmup is stable.
  2. `add_missingness_indicators` — explicit `{col}_is_missing ∈ {0,1}`
                                   columns so the model can learn that
                                   "missing" itself is informative.
  3. `sector_median_fill`       — fill remaining NaNs with the same-date
                                   same-sector median.
  4. `compute_age_weight`       — linear ramp from 0 at listing date to
                                   1.0 at `warmup_days`, capped thereafter.

None of these functions mutate their inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_min_history_gate(
    feature_frames: dict[str, pd.DataFrame],
    min_history_days: int = 252,
) -> dict[str, pd.DataFrame]:
    """Drop the first `min_history_days` bars of each per-ticker frame.

    Tickers whose history is shorter than `min_history_days` are dropped
    entirely (they have no usable rows after the gate).
    """
    out: dict[str, pd.DataFrame] = {}
    for t, ff in feature_frames.items():
        if len(ff) <= min_history_days:
            continue
        out[t] = ff.iloc[min_history_days:].copy()
    return out


def add_missingness_indicators(
    panel: pd.DataFrame, cols: list[str],
) -> pd.DataFrame:
    """Append `{col}_is_missing` (int8, 0/1) for each col in `cols`.

    Missing columns in `panel` are silently skipped.
    """
    panel = panel.copy()
    for col in cols:
        if col not in panel.columns:
            continue
        panel[f"{col}_is_missing"] = panel[col].isna().astype(np.int8)
    return panel


def sector_median_fill(
    panel: pd.DataFrame, cols: list[str], *,
    sector_col: str = "sector", date_col: str = "date",
) -> pd.DataFrame:
    """Fill NaN cells in `cols` with the same-date same-sector median.

    Rows whose (date, sector) bucket has no finite values fall back to the
    same-date cross-sectional median. If even that is unavailable, NaN
    remains.
    """
    panel = panel.copy()
    if not cols:
        return panel

    # Per (date, sector) medians
    group = panel.groupby([date_col, sector_col], sort=False)
    bucket_medians = group[cols].transform("median")

    # Per-date fallback
    date_group = panel.groupby(date_col, sort=False)
    date_medians = date_group[cols].transform("median")

    for col in cols:
        if col not in panel.columns:
            continue
        target = panel[col]
        fill = bucket_medians[col].where(bucket_medians[col].notna(), date_medians[col])
        panel[col] = target.where(target.notna(), fill)
    return panel


def forward_fill_per_ticker(
    panel: pd.DataFrame, cols: list[str], *,
    max_gap_days: int = 5,
    ticker_col: str = "ticker",
    date_col: str = "date",
) -> pd.DataFrame:
    """Forward-fill NaN values in ``cols`` WITHIN each ticker's time series,
    capped at ``max_gap_days`` consecutive NaN to avoid stale-value leak.

    2026-05-04 — added in response to user spec "数据消失时 forward fill".
    Targeted only at slow-moving features (whitelist via config). DO NOT
    apply to high-frequency intraday-derived features — yesterday's
    afternoon-drift z-score has no information about today's, so ffill
    on those is just noise injection.

    The cap is enforced by counting consecutive NaN per ticker per col;
    runs longer than ``max_gap_days`` keep the trailing values as NaN.
    Use sector_median_fill or row_coverage filter for the long-gap case.

    Parameters
    ----------
    panel : DataFrame
        Long-form panel (one row per ticker × date). Must have ``ticker_col``
        and ``date_col``.
    cols : list[str]
        Whitelist of columns to forward-fill. Empty list → no-op.
    max_gap_days : int
        Maximum consecutive NaN run to fill (calendar-row count, not
        calendar-day; aligns with bar count). Default 5.

    Returns a NEW DataFrame; does not mutate input.
    """
    if not cols or panel.empty:
        return panel
    panel = panel.copy()
    cols_present = [c for c in cols if c in panel.columns]
    if not cols_present:
        return panel
    # Sort within ticker; pandas groupby's `ffill(limit=N)` enforces the cap.
    panel = panel.sort_values([ticker_col, date_col], kind="mergesort")
    panel[cols_present] = panel.groupby(
        ticker_col, group_keys=False, sort=False,
    )[cols_present].ffill(limit=max_gap_days)
    return panel


def compute_age_weight(
    panel: pd.DataFrame,
    listing_dates: dict[str, pd.Timestamp],
    warmup_days: int = 504,
    *,
    ticker_col: str = "ticker",
    date_col: str = "date",
) -> pd.Series:
    """Linear ramp: min(1, days_since_listing / warmup_days). Returns a
    Series aligned to panel.index with dtype float.

    Tickers not present in `listing_dates` are treated as seasoned (weight
    1.0). Rows where the bar date precedes the listing date get weight 0.0.
    """
    out = np.ones(len(panel), dtype=float)
    if not listing_dates:
        return pd.Series(out, index=panel.index)

    tickers = panel[ticker_col].values
    dates   = pd.to_datetime(panel[date_col]).values.astype("datetime64[ns]")
    for i in range(len(panel)):
        listing = listing_dates.get(tickers[i])
        if listing is None:
            continue
        age_days = (pd.Timestamp(dates[i]) - pd.Timestamp(listing)).days
        if age_days <= 0:
            out[i] = 0.0
        elif age_days >= warmup_days:
            out[i] = 1.0
        else:
            out[i] = age_days / warmup_days
    return pd.Series(out, index=panel.index)
