"""Bulk triple-barrier labeler — apply ``apply_triple_barrier`` over a
DataFrame of (date, ticker, ...) snapshots produced by P4.1's
SnapshotLogger.

Wraps :func:`apply_triple_barrier` for the case where one wants to
score N events at once against a set of pre-fetched close-price paths.

References
----------
* López de Prado 2018 AFML ch.3 (Snippet 3.4) — single-event algorithm
  in ``triple_barrier.py``
* López de Prado 2018 AFML ch.20 — meta-labeling adaptation
"""
from __future__ import annotations

from typing import Mapping

import math
import numpy as np
import pandas as pd

from .triple_barrier import apply_triple_barrier

PATH_TRIGGER_COLUMNS: tuple[str, ...] = (
    "trigger_stop_loss",
    "trigger_trailing_stop",
    "trigger_single_day_loss",
    "trigger_max_hold",
)


def _has_path_rule_trigger(row: pd.Series) -> bool:
    for col in PATH_TRIGGER_COLUMNS:
        try:
            if int(row.get(col, 0) or 0) == 1:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _resolve_sigma_daily(row: pd.Series, default_sigma_daily: float) -> float:
    """Get a daily-σ estimate for a snapshot row.

    Priority:
      1. ``realized_vol_20d`` column (annualised) → / sqrt(252)
      2. ``default_sigma_daily`` (callable kwarg)
    """
    v = row.get("realized_vol_20d", None)
    if v is None or not math.isfinite(float(v)) or float(v) == 0.0:
        return default_sigma_daily
    # realized_vol_20d is annualised → convert to daily
    return float(v) / math.sqrt(252.0)


def _fwd_geometric_return(close: pd.Series, anchor: pd.Timestamp, window: int) -> float:
    """Compute the geometric forward return from `anchor + 1` over `window` bars.

    Returns NaN if anchor not in series or fewer than `window` future bars.
    """
    if anchor not in close.index:
        return float("nan")
    pos = close.index.get_loc(anchor)
    if pos + window >= len(close):
        return float("nan")
    p0 = float(close.iloc[pos])
    pN = float(close.iloc[pos + window])
    if not (math.isfinite(p0) and math.isfinite(pN)) or p0 <= 0:
        return float("nan")
    return pN / p0 - 1.0


def label_snapshots(
    snapshot_df: pd.DataFrame,
    close_paths: Mapping[str, pd.Series],
    *,
    pt_mult: float = 10.0,
    sl_mult: float = 10.0,
    default_sigma_daily: float = 0.01,
    fwd_window: int = 20,
    fwd_short_window: int = 5,
) -> pd.DataFrame:
    """Apply triple-barrier labeling + forward-return computation to all
    rows of a snapshot DataFrame.

    Parameters
    ----------
    snapshot_df : pd.DataFrame
        Per-day per-position rows in the schema produced by
        ``kernel.meta_label.SnapshotLogger.dump_to_parquet``. Must
        contain at least ``date`` (ISO string), ``ticker`` (str),
        ``any_trigger`` (0/1).
    close_paths : Mapping[str, pd.Series]
        One close-price series per ticker, indexed by date.
    pt_mult, sl_mult, default_sigma_daily, fwd_window :
        Triple-barrier params (see ``apply_triple_barrier``).
    fwd_short_window : int, default 5
        Bars for the short forward-return column.

    Returns
    -------
    pd.DataFrame
        Copy of ``snapshot_df`` with three columns populated/added:
        ``fwd_5d_ret``, ``fwd_20d_ret`` (forward returns regardless of
        trigger), and ``meta_label`` (0/1 only for rows with a path-rule
        trigger eligible for MetaLabelVetoTask; NaN otherwise).
    """
    out = snapshot_df.copy()
    # Ensure the column shape
    for col in ("fwd_5d_ret", "fwd_20d_ret", "meta_label"):
        if col not in out.columns:
            out[col] = float("nan")

    if len(out) == 0:
        return out

    # Index out by row position for safe in-place assignment
    out = out.reset_index(drop=True)

    for i, row in out.iterrows():
        ticker = str(row["ticker"]) if pd.notna(row["ticker"]) else None
        date_iso = str(row["date"])  if pd.notna(row["date"])  else None
        if not ticker or not date_iso:
            continue
        close = close_paths.get(ticker)
        if close is None or len(close) == 0:
            continue
        try:
            anchor = pd.Timestamp(date_iso)
        except (TypeError, ValueError):
            continue
        if anchor not in close.index:
            # Snapshot date not in calendar (rare — sim might have
            # truncated). Skip — leave label NaN.
            continue

        # Forward returns regardless of trigger
        out.at[i, "fwd_5d_ret"]  = _fwd_geometric_return(close, anchor, fwd_short_window)
        out.at[i, "fwd_20d_ret"] = _fwd_geometric_return(close, anchor, fwd_window)

        # Meta-label only for rows with a path-rule trigger. Older snapshots
        # may have any_trigger=1 for model/QP exits; those are not eligible at
        # inference time and must not contaminate the training target.
        any_trig = row.get("any_trigger", 0)
        try:
            any_trig_i = int(any_trig) if pd.notna(any_trig) else 0
        except (TypeError, ValueError):
            any_trig_i = 0
        if any_trig_i != 1 or not _has_path_rule_trigger(row):
            continue

        event_price = float(close.loc[anchor])
        sigma_daily = _resolve_sigma_daily(row, default_sigma_daily)
        res = apply_triple_barrier(
            close,
            entry_idx=anchor,
            entry_price=event_price,
            pt_mult=pt_mult,
            sl_mult=sl_mult,
            sigma_daily=sigma_daily,
            max_horizon_days=fwd_window,
            return_terminal_sign=True,
        )
        if res is None:
            continue
        afml_label, _, _ = res
        # AFML -1 → continued fall → meta = 1 (correct exit)
        # AFML +1 / 0 → recovery → meta = 0 (false-positive exit)
        out.at[i, "meta_label"] = 1 if afml_label == -1 else 0

    return out
