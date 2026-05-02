"""Empirical-Bayes shrinkage for per-sector percentile estimates — Layer 5
of the sector-aware ranking architecture.

The problem
-----------
Per-sector percentile (Layer 1) is the right answer for cross-sector
comparability. But when a sector has only N≈20–30 tickers on a given
date, the percentile is order-statistic-noisy: the gap between rank 1
and rank 2 has a standard deviation of order σ/√N ≈ 0.22σ — i.e. the
"top pick in Energy" is a coin flip in noisy regimes.

The fix (Robbins 1955, Wheeler 2018 practitioner note)
------------------------------------------------------
Shrink each per-sector percentile toward the global percentile, with
shrinkage weight inversely proportional to sector size::

    p_shrunk = (n / (n + k)) * p_sector + (k / (n + k)) * p_global

where ``n`` is the sector's ticker count and ``k`` is a hyperparameter
(start with k = 10). Big sectors (n >> k) → shrinkage ≈ 0 → p_shrunk
≈ p_sector. Small sectors (n << k) → shrinkage → 1 → p_shrunk
≈ p_global. This converts to the global benchmark exactly when the
local estimate is too noisy to trust.

Invariant
---------
For any input (p_sector ∈ [0, 1], p_global ∈ [0, 1], n ≥ 0, k > 0):
    p_shrunk ∈ [min(p_sector, p_global), max(p_sector, p_global)]
    p_shrunk is a convex combination — never extrapolates outside the
    convex hull of the two estimates.

References
----------
- Robbins, H. (1956). "An Empirical Bayes Approach to Statistics".
  Proc. Berkeley Symp. Math. Stat. Probab. 1: 157-163.
- Wheeler, A.P. (2018). "Sorting rates using empirical Bayes" —
  practitioner derivation: andrewpwheeler.com/2018/07/23/...
- Efron, B. & Morris, C. (1973). "Stein's Estimation Rule and Its
  Competitors". JASA 68: 117-130. Original James-Stein paper.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def eb_shrink_percentile(
    p_sector:  float,
    p_global:  float,
    n_sector:  int,
    *,
    k:         float = 10.0,
) -> float:
    """Shrink a single per-sector percentile toward the global percentile.

    Parameters
    ----------
    p_sector : float in [0, 1]
        Within-sector percentile rank for this ticker.
    p_global : float in [0, 1]
        Cross-sector (global) percentile rank for this ticker on the
        same date.
    n_sector : int ≥ 1
        Count of tickers in the same sector on this date (excluding
        non-finite or NaN entries).
    k : float > 0
        Shrinkage hyperparameter — equivalent prior sample size in the
        Robbins-Beta-Binomial setup. k = 10 means a sector of 10 tickers
        is shrunk halfway toward the global; a sector of 100 is shrunk
        ~9 % toward global. Default k = 10 is the standard
        practitioner choice.

    Returns
    -------
    float in [0, 1]
        The convex combination p_shrunk = w*p_sector + (1-w)*p_global,
        with w = n / (n + k). NaN inputs propagate to NaN.
    """
    if (
        not math.isfinite(p_sector) or not math.isfinite(p_global)
        or not math.isfinite(n_sector) or not math.isfinite(k)
    ):
        return float("nan")
    if n_sector < 0 or k <= 0:
        return float("nan")
    w = n_sector / (n_sector + k)
    return w * p_sector + (1.0 - w) * p_global


def eb_shrink_per_ticker(
    sector_pct:    dict[str, pd.Series],
    global_pct:    dict[str, pd.Series],
    sector_size:   dict[str, dict[str, int]] | dict[str, int],
    *,
    k:             float = 10.0,
) -> dict[str, pd.Series]:
    """Vectorized EB shrinkage across the full panel.

    Parameters
    ----------
    sector_pct : dict[ticker, pd.Series]
        Per-(date) within-sector percentile, output of
        ``cross_sectional_rank_within_sector``.
    global_pct : dict[ticker, pd.Series]
        Per-(date) global cross-sectional percentile across the full
        universe (no sector grouping).
    sector_size : dict[ticker, dict[date, int]] OR dict[ticker, int]
        Sector size on each date for each ticker. The dict-of-dict form
        allows per-date variation (e.g. a sector grew over time); the
        dict-of-int form treats sector size as constant across dates
        (acceptable if the panel is short enough that no constituent
        churn happened).
    k : float > 0
        Shrinkage hyperparameter (default 10).

    Returns
    -------
    dict[ticker, pd.Series]
        Shrunken percentiles per ticker. NaN propagates from any of
        the inputs.

    Notes
    -----
    NaN handling is critical: a NaN p_sector + valid p_global must NOT
    silently fall back to global (that would mask sector-data missing).
    Either both are valid OR the result is NaN. Caller decides whether
    to use a fall-back.
    """
    out: dict[str, pd.Series] = {}
    for ticker, ps_series in sector_pct.items():
        pg_series = global_pct.get(ticker)
        if pg_series is None:
            out[ticker] = pd.Series(
                np.nan, index=ps_series.index, name=ticker,
            )
            continue
        # Align on the union of indices (caller may have sparse data
        # on either path). Reindexing fills missing dates with NaN
        # which is the correct behavior for unobserved dates.
        idx = ps_series.index.union(pg_series.index)
        ps = ps_series.reindex(idx).astype(float)
        pg = pg_series.reindex(idx).astype(float)

        # Resolve per-date n_sector for this ticker
        if isinstance(sector_size, dict) and ticker in sector_size:
            sz = sector_size[ticker]
            if isinstance(sz, dict):
                # date → n
                n_arr = np.array(
                    [sz.get(d, np.nan) for d in idx], dtype=float,
                )
            else:
                # constant
                n_arr = np.full(len(idx), float(sz))
        else:
            n_arr = np.full(len(idx), np.nan)

        with np.errstate(invalid="ignore"):
            w = n_arr / (n_arr + k)
            shrunk = w * ps.values + (1.0 - w) * pg.values

        # Result is NaN whenever any input is NaN (preserves the
        # data-missing signal — caller decides fallback).
        invalid = (~np.isfinite(ps.values)) | (~np.isfinite(pg.values)) | (~np.isfinite(n_arr))
        shrunk = np.where(invalid, np.nan, shrunk)

        out[ticker] = pd.Series(shrunk, index=idx, name=ticker)
    return out


def compute_sector_size_per_date(
    sector_pct:     dict[str, pd.Series],
    ticker_sectors: dict[str, str],
) -> dict[str, dict[str, int]]:
    """Helper: derive per-date sector size from a sector_pct dict.

    For each (ticker, date), counts how many tickers in the same sector
    have a non-NaN percentile on that date. Returns
    {ticker: {date_iso: count}}.
    """
    # Flatten to a long frame (date, ticker, sector, has_value)
    rows = []
    for ticker, s in sector_pct.items():
        sector = ticker_sectors.get(ticker, "_unmapped")
        for date, val in s.items():
            rows.append({
                "date":   date,
                "ticker": ticker,
                "sector": sector,
                "valid":  bool(pd.notna(val)),
            })
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    sizes = (
        df[df["valid"]]
        .groupby(["date", "sector"])
        .size()
        .rename("n")
        .reset_index()
    )
    # Build {ticker: {date: n}} by joining each ticker back to its sector
    out: dict[str, dict[str, int]] = {}
    for ticker, sector in ticker_sectors.items():
        sub = sizes[sizes["sector"] == sector]
        out[ticker] = {row["date"]: int(row["n"]) for _, row in sub.iterrows()}
    return out
