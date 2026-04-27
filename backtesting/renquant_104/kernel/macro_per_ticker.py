"""Per-ticker rolling β to macro factors — Macro v2 (2026-04-27).

Per the v1 → v2 redesign documented in
`doc/components/macro-factor-frame-redesign.md`:

The v1 macro frame broadcast identical macro values to every ticker
on each date — providing ZERO within-date variance for cross-sectional
rank loss. v2 instead computes PER-TICKER rolling β to each macro
factor, producing values that DIFFER per ticker on the same date and
therefore enter the rank loss as proper differentiation features.

Public API
==========

`compute_per_ticker_macro_betas(ohlcv, macro_returns, *,
    rolling_window=60, min_window=30) -> dict[ticker, DataFrame]`

For each ticker, returns a DataFrame indexed by date with columns
`beta_<macro_factor>_<window>d` for each macro symbol. Strict-prior
discipline: β at bar `t` is computed from data [t-rolling_window, t-1]
only. Result shifted by 1 to ensure no look-ahead leak.

Used as additional per-ticker features in `factor_frames` (alongside
size_z, mom_12_1_z, beta_60d_z, resid_mom_z), so they go through
existing FactorZScoreTask cross-sectional z-score before reaching the
panel-LTR ranker.

References
==========
- Kelly, Pruitt, Su (2019) "Characteristics are Covariances" — IPCA
  framework where per-stock factor exposures (β to macro) drive
  cross-sectional return prediction.
- Microsoft Qlib (`qlib/contrib/data/handler.py::Alpha158`) — same
  pattern: macro factors enter as per-stock derived quantities, never
  as broadcast features.
- Vasicek (1973) — Bayesian shrinkage toward 1.0 (market β) for noisy
  rolling β. NOT applied in v2 initial implementation (TODO if rolling
  β proves too noisy in A/B).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("kernel.macro_per_ticker")


def compute_per_ticker_macro_betas(
    ohlcv: dict[str, pd.DataFrame],
    macro_returns: pd.DataFrame,
    *,
    rolling_window: int = 60,
    min_window: int = 30,
) -> dict[str, pd.DataFrame]:
    """Per-ticker rolling β to each macro factor.

    Parameters
    ----------
    ohlcv : dict[ticker, DataFrame]
        Per-ticker OHLCV. Each DataFrame must contain a 'close' column
        and be indexed by date.
    macro_returns : pd.DataFrame
        Date-indexed; columns are macro factor returns (already converted
        to returns from levels — caller's responsibility). Typical
        columns: vxx_chg, hyg_chg, uup_chg, etc.
    rolling_window : int, default 60
        Lookback window for OLS β. β_t uses [t-rolling_window, t-1].
    min_window : int, default 30
        Minimum data points required to compute a β; below this, β is NaN.

    Returns
    -------
    dict[ticker, DataFrame]
        For each ticker, DataFrame indexed by date with one column per
        macro factor: `beta_<factor>_<rolling_window>d`. Values are
        shift(1)'d to ensure strict-prior discipline (β at bar t uses
        only data up to bar t-1).

    Notes
    -----
    F1-F5 safety harness compatible:
    - F1 per-symbol load isolation: each ticker computed independently;
      one ticker's failure doesn't affect others.
    - F2 minimum-data guard: rolling().cov() returns NaN when fewer
      than min_periods samples are available.
    - F3 zero-variance protection: var=0 → division by zero handled
      via .replace(0, np.nan) → β becomes NaN for that bar.
    """
    out: dict[str, pd.DataFrame] = {}

    if macro_returns is None or macro_returns.empty:
        log.warning("compute_per_ticker_macro_betas: macro_returns empty — returning {}")
        return out

    macro_cols = list(macro_returns.columns)

    for ticker, df in ohlcv.items():
        if df is None or df.empty or "close" not in df.columns:
            continue

        # Per-ticker daily returns (close-to-close)
        ticker_returns = df["close"].pct_change()

        if len(ticker_returns) < min_window:
            log.debug(
                "compute_per_ticker_macro_betas: %s has only %d bars, "
                "below min_window=%d — skipping",
                ticker, len(ticker_returns), min_window,
            )
            continue

        cols: dict[str, pd.Series] = {}

        for macro_col in macro_cols:
            macro_r = macro_returns[macro_col].reindex(ticker_returns.index)

            # Rolling OLS β = Cov(stock, macro) / Var(macro)
            cov = ticker_returns.rolling(
                rolling_window, min_periods=min_window
            ).cov(macro_r)
            var = macro_r.rolling(
                rolling_window, min_periods=min_window
            ).var()

            # F3 zero-variance protection — divide by NaN where var=0
            beta = cov / var.replace(0, np.nan)

            # Strict-prior shift: β at bar t computed from [t-window, t-1]
            # — without shift, β at t includes t in the window. Shift by
            # 1 to ensure the value for "today" excludes today's data.
            cols[f"beta_{macro_col}_{rolling_window}d"] = beta.shift(1)

        out[ticker] = pd.DataFrame(cols, index=ticker_returns.index)

    return out


def macro_levels_to_returns(macro_levels: pd.DataFrame) -> pd.DataFrame:
    """Convert macro factor LEVELS (z-scored prices) to RETURNS.

    The v1 `kernel.macro::build_macro_frame` produces z-scored levels
    (vxx_level_z, hyg_level_z, etc.). For β computation we need
    returns; this helper produces a 1-day-difference proxy.

    Convention: name columns `<symbol>_chg` (e.g. vxx_chg, hyg_chg).
    """
    if macro_levels is None or macro_levels.empty:
        return pd.DataFrame()

    out: dict[str, pd.Series] = {}
    for col in macro_levels.columns:
        # Heuristic: pick only the *_level_z columns; chg_*d_z columns
        # are already differenced. We diff levels to get clean returns.
        if col.endswith("_level_z"):
            base = col.replace("_level_z", "")
            out[f"{base}_chg"] = macro_levels[col].diff()
        # else: skip — chg_5d / chg_20d are smoothed, not point returns

    return pd.DataFrame(out, index=macro_levels.index).dropna(how="all")


__all__ = [
    "compute_per_ticker_macro_betas",
    "macro_levels_to_returns",
]
