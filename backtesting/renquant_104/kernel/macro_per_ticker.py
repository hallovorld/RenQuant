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

# Single source of truth for rolling-β defaults so the pipeline caller
# in pp_panel_training.py and the function default can't drift.
DEFAULT_ROLLING_WINDOW = 60
DEFAULT_MIN_WINDOW = 30


def compute_per_ticker_macro_betas(
    ohlcv: dict[str, pd.DataFrame],
    macro_returns: pd.DataFrame,
    *,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    min_window: int = DEFAULT_MIN_WINDOW,
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
    skipped_no_close: list[str] = []
    skipped_short:    list[tuple[str, int]] = []

    for ticker, df in ohlcv.items():
        if df is None or df.empty or "close" not in df.columns:
            skipped_no_close.append(ticker)
            continue

        # Per-ticker daily returns (close-to-close)
        ticker_returns = df["close"].pct_change()

        if len(ticker_returns) < min_window:
            skipped_short.append((ticker, len(ticker_returns)))
            continue

        cols: dict[str, pd.Series] = {}

        for macro_col in macro_cols:
            macro_r = macro_returns[macro_col]
            # AUDIT 2026-05-10 C9 — defense in depth per §5.13.11. Upstream
            # fix is at macro.py:178 (dedup at source); this is a belt-and-
            # suspenders guard so any future regression doesn't reach
            # reindex with duplicate labels. APP ticker triggered this at
            # cutoff=2024-05-06; training_panel proceeded without APP for
            # that retrain.
            if macro_r.index.has_duplicates:
                macro_r = macro_r[~macro_r.index.duplicated(keep="last")]
            macro_r = macro_r.reindex(ticker_returns.index)

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

    # Audit M4 fix (2026-04-27): silent skips were hiding watchlist members
    # that would silently get the missing→0.0 fill in build_panel_frame —
    # making it look like a working β when really we never computed one.
    # Surface skips so operators can fix the underlying data gap.
    if skipped_no_close:
        log.warning(
            "compute_per_ticker_macro_betas: %d ticker(s) skipped (no close column / empty df): %s",
            len(skipped_no_close), ",".join(sorted(skipped_no_close)),
        )
    if skipped_short:
        log.warning(
            "compute_per_ticker_macro_betas: %d ticker(s) skipped (< min_window=%d bars): %s",
            len(skipped_short), min_window,
            ",".join(f"{t}({n})" for t, n in sorted(skipped_short)),
        )
    log.info(
        "compute_per_ticker_macro_betas: produced β for %d/%d tickers (window=%dd, min=%d)",
        len(out), len(ohlcv), rolling_window, min_window,
    )
    return out


def macro_levels_to_returns(macro_levels: pd.DataFrame) -> pd.DataFrame:
    """Convert macro factor LEVELS (z-scored prices) to RETURNS.

    The v1 `kernel.macro::build_macro_frame` produces z-scored levels
    (vxx_level_z, hyg_level_z, etc.). For β computation we need
    returns; this helper produces a 1-day return proxy.

    Convention: name columns `<symbol>_chg` (e.g. vxx_chg, hyg_chg).

    Bug-3 fix (2026-04-27): the original implementation applied diff()
    to z-scored levels.  diff(z) = (close_t - close_{t-1}) / σ, which
    is a *scaled price change* in z-score units — not a return.  The
    resulting β_macro carries units of "(ticker pct-return) per (σ of
    macro level)", which is economically uninterpretable and makes β
    values incomparable across macro symbols with different volatilities.

    Fix: use pct_change() on the z-scored level series as the closest
    available proxy for log-returns within this function's scope.  Note
    that the ideal fix is to pass raw close prices from the upstream
    MacroFactorStore directly; pct_change on z-scored prices is a
    second-best approximation that at least preserves proportionality
    with true returns (positive z-level → correct sign; near-zero z
    levels can produce outliers so the downstream cross-sectional z-score
    in FactorZScoreTask provides a safety net).
    """
    if macro_levels is None or macro_levels.empty:
        return pd.DataFrame()

    out: dict[str, pd.Series] = {}
    for col in macro_levels.columns:
        # Heuristic: pick only the *_level_z columns; chg_*d_z columns
        # are already differenced multi-day returns — skip them.
        if col.endswith("_level_z"):
            base = col.replace("_level_z", "")
            # Bug-3 fix: pct_change() gives a dimensionless return proxy;
            # diff() gave Δz in z-score units (no economic meaning for β).
            out[f"{base}_chg"] = macro_levels[col].pct_change()
        # else: skip — chg_5d / chg_20d are smoothed, not point returns

    return pd.DataFrame(out, index=macro_levels.index).dropna(how="all")


__all__ = [
    "compute_per_ticker_macro_betas",
    "macro_levels_to_returns",
]
