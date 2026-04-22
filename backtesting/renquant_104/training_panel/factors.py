"""Cross-sectional factor features for the Stage-1 LTR panel.

Four named factors:

- **size** — `log(close × shares_out)`. If `shares_out` is not provided
  we fall back to `log(close)` as a coarse proxy.
- **mom_12_1** — 252-day return minus the most recent 21 days
  (classical 12-1 momentum; skips last month to avoid microstructure reversal).
- **beta_60d** — rolling 60-day OLS slope of daily returns vs SPY.
- **resid_mom** — `mom_12_1 − β_60d × SPY's mom_12_1`.

Each factor is then cross-sectionally z-scored per date. The final output
is a dict of per-ticker factor frames with z-scored columns ready to be
concat'd into `build_panel_frame`'s `factor_frames` argument.

Public API::

    compute_momentum_12_1
    compute_rolling_beta
    compute_residual_momentum
    compute_size_feature
    cross_sectional_zscore
    build_factor_bundle
"""
from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd


def compute_momentum_12_1(
    ohlcv: dict[str, pd.DataFrame],
    mom_window: int = 252, skip: int = 21,
) -> dict[str, pd.Series]:
    """(close[t-skip] / close[t-mom_window]) - 1 per ticker."""
    out: dict[str, pd.Series] = {}
    for t, df in ohlcv.items():
        close = df["close"].astype(float)
        # 12-1: return from t-mom_window to t-skip
        ret_full = close.pct_change(mom_window)
        ret_skip = close.pct_change(skip)
        # (1 + ret_full) = (1 + ret_mom_12_1) * (1 + ret_skip)
        # ⇒ ret_mom_12_1 = (1 + ret_full) / (1 + ret_skip) - 1
        mom_12_1 = (1.0 + ret_full) / (1.0 + ret_skip) - 1.0
        out[t] = mom_12_1
    return out


def compute_rolling_beta(
    ohlcv: dict[str, pd.DataFrame], spy: pd.DataFrame,
    window: int = 60,
) -> dict[str, pd.Series]:
    """cov(r_i, r_spy) / var(r_spy) over a rolling `window`-bar window."""
    r_spy = spy["close"].astype(float).pct_change()
    out: dict[str, pd.Series] = {}
    for t, df in ohlcv.items():
        r_i = df["close"].astype(float).pct_change()
        idx = r_i.index.intersection(r_spy.index)
        r_i_a = r_i.reindex(idx)
        r_s_a = r_spy.reindex(idx)
        cov = r_i_a.rolling(window, min_periods=window).cov(r_s_a)
        var = r_s_a.rolling(window, min_periods=window).var()
        beta = cov / var.replace(0, np.nan)
        # Return aligned back to the ticker's original index
        out[t] = beta.reindex(r_i.index)
    return out


def compute_residual_momentum(
    ohlcv: dict[str, pd.DataFrame], spy: pd.DataFrame,
    window: int = 60, mom_window: int = 252, skip: int = 21,
) -> dict[str, pd.Series]:
    """mom_12_1_i − β_i × mom_12_1_spy."""
    mom = compute_momentum_12_1(ohlcv, mom_window=mom_window, skip=skip)
    beta = compute_rolling_beta(ohlcv, spy, window=window)
    mom_spy_full = compute_momentum_12_1({"SPY": spy}, mom_window=mom_window, skip=skip)["SPY"]
    out: dict[str, pd.Series] = {}
    for t in mom:
        m = mom[t]
        b = beta[t].reindex(m.index)
        s = mom_spy_full.reindex(m.index)
        out[t] = m - b * s
    return out


def compute_size_feature(
    ohlcv: dict[str, pd.DataFrame],
    shares_outstanding: dict[str, pd.Series] | None = None,
) -> dict[str, pd.Series]:
    """log(close × shares_out) per bar. Fallback: log(close)."""
    out: dict[str, pd.Series] = {}
    for t, df in ohlcv.items():
        close = df["close"].astype(float)
        if shares_outstanding and t in shares_outstanding:
            shr = shares_outstanding[t].reindex(close.index).ffill()
            mcap = close * shr
            out[t] = np.log(mcap.where(mcap > 0))
        else:
            out[t] = np.log(close.where(close > 0))
    return out


def cross_sectional_zscore(
    feature: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    """Per date: (value − mean) / std across tickers."""
    frames = []
    for t, s in feature.items():
        frames.append(pd.DataFrame({"date": s.index, "ticker": t, "val": s.values}))
    long = pd.concat(frames, ignore_index=True)

    grp = long.groupby("date", sort=False)["val"]
    long["mean"] = grp.transform("mean")
    long["std"]  = grp.transform("std")
    # Guard against zero / NaN std (only 1 ticker on a date)
    long["z"] = np.where(
        (long["std"] > 0) & long["std"].notna(),
        (long["val"] - long["mean"]) / long["std"],
        0.0,
    )
    # Rows whose original value was NaN should stay NaN
    long.loc[long["val"].isna(), "z"] = np.nan

    out: dict[str, pd.Series] = {}
    for t, sub in long.groupby("ticker", sort=False):
        s = pd.Series(sub["z"].values, index=pd.Index(sub["date"].values)).sort_index()
        out[t] = s
    return out


def build_factor_bundle(
    ohlcv: dict[str, pd.DataFrame],
    spy: pd.DataFrame,
    shares_outstanding: dict[str, pd.Series] | None = None,
    *,
    mom_window: int = 252,
    skip: int = 21,
    beta_window: int = 60,
) -> dict[str, pd.DataFrame]:
    """Return {ticker: DataFrame[size_z, mom_12_1_z, beta_60d_z, resid_mom_z]}."""
    size = compute_size_feature(ohlcv, shares_outstanding)
    mom  = compute_momentum_12_1(ohlcv, mom_window=mom_window, skip=skip)
    beta = compute_rolling_beta(ohlcv, spy, window=beta_window)
    rmom = compute_residual_momentum(
        ohlcv, spy, window=beta_window, mom_window=mom_window, skip=skip,
    )

    size_z = cross_sectional_zscore(size)
    mom_z  = cross_sectional_zscore(mom)
    beta_z = cross_sectional_zscore(beta)
    rmom_z = cross_sectional_zscore(rmom)

    out: dict[str, pd.DataFrame] = {}
    for t in ohlcv:
        idx = ohlcv[t].index
        df = pd.DataFrame({
            "size_z":      size_z.get(t, pd.Series(index=idx)).reindex(idx),
            "mom_12_1_z":  mom_z.get(t, pd.Series(index=idx)).reindex(idx),
            "beta_60d_z":  beta_z.get(t, pd.Series(index=idx)).reindex(idx),
            "resid_mom_z": rmom_z.get(t, pd.Series(index=idx)).reindex(idx),
        }, index=idx)
        out[t] = df
    return out
