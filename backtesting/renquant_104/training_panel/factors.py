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
    *, clip_low: float = -3.0, clip_high: float = 5.0,
) -> dict[str, pd.Series]:
    """cov(r_i, r_spy) / var(r_spy) over a rolling `window`-bar window.

    Audit fix D-1 (2026-04-25): β clipped to [clip_low, clip_high]
    (default [-3, +5]). Same rationale as labels._rolling_beta_purged —
    near-zero variance produces explosive β that dominates the
    downstream residual_momentum factor.
    """
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
        beta = beta.clip(lower=clip_low, upper=clip_high)
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


# ── Orthogonal factors (Round 3) ─────────────────────────────────────────────
#
# All four emit a per-ticker pd.Series aligned to the OHLCV index. Raw values
# go into `raw_factor_frame`; FactorZScoreTask cross-sectionally z-scores them
# per date before they enter the panel.

def compute_amihud_illiquidity(
    ohlcv: dict[str, pd.DataFrame], window: int = 21,
) -> dict[str, pd.Series]:
    """Amihud (2002) illiquidity: rolling mean of |return| / dollar_volume.

    Higher values ⇒ less liquid. Well-documented cross-sectional premium
    (illiquid names compensated with higher expected returns) and
    largely orthogonal to size once dollar-volume is in the denominator.
    """
    out: dict[str, pd.Series] = {}
    for t, df in ohlcv.items():
        close = df["close"].astype(float)
        vol   = df["volume"].astype(float) if "volume" in df.columns else None
        if vol is None:
            out[t] = pd.Series(np.nan, index=close.index)
            continue
        dollar_vol = (close * vol).replace(0, np.nan)
        abs_ret    = close.pct_change().abs()
        illiq      = (abs_ret / dollar_vol)
        # Take rolling average × 1e6 for numerical scale
        out[t] = illiq.rolling(window, min_periods=max(5, window // 2)).mean() * 1e6
    return out


def compute_volume_shift(
    ohlcv: dict[str, pd.DataFrame],
    short_window: int = 20,
    long_window:  int = 60,
) -> dict[str, pd.Series]:
    """log(avg_volume_short / avg_volume_long) — picks up trading-interest shifts.

    Rising accumulation / distribution signal. Complements size (level) and
    Amihud (cost); this is about the CHANGE in trading activity.
    """
    out: dict[str, pd.Series] = {}
    for t, df in ohlcv.items():
        vol = df["volume"].astype(float) if "volume" in df.columns else None
        if vol is None:
            out[t] = pd.Series(np.nan, index=df.index)
            continue
        short = vol.rolling(short_window, min_periods=max(5, short_window // 2)).mean()
        long  = vol.rolling(long_window,  min_periods=max(10, long_window // 2)).mean()
        ratio = (short / long.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out[t] = np.log(ratio.where(ratio > 0))
    return out


def compute_price_to_high(
    ohlcv: dict[str, pd.DataFrame], window: int = 252,
) -> dict[str, pd.Series]:
    """close / rolling_max(close, 252d) — price relative to 1-year high.

    Values near 1 indicate a stock at/near its recent peak; near 0 means
    deep underperformance. The 52-week-high anchor is a well-documented
    behavioral factor (George & Hwang 2004): stocks near their 52w high
    outperform those far below it, even after controlling for momentum.
    """
    out: dict[str, pd.Series] = {}
    for t, df in ohlcv.items():
        close = df["close"].astype(float)
        hi    = close.rolling(window, min_periods=max(20, window // 4)).max()
        out[t] = close / hi.replace(0, np.nan)
    return out


def compute_realized_vol(
    ohlcv: dict[str, pd.DataFrame], window: int = 20,
) -> dict[str, pd.Series]:
    """Annualized realized volatility — std of daily returns × √252.

    Complements beta: beta is exposure to market risk; vol is total risk
    including idiosyncratic. Low-vol anomaly supports negative loading.
    """
    out: dict[str, pd.Series] = {}
    for t, df in ohlcv.items():
        ret = df["close"].astype(float).pct_change()
        out[t] = ret.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(252)
    return out


def compute_drawdown_from_peak(
    ohlcv: dict[str, pd.DataFrame], window: int = 252,
) -> dict[str, pd.Series]:
    """Current drawdown from rolling 252-day peak = (close / peak) − 1.

    Always ≤ 0. Distinct from price-to-high by being path-dependent and
    behavioral: investor reluctance to realize losses creates predictable
    post-drawdown drift patterns.
    """
    out: dict[str, pd.Series] = {}
    for t, df in ohlcv.items():
        close = df["close"].astype(float)
        peak  = close.rolling(window, min_periods=max(20, window // 4)).max()
        out[t] = (close / peak.replace(0, np.nan)) - 1.0
    return out


def cross_sectional_zscore(
    feature: dict[str, pd.Series],
    winsorize_clip: float | None = 3.0,
) -> dict[str, pd.Series]:
    """Per date: (value − mean) / std across tickers.

    2026-04-24: winsorize by default at ±3σ (post z-score). Clipping
    outliers reduces tree-split influence from extreme values and is
    standard practice at quant shops — expected +0.002-0.005 OOS IC
    per doc/experiments/panel-ic-improvement.md. Pass winsorize_clip=None
    to disable (kept for A/B comparison + backward compat).
    """
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

    # Winsorize z-scores to ±clip (industry standard). A 3σ cap on a
    # standardized column keeps ~99.7% of a normal tail intact while
    # preventing a single blown-up outlier from dominating a tree split.
    if winsorize_clip is not None and winsorize_clip > 0:
        long["z"] = long["z"].clip(-winsorize_clip, winsorize_clip)

    out: dict[str, pd.Series] = {}
    for t, sub in long.groupby("ticker", sort=False):
        s = pd.Series(sub["z"].values, index=pd.Index(sub["date"].values)).sort_index()
        out[t] = s
    return out


def cross_sectional_rank_within_sector(
    feature: dict[str, pd.Series],
    ticker_sectors: dict[str, str],
    *,
    min_sector_size: int = 5,
    fallback_global: bool = True,
) -> dict[str, pd.Series]:
    """Per (date, sector): rank-normalize values across tickers in the
    same sector to a percentile in [0, 1].

    Why this exists
    ---------------
    The default ``cross_sectional_zscore`` standardizes each feature
    across the full universe per date. When the universe is heterogeneous
    in feature distribution (Witter 2025: e.g. tech mom_12_1 lives on a
    different scale than energy mom_12_1), the global z-score forces
    comparison between values whose underlying distributions are not
    comparable. The result was the wl178 fitting collapse documented in
    failed-experiments-log E17/E21 — train IC dropped from +0.118 to
    +0.085 because the rank-pairwise loss couldn't extract signal from
    cross-distribution comparisons.

    This helper produces a sector-relative percentile in [0, 1]. Two
    tickers in the same sector with the same percentile have the same
    rank within their cohort, regardless of how that cohort's raw values
    compare to another sector's. Reference: Microsoft qlib's
    ``CSRankNorm`` (see qlib/data/dataset/processor.py).

    Invariant
    ---------
    For any (date, sector) group with ≥ ``min_sector_size`` tickers:
        result_per_ticker ∈ [0, 1] AND values are unique-per-rank
        (ties broken by 'average' rank). Tickers in under-populated
        sectors fall back to global-percentile if ``fallback_global``,
        else NaN.

    Parameters
    ----------
    feature : dict[ticker, pd.Series]
        Time-indexed series of raw feature values, one per ticker.
    ticker_sectors : dict[ticker, sector]
        GICS-style sector labels (or any consistent partitioning). Used
        to group within each date.
    min_sector_size : int
        Below this threshold a sector is treated as under-populated and
        either falls back to the global cross-section or yields NaN. The
        default 5 is conservative — empirical-Bayes math (Robbins 1955;
        practitioner Wheeler 2018) shows percentile noise dominates below
        this scale anyway.
    fallback_global : bool
        When True, under-populated tickers get a global cross-sectional
        percentile (across the full universe on that date). When False,
        they get NaN — useful for debugging / strict experiments.

    Returns
    -------
    dict[ticker, pd.Series]
        Same shape as input. Values in [0, 1] for non-NaN entries. NaN
        rows propagated from the input remain NaN in the output.
    """
    if not feature:
        return {}

    frames = []
    for t, s in feature.items():
        frames.append(pd.DataFrame({
            "date":   s.index,
            "ticker": t,
            "sector": ticker_sectors.get(t, "_unmapped"),
            "val":    s.values,
        }))
    long = pd.concat(frames, ignore_index=True)

    # Per (date, sector) percentile rank among non-NaN values.
    # ``method='average'`` handles ties symmetrically — important when
    # many tickers share an exact value (e.g. zero overnight gap on a
    # quiet day). pct=True normalizes by group size to land in (0, 1].
    long["sector_pct"] = (
        long.groupby(["date", "sector"], sort=False, dropna=False)["val"]
        .rank(method="average", pct=True, na_option="keep")
    )

    # Per-(date, sector) sample size — used to enforce min_sector_size.
    long["sector_n"] = long.groupby(
        ["date", "sector"], sort=False, dropna=False,
    )["val"].transform(lambda s: int(s.notna().sum()))

    # Global per-date percentile rank — used as fallback for sectors
    # below the min size, or as primary for tickers with no sector label.
    long["global_pct"] = (
        long.groupby("date", sort=False)["val"]
        .rank(method="average", pct=True, na_option="keep")
    )

    if fallback_global:
        # Under-populated sector OR unmapped ticker → use global percentile.
        long["pct"] = np.where(
            (long["sector_n"] < min_sector_size) | (long["sector"] == "_unmapped"),
            long["global_pct"],
            long["sector_pct"],
        )
    else:
        long["pct"] = np.where(
            (long["sector_n"] < min_sector_size) | (long["sector"] == "_unmapped"),
            np.nan,
            long["sector_pct"],
        )

    # Preserve NaN-ness of input — a NaN raw value should not become a
    # spurious percentile.
    long.loc[long["val"].isna(), "pct"] = np.nan

    out: dict[str, pd.Series] = {}
    for t, sub in long.groupby("ticker", sort=False):
        s = pd.Series(
            sub["pct"].values,
            index=pd.Index(sub["date"].values),
        ).sort_index()
        out[t] = s
    return out


FUNDAMENTAL_COLS: list[str] = [
    "earnings_yield",
    "roe",
    "gross_profitability",
    "book_to_price",
    "short_pct_float",   # yfinance .info.shortPercentOfFloat — orthogonal sentiment factor
]

# Time-series factors derived from yfinance earnings_dates (updates step-wise
# at each earnings announcement, ffilled otherwise). Different from
# FUNDAMENTAL_COLS because these vary within a ticker over time.
TIMESERIES_EXTRA_COLS: list[str] = [
    "earnings_surprise_cum",   # trailing-4Q cumulative EPS surprise %
]


def _sector_median_fill(
    values: dict[str, float],
    sector_map: dict[str, str] | None,
) -> dict[str, float]:
    """Fill NaN entries with same-sector median across the remaining tickers.

    Used for fundamentals where point-in-time data is often missing for
    recent IPOs or sector-thin listings. Tickers without a sector map entry
    are filled with the global median of the remaining non-NaN values.
    """
    out = dict(values)
    non_nan = {k: v for k, v in values.items() if v == v}  # x != NaN
    if not non_nan:
        return out

    sector_non_nan: dict[str, list[float]] = {}
    if sector_map:
        for t, v in non_nan.items():
            sec = sector_map.get(t)
            if sec:
                sector_non_nan.setdefault(sec, []).append(v)

    sector_medians: dict[str, float] = {
        s: float(pd.Series(vs).median()) for s, vs in sector_non_nan.items()
    }
    global_median = float(pd.Series(list(non_nan.values())).median())

    for t, v in values.items():
        if v == v:   # already a finite number
            continue
        sec = (sector_map or {}).get(t)
        if sec and sec in sector_medians:
            out[t] = sector_medians[sec]
        else:
            out[t] = global_median
    return out


def _cross_sectional_zscore_static(
    values: dict[str, float],
) -> dict[str, float]:
    """z-score a {ticker → scalar} map across tickers; NaN input stays NaN."""
    xs = [v for v in values.values() if v == v]
    if len(xs) < 2:
        return {k: 0.0 if v == v else float("nan") for k, v in values.items()}
    s = pd.Series(xs)
    mean = float(s.mean())
    std  = float(s.std())
    if std <= 0 or std != std:
        return {k: 0.0 if v == v else float("nan") for k, v in values.items()}
    return {k: ((v - mean) / std) if v == v else float("nan")
            for k, v in values.items()}


def build_factor_bundle(
    ohlcv: dict[str, pd.DataFrame],
    spy: pd.DataFrame,
    shares_outstanding: dict[str, pd.Series] | None = None,
    *,
    mom_window: int = 252,
    skip: int = 21,
    beta_window: int = 60,
    fundamentals: dict[str, dict[str, float]] | None = None,
    sector_map: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Return {ticker: DataFrame[<factor_z columns>]}.

    Technical factors always present: size_z, mom_12_1_z, beta_60d_z, resid_mom_z.

    When `fundamentals` is passed as `{ticker: {earnings_yield, roe,
    gross_profitability, book_to_price}}`, four additional z-scored columns
    are appended: earnings_yield_z, roe_z, gross_profitability_z,
    book_to_price_z. Missing values are filled with the sector median
    (falling back to global median) before z-scoring, so a thin fundamentals
    cache doesn't torpedo the whole bundle.

    Fundamentals are a time-invariant snapshot in this release — the
    resulting columns are constant across each ticker's date index.
    """
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

    # Static fundamental z-scores (one scalar per ticker per column)
    fund_z: dict[str, dict[str, float]] = {}
    if fundamentals:
        for col in FUNDAMENTAL_COLS:
            raw = {t: float(fundamentals.get(t, {}).get(col, float("nan")))
                   for t in ohlcv}
            filled = _sector_median_fill(raw, sector_map)
            fund_z[col] = _cross_sectional_zscore_static(filled)

    out: dict[str, pd.DataFrame] = {}
    for t in ohlcv:
        idx = ohlcv[t].index
        cols = {
            "size_z":      size_z.get(t, pd.Series(index=idx)).reindex(idx),
            "mom_12_1_z":  mom_z.get(t, pd.Series(index=idx)).reindex(idx),
            "beta_60d_z":  beta_z.get(t, pd.Series(index=idx)).reindex(idx),
            "resid_mom_z": rmom_z.get(t, pd.Series(index=idx)).reindex(idx),
        }
        for col in FUNDAMENTAL_COLS:
            if col in fund_z:
                v = fund_z[col].get(t, float("nan"))
                cols[f"{col}_z"] = pd.Series(v, index=idx)
        out[t] = pd.DataFrame(cols, index=idx)
    return out
