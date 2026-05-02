"""Risk-adjusted performance metrics — Sharpe / Sortino / Calmar / Max DD / Vol.

Pure functions, no side effects. Take a portfolio equity series (price /
NAV indexed by date) and return scalar metrics annualized to 252 trading
days (US equity convention).

Why this module exists
----------------------
The golden config goal is APY 41% / **Sharpe 2.0** (CLAUDE.md), but Sharpe
hasn't been measured in any sim output until now — meaning every "+X% APY
improvement" claim has been blind to risk-adjusted change. Adding these
metrics is the prerequisite for any A/B that's supposed to confirm
risk-adjusted improvement.

References
----------
- Sharpe, W.F. (1994). "The Sharpe Ratio". J. Portfolio Management 21(1): 49-58.
  Defines the modern excess-return-over-volatility form. We use rf=0
  (no risk-free rate adjustment) per the convention in single-strategy
  backtests; user can post-hoc subtract a benchmark return if needed.
- Sortino, F.A. & Price, L.N. (1994). "Performance Measurement in a
  Downside Risk Framework". J. Investing 3(3): 59-64. Original Sortino
  paper using downside-only deviation (not full std).
- Young, T.W. (1991). "Calmar Ratio: A Smoother Tool". Futures 20(1).
  Original definition: APY / Max DD over a fixed window.
- Magdon-Ismail-Atiya 2004 — analytic max drawdown for GBM as a
  complement; we compute empirical here.
"""
from __future__ import annotations

import math
from typing import Sequence, Union

import numpy as np
import pandas as pd

# Standard US equity convention: 252 trading days per year. Used for
# annualization of daily-resolution metrics.
TRADING_DAYS_PER_YEAR: int = 252

# Floating-point tolerance below which std is treated as zero. Constant
# return series produce fp64 std ~3e-18 due to summation noise; treating
# anything below this as "effectively zero" returns clean 0.0 / NaN
# rather than enormous spurious ratios (1e16-scale Sharpe nonsense).
_STD_ZERO_EPSILON: float = 1e-12


_SeriesLike = Union[pd.Series, np.ndarray, Sequence[float]]


def _to_series(x: _SeriesLike) -> pd.Series:
    """Coerce array-likes to pd.Series. Raises on non-numeric / empty."""
    if isinstance(x, pd.Series):
        s = x.astype(float)
    else:
        s = pd.Series(np.asarray(x, dtype=float))
    return s


def daily_returns_from_equity(equity: _SeriesLike) -> pd.Series:
    """Compute daily simple returns from an equity (NAV) series.

    First row is NaN (no prior to diff against). NaN propagation matches
    pandas pct_change default.

    Invariant
    ---------
    For ``equity`` of length N, the result has length N with the first
    entry NaN. ``returns[i] = equity[i] / equity[i-1] - 1`` for i ≥ 1.
    """
    s = _to_series(equity)
    if len(s) < 2:
        return pd.Series([np.nan] * len(s), index=s.index)
    return s.pct_change()


def annualized_volatility(
    returns: _SeriesLike,
    *,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """σ(returns) × √252.

    Returns NaN if there are fewer than 2 valid (non-NaN) observations.
    """
    r = _to_series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    std = float(r.std(ddof=1))
    # Treat fp-noise std (constant series) as exactly zero so callers
    # see a clean 0.0 instead of e.g. 5.5e-17.
    if std < _STD_ZERO_EPSILON:
        return 0.0
    return float(std * math.sqrt(trading_days_per_year))


def sharpe_ratio(
    returns: _SeriesLike,
    *,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio.

    Sharpe = ( mean(returns) − rf/N ) / std(returns) × √N
           = annualized excess return / annualized volatility

    risk_free_rate is the ANNUAL rate (e.g. 0.05 for 5%). It is divided by
    N (trading days) before subtracting. Default 0 — appropriate for
    most single-strategy comparisons; subtract benchmark return ex-post
    if benchmark-relative Sharpe is needed.

    Returns NaN if there are fewer than 2 valid observations OR if std=0
    (degenerate constant return series).
    """
    r = _to_series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    daily_rf = risk_free_rate / trading_days_per_year
    excess = r - daily_rf
    std = float(excess.std(ddof=1))
    # Constant series → fp std on the order of 1e-17. Treat below
    # _STD_ZERO_EPSILON as zero — Sharpe is undefined for zero std,
    # so return NaN rather than a meaningless 1e16-scale number.
    if std < _STD_ZERO_EPSILON or not math.isfinite(std):
        return float("nan")
    return float(excess.mean() / std * math.sqrt(trading_days_per_year))


def sortino_ratio(
    returns: _SeriesLike,
    *,
    target_return: float = 0.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sortino ratio — Sharpe but with downside-only deviation.

    Sortino = ( mean(returns) − target/N ) / downside_std × √N
    where downside_std uses only returns below target (annualized rate).

    target_return = 0 (default) means the deviation considered is just
    "negative returns." Other common choices: rf rate, MAR (minimum
    acceptable return).

    Returns NaN when there are no downside observations (constant
    above-target return) — the metric is undefined in that case.
    """
    r = _to_series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    daily_target = target_return / trading_days_per_year
    excess = r - daily_target
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("nan")
    downside_std = math.sqrt((downside ** 2).mean())
    if downside_std < _STD_ZERO_EPSILON or not math.isfinite(downside_std):
        return float("nan")
    return float(excess.mean() / downside_std * math.sqrt(trading_days_per_year))


def max_drawdown(equity: _SeriesLike) -> float:
    """Maximum peak-to-trough decline expressed as a positive fraction.

    For an equity curve with peak P and subsequent trough T:
        max_dd = (P − T) / P  (in [0, 1])

    Returns 0.0 for monotone-increasing curves; NaN for empty / single-point
    inputs.

    Invariant
    ---------
    Always non-negative. ``max_drawdown(constant) == 0``. Sign convention:
    we report DD as a positive fraction, NOT a negative percent —
    consumers (Calmar) divide APY by this directly.
    """
    s = _to_series(equity).dropna()
    if len(s) < 2:
        return float("nan")
    running_max = s.cummax()
    drawdowns = (running_max - s) / running_max
    return float(drawdowns.max())


def calmar_ratio(apy: float, max_dd: float) -> float:
    """APY / Max DD. Higher = better risk-adjusted return.

    Conventionally computed over a 36-month window in PM literature
    (Young 1991), but here we accept any apy + dd from caller. Returns
    NaN when max_dd == 0 (monotone gain) or NaN/inf — the metric is
    undefined in those cases.
    """
    if not math.isfinite(apy) or not math.isfinite(max_dd):
        return float("nan")
    if max_dd <= 0:
        return float("nan")
    return float(apy / max_dd)


def compute_risk_metrics(
    equity: _SeriesLike,
    *,
    apy: float | None = None,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Compute the full bundle in one pass — convenience for sim/B2 callers.

    Returns
    -------
    dict with keys: sharpe, sortino, calmar, max_dd, ann_vol,
    n_observations. NaN in any field signals "not enough data" rather
    than zero (don't conflate "no data" with "perfect Sharpe").

    The caller MUST pass `apy` (annualized return) when they want
    Calmar; otherwise we can't compute Calmar from equity alone in a
    single pass without re-deriving it (which can disagree with the
    caller's own APY definition — better to take their number).
    """
    r = daily_returns_from_equity(equity)
    n_valid = int(r.notna().sum())
    sharpe = sharpe_ratio(
        r, risk_free_rate=risk_free_rate,
        trading_days_per_year=trading_days_per_year,
    )
    sortino = sortino_ratio(
        r, target_return=target_return,
        trading_days_per_year=trading_days_per_year,
    )
    mdd = max_drawdown(equity)
    vol = annualized_volatility(
        r, trading_days_per_year=trading_days_per_year,
    )
    if apy is None:
        # Approximate APY from equity if the caller didn't pass one.
        # Used as fallback for Calmar; less accurate than the caller's
        # own APY definition (which may use exact day count, etc.).
        s = _to_series(equity).dropna()
        if len(s) >= 2:
            n_years = (len(s) - 1) / trading_days_per_year
            total_ret = float(s.iloc[-1] / s.iloc[0] - 1.0) if s.iloc[0] != 0 else 0.0
            apy = (1.0 + total_ret) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
        else:
            apy = float("nan")
    calmar = calmar_ratio(apy, mdd)

    return {
        "sharpe":         sharpe,
        "sortino":        sortino,
        "calmar":         calmar,
        "max_dd":         mdd,
        "ann_vol":        vol,
        "n_observations": float(n_valid),
    }


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "daily_returns_from_equity",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "compute_risk_metrics",
]
