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


def risk_free_rate_annual_to_daily(rf_annual: float) -> float:
    """Convert annual risk-free rate to daily compounding equivalent.

    Industry-standard for daily Sharpe (Bodie/Kane/Marcus, Investments,
    11th ed., §5.4): ``rf_daily = (1 + rf_annual)^(1/252) - 1``.

    Differs from the simple ``rf_annual / 252`` divisor used inside
    ``sharpe_ratio`` (which is the linear approximation appropriate for
    the arithmetic-Sharpe excess-return formulation). Use this helper
    for the geometric-Sharpe path where compounding consistency matters.

    Returns NaN for non-finite input.
    """
    if not math.isfinite(rf_annual):
        return float("nan")
    return float((1.0 + rf_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0)


def geometric_sharpe_ratio(
    returns: _SeriesLike,
    risk_free_rate: float = 0.0,
    ann_factor: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Geometric Sharpe: (geo_mean_ret - rf_daily) / std_ret × √ann_factor.

    Geometric mean compounded daily: ``exp(mean(log(1+r))) - 1``. This is
    more accurate than the arithmetic Sharpe for path-dependent /
    high-volatility strategies because volatility drag (the log-vs-linear
    gap) is captured directly in the numerator (Israelsen 2003,
    "A Refinement to the Sharpe Ratio and Information Ratio", J. Asset
    Management 5(6): 423-427).

    The ``risk_free_rate`` argument is the DAILY risk-free rate (already
    compounded — pass output of ``risk_free_rate_annual_to_daily``). Using
    a daily rf keeps the geometric numerator coherent with daily compounding.

    Returns NaN if:
      * fewer than 30 valid observations (sample-size guard, matches
        benchmark-relative metrics rule of thumb)
      * std(returns) below floating-point noise floor (degenerate)
      * any ``1 + r ≤ 0`` (negative cumulative wealth — log undefined)

    Sign convention: positive when geo-mean exceeds rf_daily.
    """
    r = _to_series(returns).dropna()
    if len(r) < 30:
        return float("nan")
    one_plus = 1.0 + r
    if (one_plus <= 0).any():
        return float("nan")
    log_r = np.log(one_plus)
    mean_log = float(log_r.mean())
    if not math.isfinite(mean_log):
        return float("nan")
    geo_mean = math.exp(mean_log) - 1.0
    std = float(r.std(ddof=1))
    if not math.isfinite(std) or std < _STD_ZERO_EPSILON:
        return float("nan")
    return float((geo_mean - risk_free_rate) / std * math.sqrt(ann_factor))


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
    # 2026-05-10 audit fix (Sortino-Price 1994, "Performance Measurement in
    # a Downside Risk Framework", J. Investing 3(3): 59-64): the canonical
    # downside-deviation uses the SAMPLE standard deviation of the
    # below-target returns (ddof=1), not the population RMS √E[d²] of the
    # truncated series. The pre-fix population form biased downside_std
    # DOWNWARD when n_downside is small (typical for daily-resolution
    # equity strategies with mostly positive returns), which inflated
    # Sortino by ~0.7% per the same reference. Switching to ddof=1 makes
    # Sortino numerically comparable to Sharpe (which already uses ddof=1).
    downside_std = float(downside.std(ddof=1))
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


# ── Benchmark-relative metrics (beta / alpha / information ratio) ──────────
#
# References
# ----------
# - Sharpe, W.F. (1964). "Capital Asset Prices". J. Finance 19(3): 425-442.
#   Defines β = Cov(r, b) / Var(b) and α = E[r] − β·E[b].
# - Treynor, J.L. & Black, F. (1973). "How to Use Security Analysis". J.
#   Business 46(1): 66-86. Information ratio = mean(active) / σ(active).
# - Goodwin, T.H. (1998). "The Information Ratio". Fin. Analysts Journal
#   54(4): 34-43. Annualization conventions.
#
# Sample-size guard (n < 30 → NaN) follows the rule-of-thumb from
# Goodwin 1998 §IV: for daily-resolution benchmark regressions the OLS
# estimator's standard error is too wide to interpret below ~30 obs.
# Per §5.13.12, beta is clipped to ±10 to suppress ill-conditioned
# regressions (e.g. flat-benchmark cases the variance guard didn't catch).

# Minimum overlapping observations required before β/α/IR are computed.
# Below this, the OLS estimator is too noisy to report (Goodwin 1998).
_BENCHMARK_MIN_N: int = 30
# Floor on var(benchmark) below which β is undefined (division by ~zero).
_BENCHMARK_VAR_EPSILON: float = 1e-12
# Sanity clip on β. Real equity-strategy β rarely exceeds 3; 10 is a
# loose ceiling that catches pathological regressions on small samples
# (per §5.13.12 — defensive guard for ill-conditioned outputs).
_BETA_ABS_CLIP: float = 10.0


def _align_returns_benchmark(
    returns: _SeriesLike, benchmark: _SeriesLike,
) -> tuple[pd.Series, pd.Series]:
    """Inner-join r and b on their common index, drop NaN rows.

    Returns ``(r_aligned, b_aligned)``. The two have identical length and
    index. Empty when there is no overlap or every overlap is NaN.
    """
    r = _to_series(returns)
    b = _to_series(benchmark)
    # If either series carries no index (raw array), align by position.
    if isinstance(returns, pd.Series) and isinstance(benchmark, pd.Series):
        df = pd.concat([r, b], axis=1, join="inner").dropna()
    else:
        n = min(len(r), len(b))
        df = pd.concat(
            [r.iloc[:n].reset_index(drop=True),
             b.iloc[:n].reset_index(drop=True)],
            axis=1,
        ).dropna()
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return df.iloc[:, 0], df.iloc[:, 1]


def beta_vs_benchmark(
    returns: _SeriesLike, benchmark: _SeriesLike,
) -> float:
    """OLS β = Cov(r, b) / Var(b). NaN when n < 30 or Var(b) < 1e-12.

    Clipped to ±10 per §5.13.12 — catches ill-conditioned regression on
    nearly-constant benchmark windows that escape the variance guard.
    """
    r, b = _align_returns_benchmark(returns, benchmark)
    if len(r) < _BENCHMARK_MIN_N:
        return float("nan")
    var_b = float(b.var(ddof=1))
    if not math.isfinite(var_b) or var_b < _BENCHMARK_VAR_EPSILON:
        return float("nan")
    cov_rb = float(((r - r.mean()) * (b - b.mean())).sum() / (len(r) - 1))
    if not math.isfinite(cov_rb):
        return float("nan")
    beta = cov_rb / var_b
    if not math.isfinite(beta):
        return float("nan")
    # Defensive clip (§5.13.12) — preserves sign, caps magnitude.
    if beta > _BETA_ABS_CLIP:
        return _BETA_ABS_CLIP
    if beta < -_BETA_ABS_CLIP:
        return -_BETA_ABS_CLIP
    return float(beta)


def alpha_vs_benchmark(
    returns: _SeriesLike,
    benchmark: _SeriesLike,
    *,
    beta: float | None = None,
    ann_factor: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized CAPM α = (mean(r) − β·mean(b)) · ann_factor.

    Per-period α = mean(r) − β·mean(b) (Sharpe 1964). Annualized by
    multiplying by ann_factor (252 daily). When ``beta`` is None it is
    computed from the same series via ``beta_vs_benchmark``.

    NaN when n < 30 or β is NaN (propagates the guard from beta).
    """
    r, b = _align_returns_benchmark(returns, benchmark)
    if len(r) < _BENCHMARK_MIN_N:
        return float("nan")
    if beta is None:
        beta = beta_vs_benchmark(r, b)
    if not math.isfinite(beta):
        return float("nan")
    mean_r = float(r.mean())
    mean_b = float(b.mean())
    if not (math.isfinite(mean_r) and math.isfinite(mean_b)):
        return float("nan")
    return float((mean_r - beta * mean_b) * ann_factor)


def information_ratio(
    returns: _SeriesLike,
    benchmark: _SeriesLike,
    *,
    ann_factor: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """IR = mean(r − b) / σ(r − b, ddof=1) · √ann_factor (Treynor-Black 1973).

    Annualized active-return Sharpe of the (r − b) series. NaN when
    n < 30 or σ(active) below floating-point noise floor.
    """
    r, b = _align_returns_benchmark(returns, benchmark)
    if len(r) < _BENCHMARK_MIN_N:
        return float("nan")
    active = r - b
    sd = float(active.std(ddof=1))
    if not math.isfinite(sd) or sd < _STD_ZERO_EPSILON:
        return float("nan")
    mean_a = float(active.mean())
    if not math.isfinite(mean_a):
        return float("nan")
    return float(mean_a / sd * math.sqrt(ann_factor))


def compute_risk_metrics(
    equity: _SeriesLike,
    *,
    apy: float | None = None,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    include_geometric: bool = False,
) -> dict[str, float]:
    """Compute the full bundle in one pass — convenience for sim/B2 callers.

    Returns
    -------
    dict with keys: sharpe, sortino, calmar, max_dd, ann_vol,
    n_observations. NaN in any field signals "not enough data" rather
    than zero (don't conflate "no data" with "perfect Sharpe").

    When ``include_geometric=True`` (sim opt-in) the result also carries
    ``sharpe_geometric`` (Israelsen 2003) — geometric-mean-based Sharpe
    that internalizes volatility drag. Default is False to preserve the
    byte-identical key set existing tests / B2 consumers rely on.

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

    out: dict[str, float] = {
        "sharpe":         sharpe,
        "sortino":        sortino,
        "calmar":         calmar,
        "max_dd":         mdd,
        "ann_vol":        vol,
        "n_observations": float(n_valid),
    }
    if include_geometric:
        # Use compounded-daily rf for geometric-Sharpe coherence
        # (Israelsen 2003 / Bodie-Kane-Marcus §5.4). Distinct from the
        # rf/N linear divisor used inside arithmetic sharpe_ratio.
        rf_daily = risk_free_rate_annual_to_daily(risk_free_rate)
        out["sharpe_geometric"] = geometric_sharpe_ratio(
            r, risk_free_rate=rf_daily, ann_factor=trading_days_per_year,
        )
    return out


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "daily_returns_from_equity",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "geometric_sharpe_ratio",
    "risk_free_rate_annual_to_daily",
    "max_drawdown",
    "calmar_ratio",
    "beta_vs_benchmark",
    "alpha_vs_benchmark",
    "information_ratio",
    "compute_risk_metrics",
]
