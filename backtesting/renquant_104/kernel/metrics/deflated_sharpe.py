"""Deflated Sharpe Ratio (Bailey & López de Prado 2014, SSRN 2460551).

DSR adjusts an observed Sharpe for (a) selection bias from the number
of trials and (b) non-normality (skew / kurtosis).

Formulas (per the paper, Eqs. 5 + 9 + 12):
  σ̂(SR) = sqrt( (1 − γ_3·SR + (γ_4 − 1)/4 · SR²) / (n − 1) )
      γ_3 = skewness, γ_4 = raw (NOT excess) kurtosis.
  E[max SR | N iid N(0, V)] ≈ √V · ((1 − γ_em)·Z⁻¹(1 − 1/N)
                                     + γ_em·Z⁻¹(1 − 1/(N·e)))
      γ_em = 0.5772156649... (Euler-Mascheroni); Z⁻¹ = inverse-normal CDF.
  DSR = Φ( (SR_obs − E[max SR]) / σ̂(SR) )

Public input uses EXCESS kurtosis (κ_e = γ_4 − 3, scipy convention);
converted to raw internally.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def sharpe_std_error(
    sr_observed: float,
    n_returns: int,
    skew: float,
    excess_kurtosis: float,
) -> float:
    """σ̂(SR) per Eq. 5 of Bailey & López de Prado 2014.

    Inputs use EXCESS kurtosis (scipy default). Converts internally to
    raw kurtosis γ_4 = κ_e + 3 so the (γ_4 − 1)/4 = (κ_e + 2)/4 term
    matches the paper.
    """
    if n_returns < 2:
        raise ValueError("n_returns must be >= 2")
    raw_kurtosis = excess_kurtosis + 3.0
    variance = (
        1.0
        - skew * sr_observed
        + (raw_kurtosis - 1.0) / 4.0 * sr_observed ** 2
    ) / (n_returns - 1)
    if variance <= 0.0 or not math.isfinite(variance):
        # Pathological — fall back to the iid-normal SR variance (1/(n-1)).
        variance = 1.0 / (n_returns - 1)
    return math.sqrt(variance)


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """E[max_{i=1..N} SR_i] under H0:SR_i iid N(0, V).  Eq. 12.

    Uses the closed-form approximation from the paper (good for N ≥ 2).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if sharpe_variance < 0:
        raise ValueError("sharpe_variance must be >= 0")
    if n_trials == 1:
        return 0.0
    sqrt_v = math.sqrt(sharpe_variance)
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return sqrt_v * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def deflated_sharpe_ratio(
    sr_observed: float,
    n_returns: int,
    n_trials: int,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
    sharpe_variance: Optional[float] = None,
) -> float:
    """Bailey & López de Prado 2014 DSR — probability that the true SR > 0
    given the observed SR was the max of n_trials independent attempts.

    Returns a probability in [0, 1].  DSR < 0.95 typically means the
    finding does not survive selection-bias correction at 5%.

    sharpe_variance: variance of SR estimator across trials. Defaults to
        σ̂(SR)² of the observed series — a sensible self-contained estimate
        when no separate trial-level variance is available.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    se = sharpe_std_error(sr_observed, n_returns, skew, excess_kurtosis)
    if sharpe_variance is None:
        sharpe_variance = se ** 2
    e_max_sr = expected_max_sharpe(n_trials, sharpe_variance)
    if se <= 0:
        return float("nan")
    z = (sr_observed - e_max_sr) / se
    return float(norm.cdf(z))


def annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Helper: SR_ann = mean(r)/std(r) * sqrt(periods_per_year).

    Returns NaN if std == 0 or fewer than 2 observations.
    """
    arr = np.asarray(returns, dtype=float)
    if arr.size < 2:
        return float("nan")
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    # Threshold below float-roundoff noise (np.std on a constant array
    # can return ~1e-19 due to FP cancellation).
    if sd < 1e-12 or not math.isfinite(sd):
        return float("nan")
    return mu / sd * math.sqrt(periods_per_year)
