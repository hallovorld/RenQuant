"""Newey-West HAC SE — thin wrapper over statsmodels.

We use statsmodels' battle-tested implementation rather than rolling our
own. statsmodels.stats.sandwich_covariance.S_hac_simple (and the OLS
HAC-cov path) is what AQR / Two Sigma / academic finance researchers
use; reproducing this in custom code is forbidden under CLAUDE.md
§5.12 ("Default to canonical references; never reinvent").

Reference: Newey & West 1987 *Econometrica* 55(3):703.
Lag selection: Andrews 1991 *Econometrica* 59:817.

statsmodels docs:
  https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def andrews_optimal_lag(n: int) -> int:
    """Andrews 1991 plug-in bandwidth for Newey-West (Bartlett kernel).

    L = floor(4 × (n/100)^(2/9))
    """
    if n <= 1:
        return 0
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_t_stat(returns: Sequence[float], lag: int | None = None,
               null_mean: float = 0.0) -> dict:
    """Newey-West HAC t-stat for the mean of a return series.

    Uses statsmodels' OLS with HAC covariance — the canonical approach.
    Regression: r_t = α + ε_t, test H0: α = null_mean.

    Returns dict with: mean, se_nw, t_stat, lag, n, p_value.
    """
    import statsmodels.api as sm  # noqa: PLC0415
    r = np.asarray(list(returns), dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2:
        return {"mean": float("nan"), "se_nw": float("nan"),
                "t_stat": float("nan"), "lag": 0, "n": n, "p_value": float("nan")}
    lag_used = andrews_optimal_lag(n) if lag is None else int(lag)
    # OLS of r on constant with HAC covariance
    X = np.ones((n, 1))
    model = sm.OLS(r, X)
    # cov_type='HAC' uses Newey-West with Bartlett kernel
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": max(0, lag_used)})
    mean = float(res.params[0])
    se = float(res.bse[0])
    t = (mean - null_mean) / se if se > 0 else float("nan")
    # statsmodels gives p-value for H0: param=0, recompute for our null_mean
    from scipy.stats import norm  # noqa: PLC0415
    p = 2.0 * float(1.0 - norm.cdf(abs(t))) if math.isfinite(t) else float("nan")
    return {"mean": mean, "se_nw": se, "t_stat": t,
            "lag": lag_used, "n": n, "p_value": p}


def newey_west_se(returns: Sequence[float], lag: int | None = None) -> float:
    """Thin wrapper returning just the SE (backward-compat shim)."""
    return hac_t_stat(returns, lag=lag).get("se_nw", float("nan"))


__all__ = ["andrews_optimal_lag", "newey_west_se", "hac_t_stat"]
