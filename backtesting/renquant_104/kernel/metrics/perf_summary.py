"""Compute the (mean ± std, DSR, PBO) performance triple required by
CLAUDE.md §5.13.4 for any reported Sharpe / APY / IC.

Single-series mode: only DSR is computed (PBO requires ≥ 2 candidate
return series). Multi-seed mode: PBO is computed across the seeds, and
sharpe_mean / sharpe_std come from the seed Sharpes.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.stats import kurtosis, skew

from .deflated_sharpe import annualized_sharpe, deflated_sharpe_ratio
from .pbo import probability_of_backtest_overfitting


def compute_perf_triple(
    returns: np.ndarray,
    n_trials: int,
    multi_seed_returns: Optional[np.ndarray] = None,
    periods_per_year: int = 252,
    pbo_n_slices: int = 16,
) -> dict:
    """Return {sharpe, sharpe_mean, sharpe_std, dsr, pbo, n_returns, n_trials}.

    Parameters
    ----------
    returns : 1-D array of period returns (the "headline" series)
    n_trials : number of strategies / configurations searched to find
        this one. Feeds DSR's selection-bias correction.
    multi_seed_returns : optional (T × K) matrix of per-seed return
        series. K ≥ 2 enables PBO + sharpe_mean/std. If None, sharpe_mean
        falls back to the headline value and sharpe_std to NaN; PBO is NaN.
    periods_per_year : 252 (daily), 12 (monthly), etc.
    pbo_n_slices : S parameter for CSCV; must be even.

    Returns
    -------
    dict with the seven keys above. All Sharpes are annualized.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n_returns = int(arr.size)
    sharpe = annualized_sharpe(arr, periods_per_year)

    # Higher moments for DSR — scipy gives EXCESS kurtosis by default.
    if n_returns >= 3 and np.std(arr, ddof=1) > 0:
        sk = float(skew(arr, bias=False))
        ek = float(kurtosis(arr, fisher=True, bias=False))
    else:
        sk, ek = 0.0, 0.0

    # DSR uses the per-period Sharpe (not annualized) — undo the √P scale
    # so σ̂(SR) and E[max SR] are expressed on the same per-period scale.
    if math.isfinite(sharpe) and periods_per_year > 0:
        sr_per_period = sharpe / math.sqrt(periods_per_year)
    else:
        sr_per_period = float("nan")

    if math.isfinite(sr_per_period) and n_returns >= 2:
        dsr = deflated_sharpe_ratio(
            sr_observed=sr_per_period,
            n_returns=n_returns,
            n_trials=int(n_trials),
            skew=sk,
            excess_kurtosis=ek,
        )
    else:
        dsr = float("nan")

    # Multi-seed → PBO + per-seed Sharpe distribution.
    sharpe_mean, sharpe_std, pbo = sharpe, float("nan"), float("nan")
    if multi_seed_returns is not None:
        M = np.asarray(multi_seed_returns, dtype=float)
        if M.ndim != 2 or M.shape[1] < 2:
            raise ValueError("multi_seed_returns must be 2-D with K >= 2 columns")
        per_seed = np.array([
            annualized_sharpe(M[:, k], periods_per_year) for k in range(M.shape[1])
        ])
        finite = per_seed[np.isfinite(per_seed)]
        if finite.size > 0:
            sharpe_mean = float(np.mean(finite))
            sharpe_std = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
        try:
            pbo = probability_of_backtest_overfitting(M, n_slices=pbo_n_slices)
        except ValueError:
            pbo = float("nan")

    return {
        "sharpe": sharpe,
        "sharpe_mean": sharpe_mean,
        "sharpe_std": sharpe_std,
        "dsr": dsr,
        "pbo": pbo,
        "n_returns": n_returns,
        "n_trials": int(n_trials),
    }
