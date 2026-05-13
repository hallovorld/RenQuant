"""Stationary block bootstrap — thin wrapper over `arch` package.

We use Kevin Sheppard's `arch` package (the canonical implementation
used in academic econometrics) rather than rolling our own. arch is
maintained by the author of *Financial Econometrics Using MATLAB*
(2013) and is what UVA Darden, Oxford-Man, and AQR cite.

References:
  Politis & Romano 1994 *J. Amer. Stat. Assoc.* 89:1303 (stationary bootstrap)
  Politis-White 2004 *Econometric Reviews* 23:53 (optimal block length)
  Politis-Romano-Wolf 2008 *J. Empirical Finance* 15:319 (bootstrap for Sharpe)

arch docs:
  https://arch.readthedocs.io/en/latest/bootstrap/bootstrap.html
"""
from __future__ import annotations

import math
import numpy as np
from typing import Callable, Sequence


def optimal_block_length(series) -> int:
    """Politis-White 2004 optimal block length via arch.

    arch.bootstrap.optimal_block_length returns (sb_lag, cb_lag) — we
    use the stationary-bootstrap variant (`sb`).
    """
    from arch.bootstrap import optimal_block_length as _arch_obl  # noqa: PLC0415
    arr = np.asarray(list(series), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 8:
        return max(2, len(arr) // 2)
    try:
        df = _arch_obl(arr)
        # df is a pandas DataFrame; row '0' has columns 'stationary' and 'circular'
        return max(2, int(round(float(df["stationary"].iloc[0]))))
    except Exception:
        return max(2, int(round(len(arr) ** (1.0 / 3.0))))


def stationary_bootstrap_ci(
    series: Sequence[float],
    stat_fn: Callable[[np.ndarray], float] = lambda x: float(np.mean(x)),
    block_length: int | None = None,
    B: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Stationary-bootstrap CI on any statistic of a 1-D time-series.

    Wraps arch.bootstrap.StationaryBootstrap.
    """
    from arch.bootstrap import StationaryBootstrap  # noqa: PLC0415
    s = np.asarray(list(series), dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 4:
        return {"point": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "se_boot": float("nan"),
                "block_length": 0, "B": 0}
    if block_length is None:
        block_length = optimal_block_length(s)
    seed = int(rng.integers(0, 2**31 - 1)) if rng is not None else 42

    def _stat(x):
        # arch passes numpy arrays one-per-call
        return float(stat_fn(np.asarray(x)))

    bs = StationaryBootstrap(int(block_length), s, seed=seed)
    # bs.apply runs `stat_fn` over each resample; returns ndarray of length reps
    boot = bs.apply(_stat, reps=int(B))
    boot = np.asarray(boot).reshape(-1)
    point = float(_stat(s))
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1.0 - alpha / 2))
    se = float(boot.std(ddof=1)) if len(boot) > 1 else float("nan")
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "se_boot": se,
            "block_length": int(block_length), "B": int(B)}


def sharpe_ratio_ci(
    returns: Sequence[float],
    block_length: int | None = None,
    B: int = 2000,
    alpha: float = 0.05,
    rf: float = 0.0,
    periods_per_year: int = 252,
    rng: np.random.Generator | None = None,
) -> dict:
    """Annualized Sharpe ratio CI via stationary block bootstrap.

    Per Politis-Romano-Wolf 2008, naive parametric CI on Sharpe
    underestimates true variance under autocorrelated returns;
    block bootstrap is the standard correction.
    """
    def _sr(r: np.ndarray) -> float:
        m = float(np.asarray(r).mean()) - rf / periods_per_year
        sd = float(np.asarray(r).std(ddof=1))
        if sd <= 0:
            return 0.0
        return (m / sd) * math.sqrt(periods_per_year)
    return stationary_bootstrap_ci(returns, _sr, block_length, B, alpha, rng)


__all__ = ["optimal_block_length", "stationary_bootstrap_ci", "sharpe_ratio_ci"]
