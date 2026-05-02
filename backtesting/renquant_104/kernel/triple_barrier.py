"""Triple-barrier labels — Lopez de Prado AFML §3.

Replaces fixed-horizon forward returns (e.g. `fwd_5d = close.shift(-5)/close - 1`)
with the realised return at the FIRST hit among three barriers:

  upper barrier (profit-take):   p_t × (1 + α × σ_t)
  lower barrier (stop-loss):     p_t × (1 − β × σ_t)
  time barrier (max horizon):    t + max_horizon_days

where σ_t is the trailing 20-day daily-return standard deviation. Once
any barrier is hit, the realised return at that hit is the label;
otherwise (timeout) the close-to-close return at the time barrier is used.

Why
---
fwd_N labels treat asymmetric within-window paths identically:
  - Stock A grinds +1% over 5 days
  - Stock B spikes +3% on day 1, drifts back to +1% by day 5
Both labels are +1%. But the model that holds A captures +1% while
the model that holds B should have sold day 1 (stop / news flow).
Triple-barrier captures this by labeling B with "+3% (upper hit at day 1)".

Reference
---------
Lopez de Prado, *Advances in Financial Machine Learning* (2018), ch. 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TripleBarrierConfig:
    """Hyperparameters for the labeller.

    Attributes
    ----------
    alpha : float
        Upper barrier multiplier. p_t × (1 + alpha × σ_t).
    beta : float
        Lower barrier multiplier. p_t × (1 − beta × σ_t).
    max_horizon_days : int
        Time barrier (in trading days). Hard cap on lookahead.
    vol_window : int
        Trailing window for daily-return σ_t. AFML uses 20.
    """
    alpha: float = 2.0
    beta: float = 2.0
    max_horizon_days: int = 10
    vol_window: int = 20


HitType = Literal["upper", "lower", "time"]


def _trailing_daily_vol(close: pd.Series, window: int) -> pd.Series:
    """Trailing rolling stdev of daily returns. NaN for the first
    `window` rows. Used to size barriers."""
    daily_ret = close.pct_change()
    return daily_ret.rolling(window=window, min_periods=window).std()


def compute_triple_barrier_labels(
    ohlcv: dict[str, pd.DataFrame],
    cfg: TripleBarrierConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute triple-barrier labels per ticker.

    Returns
    -------
    dict ticker → DataFrame with columns:
      label         float — realised close-to-hit return (signed)
      hit_type      str   — 'upper' | 'lower' | 'time'
      hit_days      int   — trading days from t to first-hit (1..max_horizon_days)
      sample_weight float — 1.0 for upper/lower, 0.5 for time barrier (timeout)

    Index matches each ticker's OHLCV index. Last `max_horizon_days` rows
    are NaN (insufficient lookahead).

    No-lookahead: barriers are sized using σ_t computed from data UP TO
    AND INCLUDING t (closing prices). The hit detection walks forward
    days t+1, t+2, ..., t+max_horizon_days exclusive — so day 0 (=t)
    is never a hit.
    """
    cfg = cfg or TripleBarrierConfig()
    out: dict[str, pd.DataFrame] = {}

    for ticker, df in ohlcv.items():
        if df is None or df.empty or "close" not in df.columns:
            out[ticker] = pd.DataFrame(
                index=df.index if df is not None else pd.DatetimeIndex([]),
                columns=["label", "hit_type", "hit_days", "sample_weight"],
            )
            continue

        close = df["close"].astype(float)
        idx = close.index
        sigma = _trailing_daily_vol(close, cfg.vol_window)

        n = len(close)
        labels = np.full(n, np.nan)
        hit_types = np.full(n, "", dtype=object)
        hit_days = np.full(n, np.nan)
        sample_weights = np.full(n, np.nan)

        # Walk forward per row. O(n × max_horizon_days) — for n=1000 days,
        # max_horizon=10 → 10K comparisons, well under 100ms per ticker.
        max_h = cfg.max_horizon_days

        for i in range(n - max_h):
            p_t = close.iloc[i]
            sigma_t = sigma.iloc[i]
            if not np.isfinite(p_t) or not np.isfinite(sigma_t) or sigma_t == 0:
                continue

            upper_barrier = p_t * (1.0 + cfg.alpha * sigma_t)
            lower_barrier = p_t * (1.0 - cfg.beta * sigma_t)

            hit_idx = None
            hit_kind: HitType = "time"
            for d in range(1, max_h + 1):
                p_future = close.iloc[i + d]
                if not np.isfinite(p_future):
                    continue
                if p_future >= upper_barrier:
                    hit_idx = d
                    hit_kind = "upper"
                    break
                if p_future <= lower_barrier:
                    hit_idx = d
                    hit_kind = "lower"
                    break

            if hit_idx is None:
                hit_idx = max_h
                hit_kind = "time"

            p_hit = close.iloc[i + hit_idx]
            labels[i] = (p_hit / p_t) - 1.0
            hit_types[i] = hit_kind
            hit_days[i] = hit_idx
            sample_weights[i] = 1.0 if hit_kind in ("upper", "lower") else 0.5

        out[ticker] = pd.DataFrame(
            {
                "label": labels,
                "hit_type": hit_types,
                "hit_days": hit_days,
                "sample_weight": sample_weights,
            },
            index=idx,
        )

    return out
