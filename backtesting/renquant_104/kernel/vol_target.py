"""R-02 (2026-05-11) — portfolio-level volatility-targeting (gross-scaling).

Moskowitz, Ooi & Pedersen 2012 ("Time Series Momentum", JFE 104:228-250)
ran per-asset vol targeting at the asset level; at the portfolio level
the canonical form is:

    gross_scale_t  =  clip(  target_vol  /  realized_vol_t ,  floor,  ceiling )

so the portfolio targets a constant annualised volatility independent of
the market regime. AQR's "Risk Parity" desk and Asness-Frazzini "QMJ"
2014 follow the same template at month-end.

We proxy *portfolio* realized vol with **SPY** realized vol (β≈1 assumption)
— a cheap and operationally simple substitute that still cuts gross
exposure in vol-spike regimes (2020-Q1, 2022 bear, etc.). When a per-name
portfolio-return history becomes available, swap in the empirical
portfolio σ — same formula.

Inputs:
    spy_returns  iterable of recent daily SPY returns (latest last).
                 At least ``window_days`` rows must be present; fewer
                 returns 1.0 (no-scale, fail-open).
    window_days  rolling window, default 60 trading days (≈3 mo).
    target_vol   desired annualised vol, default 0.15 (= 15%).
    floor        clip lower bound, default 0.30 — never under-bet to <30%.
    ceiling      clip upper bound, default 1.50 — never over-lever to >150%.

Returns:
    Scale factor ``∈ [floor, ceiling]`` to multiply ``max_position_pct``
    (and any other gross-exposure knob) by.

Fail-open: NaN/inf/non-positive realized vol or empty input returns 1.0.
Both DrawdownCircuitTask and the QP solver provide downstream safety
nets; this layer is a SOFT lever.
"""
from __future__ import annotations

import math


def compute_vol_target_scale(
    spy_returns:  "list | tuple",
    *,
    target_vol:   float = 0.15,
    window_days:  int   = 60,
    floor:        float = 0.30,
    ceiling:      float = 1.50,
) -> float:
    """Return the gross-exposure scale ``∈ [floor, ceiling]``.

    Fail-open contract: any malformed input (too few returns, non-finite
    target, non-finite realized vol, non-positive σ) returns 1.0.
    """
    if not (math.isfinite(target_vol) and target_vol > 0):
        return 1.0
    if window_days <= 1:
        return 1.0
    rets = list(spy_returns) if spy_returns else []
    if len(rets) < window_days:
        return 1.0
    window = [float(r) for r in rets[-window_days:]
              if r is not None and math.isfinite(float(r))]
    if len(window) < 2:
        return 1.0
    mean = sum(window) / len(window)
    var  = sum((r - mean) ** 2 for r in window) / (len(window) - 1)
    if not math.isfinite(var) or var <= 0:
        return 1.0
    realized_vol = math.sqrt(var) * math.sqrt(252.0)
    if not math.isfinite(realized_vol) or realized_vol <= 0:
        return 1.0
    raw_scale = target_vol / realized_vol
    if not math.isfinite(raw_scale):
        return 1.0
    return max(float(floor), min(float(ceiling), raw_scale))


__all__ = ["compute_vol_target_scale"]
