"""Position sizing — confidence-scaled with oversize fallback.

Self-contained: no common/ imports.
"""
from __future__ import annotations


def sigma_multiplier(
    sigma: float | None,
    sigma_median: float | None,
    sigma_cfg: dict | None,
) -> float:
    """Scale factor ∈ [floor, ceiling] based on predictive σ.

    High-σ candidates get smaller sizes: `mult = clip(σ_median / σ, floor, ceiling)`.
    A candidate at the universe median gets multiplier 1.0.

    Returns 1.0 when σ-sizing is disabled, σ is missing, or the median is
    not a positive finite number (i.e. no change from existing behaviour).

    sigma_cfg keys (all optional):
      enabled : bool, default False
      floor   : minimum multiplier, default 0.3
      ceiling : maximum multiplier, default 1.0  (don't oversize low-σ candidates)
    """
    if not sigma_cfg or not sigma_cfg.get("enabled", False):
        return 1.0
    if sigma is None or sigma_median is None:
        return 1.0
    try:
        s = float(sigma)
        med = float(sigma_median)
    except (TypeError, ValueError):
        return 1.0
    if not (s > 0.0 and med > 0.0):
        return 1.0
    try:
        floor = float(sigma_cfg.get("floor", 0.3))
        ceil  = float(sigma_cfg.get("ceiling", 1.0))
    except (TypeError, ValueError):
        return 1.0
    if ceil < floor:
        return 1.0
    m = med / s
    return max(floor, min(ceil, m))


def universe_sigma_median(sigmas: list[float | None]) -> float | None:
    """Median over non-None, positive, finite σ values. None if empty."""
    import math
    vals = [float(s) for s in sigmas
            if s is not None and math.isfinite(float(s)) and float(s) > 0.0]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    if n % 2 == 1:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def conviction_multiplier(panel_score: float | None, sizing_cfg: dict | None) -> float:
    """Scale factor in [min_mult, 1.0] derived from a candidate's panel score.

    Rescales (panel_score - floor) / (ceiling - floor) into [min_mult, 1.0].
    Returns 1.0 when sizing is disabled, the score is missing, or the config
    is malformed — i.e. no change from existing behaviour.

    sizing_cfg keys (all optional):
      enabled  : bool, default False
      floor    : panel_score at/below which we use min_mult
      ceiling  : panel_score at/above which we use 1.0
      min_mult : minimum multiplier, default 0.5
    """
    if not sizing_cfg or not sizing_cfg.get("enabled", False):
        return 1.0
    if panel_score is None:
        return 1.0
    try:
        floor    = float(sizing_cfg.get("floor", 0.0))
        ceiling  = float(sizing_cfg.get("ceiling", 1.0))
        min_mult = float(sizing_cfg.get("min_mult", 0.5))
    except (TypeError, ValueError):
        return 1.0
    if ceiling <= floor:
        return 1.0
    span = ceiling - floor
    frac = (float(panel_score) - floor) / span
    if frac <= 0.0:
        return min_mult
    if frac >= 1.0:
        return 1.0
    return min_mult + frac * (1.0 - min_mult)


def compute_position_size(
    portfolio_value: float,
    available_cash: float,
    max_position_pct: float,   # from regime params (already confidence-scaled by caller)
    cash_reserve_pct: float,   # from regime params (already confidence-scaled by caller)
    price: float,
    override_pct: float | None = None,
) -> tuple[float, int]:
    """Return (target_pct, shares) for a buy order.

    override_pct: bypass reserve calc (BEAR defensive branch).

    Returns (0.0, 0) if there is insufficient cash for at least 1 share.
    Falls back to 25% cap if confidence-scaled pct can't cover 1 share
    (prevents high-priced stocks like LLY from being silently skipped).
    """
    if price <= 0 or portfolio_value <= 0:
        return 0.0, 0

    if override_pct is not None:
        investable = available_cash
        max_pct    = override_pct
    else:
        cash_reserve = portfolio_value * cash_reserve_pct
        investable   = max(available_cash - cash_reserve, 0.0)
        max_pct      = max_position_pct

    target_pct = min(max_pct, investable / portfolio_value)

    # Compute shares
    target_dollars = target_pct * portfolio_value
    shares = int(target_dollars / price)

    if shares < 1:
        # Oversize fallback: try 25% of portfolio
        fallback_dollars = 0.25 * portfolio_value
        shares = int(min(fallback_dollars, investable) / price)

    if shares < 1:
        return 0.0, 0

    actual_pct = (shares * price) / portfolio_value
    return actual_pct, shares
