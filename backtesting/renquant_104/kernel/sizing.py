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
    # Audit fix SIZ-1 (Round 2 deep audit, 2026-04-25): pre-fix, NaN
    # panel_score slipped past `panel_score is None`, made `frac = NaN`
    # via `(NaN-floor)/span`, then `frac <= 0.0` and `frac >= 1.0` both
    # evaluated False on NaN → fell through to `min_mult + NaN*(...)`
    # = NaN. Conviction multiplier returned NaN, which NaN-poisoned
    # `max_pct = base * conv * sig_m` downstream in SizeAndEmitTask.
    # Now: explicit isfinite guard returns 1.0 (safe default — same
    # treatment as None).
    import math as _math
    if not sizing_cfg or not sizing_cfg.get("enabled", False):
        return 1.0
    if panel_score is None:
        return 1.0
    try:
        ps_f = float(panel_score)
    except (TypeError, ValueError):
        return 1.0
    if not _math.isfinite(ps_f):
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
    frac = (ps_f - floor) / span
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
    # Audit fix S-1 (Round 5, 2026-04-25): pre-fix, NaN price/portfolio
    # passed `<= 0` (NaN comparisons False) but then `int(NaN)` later in
    # the function raised ValueError, crashing the whole sizing path.
    # Post-fix: explicit isfinite + non-positive guard at the top.
    import math
    if (not math.isfinite(price) or not math.isfinite(portfolio_value)
            or price <= 0 or portfolio_value <= 0):
        return 0.0, 0
    if not math.isfinite(available_cash):
        return 0.0, 0
    # Audit fix CPS-1 (Round 2 deep audit, 2026-04-25): pre-fix, NaN
    # max_position_pct or cash_reserve_pct (e.g. caller computed
    # `max_pct * confidence` and confidence was NaN — pre-G-1 leak,
    # or from bad regime config) propagated through `target_pct =
    # min(NaN, ...)` into `int(NaN * pv / price)` which raises
    # ValueError "cannot convert float NaN to integer", crashing the
    # entire SizeAndEmitTask. Now: validate finite at entry — non-finite
    # → return (0, 0) clean fallback (skip this ticker; caller logs).
    if (not math.isfinite(max_position_pct)
            or not math.isfinite(cash_reserve_pct)):
        return 0.0, 0
    if override_pct is not None and not math.isfinite(override_pct):
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
        # Audit fix MIN-1-SHARE (Round 4 deep audit, 2026-04-25, user spec):
        # the size cap exists to LIMIT exposure, not to BLOCK trades. If
        # confidence-scaled cap and 25% fallback both produced 0 shares but
        # we have enough investable cash for at least one share, take the
        # one share. Pre-fix, low regime confidence (e.g. 0.0041) compounded
        # with high-priced stocks caused all buys to silently disappear,
        # which the user identified as the wrong behaviour ("给的额度不够买
        # 一股的时候就买一股嘛"). Override has a sane upper bound: only
        # fires when investable >= price (≥1 share affordable).
        if investable >= price:
            shares = 1

    if shares < 1:
        return 0.0, 0

    actual_pct = (shares * price) / portfolio_value
    return actual_pct, shares
