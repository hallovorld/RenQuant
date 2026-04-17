"""Position sizing — confidence-scaled with oversize fallback.

Self-contained: no common/ imports.
"""
from __future__ import annotations


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
