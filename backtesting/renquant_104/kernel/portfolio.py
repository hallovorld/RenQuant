"""Portfolio-level helpers shared by LEAN, notebook simulation, and live runner.

Pure functions — stdlib only.  No common/ imports.
"""
from __future__ import annotations

import math


def update_drawdown_circuit_breaker(
    portfolio_value: float,
    high_water_mark: float,
    halt_threshold: float,
) -> tuple[float, bool]:
    """Update HWM and determine whether the drawdown circuit breaker fires.

    Returns:
        (new_hwm, should_halt_buys)

    Audit fix PORT-1/PORT-2 (Round 2 deep audit, 2026-04-25): pre-fix,
    `max(NaN_hwm, finite_pv)` returned NaN (CPython max-NaN semantics),
    then the `<=0` guard let NaN slip past, then drawdown = NaN/NaN
    = NaN, then `NaN >= threshold` False → returned (NaN, False)
    silently. Caller persisted NaN HWM, propagating corruption across
    bars. Fail-SAFE: non-finite inputs route to a clean fallback —
    HWM stays at the LAST finite value (or 0), no halt fires.
    """
    if not math.isfinite(portfolio_value):
        # Bad portfolio_value (broker outage, NaN equity) — preserve
        # stored HWM if finite; don't ratchet up; don't halt.
        if math.isfinite(high_water_mark):
            return float(high_water_mark), False
        return 0.0, False
    if not math.isfinite(high_water_mark):
        # Stored HWM corrupted but pv is good — reset HWM to pv so
        # future drawdown calc is meaningful.
        return float(portfolio_value), False
    new_hwm = max(high_water_mark, portfolio_value)
    if halt_threshold <= 0 or new_hwm <= 0:
        return new_hwm, False
    drawdown = (new_hwm - portfolio_value) / new_hwm
    return new_hwm, drawdown >= halt_threshold


def compute_trade_tax(
    gross_pnl: float,
    hold_days: int,
    short_term_rate: float,
    long_term_rate: float,
    long_term_threshold_days: int = 365,
) -> float:
    """Return income tax owed on a realized trade.

    Only positive P&L is taxed.  Long-term rate applies when
    hold_days >= long_term_threshold_days.

    Audit fix PORT-3 (Round 2 deep audit, 2026-04-25): pre-fix, NaN
    gross_pnl slipped past `<= 0` (NaN<=0 False), then `gross_pnl *
    rate = NaN` propagated into tax_drag and rotation cost calcs.
    Now: explicit isfinite guard returns 0 on non-finite (no tax owed
    when we can't compute the gain — fail-safe).
    """
    if not math.isfinite(gross_pnl) or gross_pnl <= 0:
        return 0.0
    rate = long_term_rate if hold_days >= long_term_threshold_days else short_term_rate
    return gross_pnl * rate
