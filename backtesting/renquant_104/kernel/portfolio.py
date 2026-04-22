"""Portfolio-level helpers shared by LEAN, notebook simulation, and live runner.

Pure functions — stdlib only.  No common/ imports.
"""
from __future__ import annotations


def update_drawdown_circuit_breaker(
    portfolio_value: float,
    high_water_mark: float,
    halt_threshold: float,
) -> tuple[float, bool]:
    """Update HWM and determine whether the drawdown circuit breaker fires.

    Returns:
        (new_hwm, should_halt_buys)
    """
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
    """
    if gross_pnl <= 0:
        return 0.0
    rate = long_term_rate if hold_days >= long_term_threshold_days else short_term_rate
    return gross_pnl * rate
