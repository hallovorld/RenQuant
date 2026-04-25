"""Market-level buy gates shared by LEAN, notebook simulation, and live runner.

Pure functions — stdlib + numpy + pandas only.  No common/ imports.
"""
from __future__ import annotations

import math
from typing import Sequence

import pandas as pd


def check_spy_velocity_crash(
    spy_returns: Sequence[float],
    lookback_days: int = 3,
    halt_pct: float = 0.03,
) -> bool:
    """Return True (block buys) if SPY fell > halt_pct over last lookback_days.

    Uses cumulative product of daily returns to match LEAN's math.prod implementation.

    Audit fix MG-1 (Round 2 deep audit, 2026-04-25): pre-fix, NaN in
    spy_returns propagated through math.prod → cumret = NaN → `NaN <
    -halt_pct` is False → gate silently disabled on bad data. Post-fix:
    fail-SAFE on non-finite — return True so we BLOCK buys when SPY
    data is unreliable (better to miss an entry than buy through a
    blown-up data feed).
    """
    if halt_pct <= 0 or len(spy_returns) < lookback_days:
        return False
    recent = list(spy_returns)[-lookback_days:]
    if any((r is None) or (not math.isfinite(r)) for r in recent):
        import logging  # noqa: PLC0415
        logging.getLogger("kernel.market_gates").warning(
            "check_spy_velocity_crash: SPY returns contain non-finite "
            "values — gate FAIL-SAFE returning True (block buys)",
        )
        return True
    cumret = math.prod(1.0 + r for r in recent) - 1.0
    return cumret < -halt_pct


def check_spy_ema_trend(
    spy_close: pd.Series,
    ema_span: int = 50,
) -> bool:
    """Return True (block buys) if SPY's latest close is below its EMA.

    Args:
        spy_close: Series of SPY daily closing prices in chronological order.
        ema_span:  EMA period (default 50).

    Audit fix MG-2 (Round 2 deep audit, 2026-04-25): pre-fix, NaN
    close or NaN EMA evaluated `float(NaN) < float(NaN)` as False →
    macro EMA50 gate did NOT fire on bad data. Post-fix: explicit
    isfinite check, fail-SAFE return True on bad data.
    """
    if spy_close is None or len(spy_close) < ema_span + 1:
        return False
    ema = spy_close.ewm(span=ema_span, adjust=False).mean()
    last_close = float(spy_close.iloc[-1])
    last_ema   = float(ema.iloc[-1])
    if not (math.isfinite(last_close) and math.isfinite(last_ema)):
        import logging  # noqa: PLC0415
        logging.getLogger("kernel.market_gates").warning(
            "check_spy_ema_trend: SPY close/EMA non-finite — gate "
            "FAIL-SAFE returning True (block buys)",
        )
        return True
    return last_close < last_ema
