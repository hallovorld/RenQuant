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
    """
    if halt_pct <= 0 or len(spy_returns) < lookback_days:
        return False
    recent = list(spy_returns)[-lookback_days:]
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
    """
    if spy_close is None or len(spy_close) < ema_span + 1:
        return False
    ema = spy_close.ewm(span=ema_span, adjust=False).mean()
    return float(spy_close.iloc[-1]) < float(ema.iloc[-1])
