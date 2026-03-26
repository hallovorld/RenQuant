"""Trend indicators: ADX."""

import numpy as np
import pandas as pd

from .registry import register


@register("adx", default_params={"period": 14})
def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index — trend strength (0-100).

    >25 = trending, <20 = ranging.  Direction-neutral: high ADX means a
    strong trend in *either* direction.

    Output columns: ``adx``, ``plus_di``, ``minus_di``
    """
    high, low, close = df["high"], df["low"], df["close"]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat(
        [high_low, high_close, low_close], axis=1
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return pd.DataFrame(
        {"adx": adx, "plus_di": plus_di, "minus_di": minus_di},
        index=df.index,
    )
