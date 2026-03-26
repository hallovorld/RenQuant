"""Momentum indicators: RSI, MACD, EMA, Momentum, Williams %R."""

import numpy as np
import pandas as pd

from .registry import register


@register("rsi", default_params={"period": 14})
def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index (EMA smoothing).

    Output columns: ``rsi``
    """
    close = df["close"]
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return pd.DataFrame({"rsi": 100 - (100 / (1 + rs))}, index=df.index)


@register("macd", default_params={"fast": 12, "slow": 26, "signal": 9})
def compute_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line, and histogram.

    Output columns: ``macd_line``, ``macd_signal``, ``macd_hist``
    """
    close = df["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_line - macd_signal,
        },
        index=df.index,
    )


@register("ema", default_params={"period": 20})
def compute_ema(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Exponential Moving Average.

    Output columns: ``ema``
    """
    return pd.DataFrame(
        {"ema": df["close"].ewm(span=period, adjust=False).mean()},
        index=df.index,
    )


@register("momentum", default_params={"period": 10})
def compute_momentum(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """Price momentum (rate of change).

    Output columns: ``momentum``
    """
    close = df["close"]
    return pd.DataFrame(
        {"momentum": close / close.shift(period) - 1},
        index=df.index,
    )


@register("williams_r", default_params={"period": 14})
def compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Williams %R — momentum oscillator, inverse of Fast Stochastic.

    Range: -100 (oversold) to 0 (overbought).

    Output columns: ``williams_r``
    """
    high = df["high"].rolling(window=period).max()
    low = df["low"].rolling(window=period).min()
    price_range = (high - low).replace(0, np.nan)
    wr = -100 * (high - df["close"]) / price_range
    return pd.DataFrame({"williams_r": wr}, index=df.index)
