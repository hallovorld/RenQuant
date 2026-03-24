"""Volatility indicators: CCI, BBP, Stochastic, PPO."""

import numpy as np
import pandas as pd

from .registry import register


@register("cci", default_params={"period": 20})
def compute_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Commodity Channel Index.

    Output columns: ``cci``
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    sma = typical_price.rolling(period).mean()
    mad = typical_price.rolling(period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    cci = (typical_price - sma) / (0.015 * mad.replace(0, np.nan))
    return pd.DataFrame({"cci": cci}, index=df.index)


@register("bbp", default_params={"period": 20})
def compute_bbp(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Bollinger Band Percentage.

    BBP = (close - SMA) / (2 * std).  Ranges roughly from -1 to +1.

    Output columns: ``bbp``
    """
    close = df["close"]
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    bbp = (close - sma) / (2 * std.replace(0, np.nan))
    return pd.DataFrame({"bbp": bbp}, index=df.index)


@register("stochastic", default_params={"window": 14, "smooth": 3})
def compute_stochastic(
    df: pd.DataFrame, window: int = 14, smooth: int = 3
) -> pd.DataFrame:
    """Stochastic Oscillator (%K, %D, and spread).

    Output columns: ``stoch_k``, ``stoch_d``, ``stoch_spread``
    """
    rolling_low = df["low"].rolling(window=window).min()
    rolling_high = df["high"].rolling(window=window).max()
    price_range = (rolling_high - rolling_low).replace(0, np.nan)
    pct_k = 100 * (df["close"] - rolling_low) / price_range
    pct_d = pct_k.rolling(window=smooth).mean()
    return pd.DataFrame(
        {
            "stoch_k": pct_k,
            "stoch_d": pct_d,
            "stoch_spread": pct_k - pct_d,
        },
        index=df.index,
    )


@register("ppo", default_params={"fast": 12, "slow": 26, "signal": 9})
def compute_ppo(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Percentage Price Oscillator (PPO line, signal, histogram).

    Output columns: ``ppo_line``, ``ppo_signal``, ``ppo_hist``
    """
    close = df["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    ppo_line = 100 * (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)
    ppo_signal = ppo_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {
            "ppo_line": ppo_line,
            "ppo_signal": ppo_signal,
            "ppo_hist": ppo_line - ppo_signal,
        },
        index=df.index,
    )
