import numpy as np
import pandas as pd


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Compute MACD line, signal line, and histogram."""
    macd_fast = close.ewm(span=fast, adjust=False).mean()
    macd_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = macd_fast - macd_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_line - macd_signal,
    })


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using EMA smoothing."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return pd.Series(100 - (100 / (1 + rs)), name="rsi")


def compute_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Compute Commodity Channel Index."""
    typical_price = (high + low + close) / 3
    mean = typical_price.rolling(period).mean()
    mad = typical_price.rolling(period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    return pd.Series((typical_price - mean) / (0.015 * mad), name="cci")


def add_indicators(
    df: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rsi_period: int = 14,
    cci_period: int = 20,
) -> pd.DataFrame:
    """Add MACD, RSI, and CCI columns to an OHLCV DataFrame."""
    df = df.copy()
    df = df.join(compute_macd(df["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal))
    df["rsi"] = compute_rsi(df["close"], period=rsi_period)
    df["cci"] = compute_cci(df["high"], df["low"], df["close"], period=cci_period)
    return df
