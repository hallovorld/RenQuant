"""Technical indicator library with registry-based API.

Typical usage::

    from common.indicators import compute_indicators

    df = compute_indicators(ohlcv, {"rsi": {"period": 14}, "macd": {}})

Backward-compatible wrapper::

    from common.indicators import add_indicators
    df = add_indicators(ohlcv)  # adds MACD + RSI + CCI with default params
"""

# Import submodules to trigger @register decorators
from . import momentum as _momentum  # noqa: F401
from . import volatility as _volatility  # noqa: F401
from . import trend as _trend  # noqa: F401
from . import volume as _volume  # noqa: F401
from .registry import INDICATOR_REGISTRY, compute_indicators, list_indicators, register

# Re-export individual compute functions for direct use
from .momentum import compute_ema, compute_macd, compute_momentum, compute_rsi, compute_williams_r
from .volatility import compute_atr, compute_bbp, compute_cci, compute_ppo, compute_stochastic
from .trend import compute_adx
from .volume import compute_obv

import pandas as pd


def add_indicators(
    df: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rsi_period: int = 14,
    cci_period: int = 20,
) -> pd.DataFrame:
    """Backward-compatible wrapper: add MACD, RSI, and CCI columns.

    Equivalent to the old ``common.indicators.add_indicators``.
    """
    return compute_indicators(
        df,
        {
            "macd": {"fast": macd_fast, "slow": macd_slow, "signal": macd_signal},
            "rsi": {"period": rsi_period},
            "cci": {"period": cci_period},
        },
    )


__all__ = [
    # registry
    "INDICATOR_REGISTRY",
    "compute_indicators",
    "list_indicators",
    "register",
    # momentum
    "compute_rsi",
    "compute_macd",
    "compute_ema",
    "compute_momentum",
    "compute_williams_r",
    # volatility
    "compute_cci",
    "compute_bbp",
    "compute_stochastic",
    "compute_ppo",
    "compute_atr",
    # trend
    "compute_adx",
    # volume
    "compute_obv",
    # backward compat
    "add_indicators",
]
