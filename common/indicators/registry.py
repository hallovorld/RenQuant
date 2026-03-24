"""Indicator registry with uniform API.

Every indicator is a function with the signature::

    (df: pd.DataFrame, **params) -> pd.DataFrame

The returned DataFrame contains one or more named columns (e.g.
``macd_line``, ``macd_signal``, ``macd_hist`` for MACD).

Use the ``@register`` decorator to add indicators to the global
registry, then call ``compute_indicators`` to compute any subset
by name.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

# name -> (function, default_params)
INDICATOR_REGISTRY: dict[str, tuple[Callable[..., pd.DataFrame], dict[str, Any]]] = {}


def register(name: str, default_params: dict[str, Any] | None = None):
    """Decorator that registers an indicator function.

    Example::

        @register("rsi", default_params={"period": 14})
        def compute_rsi(df, period=14):
            ...
    """

    def decorator(func: Callable[..., pd.DataFrame]):
        INDICATOR_REGISTRY[name] = (func, default_params or {})
        return func

    return decorator


def compute_indicators(
    df: pd.DataFrame,
    indicator_spec: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Compute multiple indicators and join them onto an OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with columns ``open, high, low, close, volume``.
        indicator_spec: Mapping of ``indicator_name -> param_overrides``.
            Example::

                {"rsi": {"period": 14}, "macd": {"fast": 12, "slow": 26, "signal": 9}}

    Returns:
        Copy of *df* with indicator columns joined.
    """
    result = df.copy()
    for name, params in indicator_spec.items():
        if name not in INDICATOR_REGISTRY:
            available = ", ".join(sorted(INDICATOR_REGISTRY))
            raise KeyError(
                f"Unknown indicator {name!r}. Registered: {available}"
            )
        func, defaults = INDICATOR_REGISTRY[name]
        merged = {**defaults, **params}
        indicator_df = func(result, **merged)
        result = result.join(indicator_df)
    return result


def list_indicators() -> dict[str, dict[str, Any]]:
    """Return ``{name: default_params}`` for every registered indicator."""
    return {name: dict(defaults) for name, (_, defaults) in INDICATOR_REGISTRY.items()}
