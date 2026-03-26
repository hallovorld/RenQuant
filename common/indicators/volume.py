"""Volume indicators: OBV."""

import numpy as np
import pandas as pd

from .registry import register


@register("obv", default_params={"signal_period": 20})
def compute_obv(df: pd.DataFrame, signal_period: int = 20) -> pd.DataFrame:
    """On-Balance Volume — cumulative volume confirming price direction.

    Rising OBV with rising price = strong trend.  Divergence between OBV
    and price often precedes a reversal.

    Output columns: ``obv``, ``obv_ema`` (signal line), ``obv_slope``
    (rate of change of EMA, positive = accumulation)
    """
    direction = np.sign(df["close"].diff())
    obv = (direction * df["volume"]).fillna(0).cumsum()
    obv_ema = obv.ewm(span=signal_period, adjust=False).mean()
    obv_slope = obv_ema.diff(5) / obv_ema.shift(5).replace(0, np.nan)

    return pd.DataFrame(
        {"obv": obv, "obv_ema": obv_ema, "obv_slope": obv_slope},
        index=df.index,
    )
