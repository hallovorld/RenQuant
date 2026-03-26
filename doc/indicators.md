# Indicator Library

All indicators use a uniform API registered via `@register`:

```python
from common.indicators import compute_indicators

df = compute_indicators(ohlcv, {
    "rsi": {"period": 14},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "adx": {"period": 14},
    "obv": {"signal_period": 20},
})
```

## Momentum Indicators

### RSI — Relative Strength Index
- **Module**: `common/indicators/momentum.py`
- **Output columns**: `rsi`
- **Parameters**: `period` (default: 14)
- **Range**: 0–100. <30 oversold, >70 overbought.

### MACD — Moving Average Convergence Divergence
- **Module**: `common/indicators/momentum.py`
- **Output columns**: `macd_line`, `macd_signal`, `macd_hist`
- **Parameters**: `fast` (12), `slow` (26), `signal` (9)

### EMA — Exponential Moving Average
- **Module**: `common/indicators/momentum.py`
- **Output columns**: `ema`
- **Parameters**: `period` (20)

### Momentum — Rate of Change
- **Module**: `common/indicators/momentum.py`
- **Output columns**: `momentum`
- **Parameters**: `period` (10)

### Williams %R
- **Module**: `common/indicators/momentum.py`
- **Output columns**: `williams_r`
- **Parameters**: `period` (14)
- **Range**: -100 (oversold) to 0 (overbought). Inverse of Fast Stochastic.
- **Uses**: high, low, close.

## Volatility Indicators

### CCI — Commodity Channel Index
- **Module**: `common/indicators/volatility.py`
- **Output columns**: `cci`
- **Parameters**: `period` (20)
- **Uses**: high, low, close. +/-100 are overbought/oversold.

### BBP — Bollinger Band Percentage
- **Module**: `common/indicators/volatility.py`
- **Output columns**: `bbp`
- **Parameters**: `period` (20)
- **Range**: roughly -1 to +1.

### Stochastic Oscillator
- **Module**: `common/indicators/volatility.py`
- **Output columns**: `stoch_k`, `stoch_d`, `stoch_spread`
- **Parameters**: `window` (14), `smooth` (3)
- **Uses**: high, low, close.

### PPO — Percentage Price Oscillator
- **Module**: `common/indicators/volatility.py`
- **Output columns**: `ppo_line`, `ppo_signal`, `ppo_hist`
- **Parameters**: `fast` (12), `slow` (26), `signal` (9)

### ATR — Average True Range
- **Module**: `common/indicators/volatility.py`
- **Output columns**: `atr`, `atr_pct` (ATR as % of close)
- **Parameters**: `period` (14)
- **Uses**: high, low, close. Measures volatility magnitude.

## Trend Indicators

### ADX — Average Directional Index
- **Module**: `common/indicators/trend.py`
- **Output columns**: `adx`, `plus_di`, `minus_di`
- **Parameters**: `period` (14)
- **Range**: 0–100. >25 = trending, <20 = ranging. Direction-neutral.
- **Uses**: high, low, close.

## Volume Indicators

### OBV — On-Balance Volume
- **Module**: `common/indicators/volume.py`
- **Output columns**: `obv`, `obv_ema` (signal line), `obv_slope` (rate of change)
- **Parameters**: `signal_period` (20)
- **Uses**: close, volume. Rising OBV = accumulation; divergence from price precedes reversals.

## Adding a New Indicator

1. Create the function in an existing or new file under `common/indicators/`
2. Decorate with `@register("name", default_params={...})`
3. The function receives a full OHLCV DataFrame, returns a DataFrame with named columns
4. Import the module in `indicators/__init__.py` to trigger registration

```python
@register("my_indicator", default_params={"period": 14})
def compute_my_indicator(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    result = df["close"].rolling(period).mean()  # example
    return pd.DataFrame({"my_indicator": result}, index=df.index)
```
