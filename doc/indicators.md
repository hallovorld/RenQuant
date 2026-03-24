# Indicator Library

All indicators use a uniform API registered via `@register`:

```python
from common.indicators import compute_indicators

df = compute_indicators(ohlcv, {
    "rsi": {"period": 14},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bbp": {"period": 20},
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

## Adding a New Indicator

1. Create the function in `momentum.py` or `volatility.py` (or a new file)
2. Decorate with `@register("name", default_params={...})`
3. The function receives a full OHLCV DataFrame, returns a DataFrame with named columns
4. Import the module in `indicators/__init__.py` to trigger registration

```python
@register("atr", default_params={"period": 14})
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return pd.DataFrame({"atr": tr.rolling(period).mean()}, index=df.index)
```
