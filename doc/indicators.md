# Indicator Library

## Two implementations

| Module | Used by | Imports |
|--------|---------|---------|
| `common/indicators/` | renquant_101, renquant_102, notebooks, live runner | pandas, numpy, scikit-learn |
| `backtesting/renquant_103/kernel/indicators.py` | LEAN Docker, renquant_103 notebook + live runner | numpy, pandas only (no common/) |

Both implement the same indicators with identical semantics. The kernel version is LEAN-safe (zero `common/` imports). Always use the kernel version inside `backtesting/renquant_103/`.

## Usage (common/)

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

## Regime Detection (non-registered, used directly)

These functions live in `common/indicators/regime.py` and are **not** registered in the indicator registry (they operate on SPY returns, not per-stock OHLCV). Import directly:

```python
from common.indicators import compute_hurst, rolling_hurst, compute_cusum, rolling_cusum, build_gmm_features, RegimeGMM
```

### Hurst Exponent
- **Function**: `compute_hurst(returns, max_lag=40)` / `rolling_hurst(returns, window=63)`
- **Input**: daily return series
- **Output**: H in [0, 1]. > 0.55 = momentum (trending), < 0.45 = mean-reversion (choppy), 0.45–0.55 = random walk
- **Method**: Rescaled Range (R/S) analysis

### CUSUM Changepoint Detection
- **Function**: `compute_cusum(returns, threshold=3.0, drift=0.5)` / `rolling_cusum(returns, window=20)`
- **Input**: daily return series
- **Output**: bool — True if a structural break is detected in the window
- **Method**: Cumulative Sum control chart, normalised internally so threshold is in σ units

### GMM Regime Classifier
- **Class**: `RegimeGMM(n_components=3)`
- **Methods**: `fit(features_df)`, `predict(features_df) → (label_series, proba_df)`, `save(path)`, `RegimeGMM.load(path)`
- **Input features** (built via `build_gmm_features(spy_ohlcv)`): `10d_return`, `20d_realized_vol`, `spy_adx`, `return_autocorr`
- **Output**: per-day regime label (`BULL_CALM`, `BULL_VOLATILE`, `BEAR`) + probability DataFrame
- **Serialisation**: `save()` writes JSON compatible with LEAN's inline GMM inference (no sklearn dependency at runtime)

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
