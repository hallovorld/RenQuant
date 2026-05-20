# Indicator Library

## Three implementations

| Module | Used by | Imports |
|--------|---------|---------|
| `common/indicators/` | renquant_101, renquant_102, notebooks (legacy) | pandas, numpy, scikit-learn |
| `backtesting/renquant_103/kernel/indicators.py` | renquant_103 LEAN + live + sim | numpy, pandas only (no common/) |
| `backtesting/renquant_104/kernel/indicators.py` | **renquant_104 (active) LEAN + live + sim + panel-LTR training** | numpy, pandas only (no common/) |

All three implement the same indicators with identical semantics. The kernel versions are LEAN-safe (zero `common/` imports). **Always use the kernel version of the strategy you're working in** — `common/indicators/` is for the legacy 101/102 path only.

## Panel-LTR feature catalog (172 features, 2026-05-20)

The active production artifact (`artifacts/prod/panel-ltr.alpha158_fund.json`) consumes 172 features, organized as:

- **158 alpha158 features** — Qlib-faithful (`Alpha158` per Qlib reference): KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2, OPEN0, HIGH0, LOW0, VWAP0, ROC{5-60}, MA{5-60}, STD{5-60}, BETA{5-60}, RSQR{5-60}, RESI{5-60}, MAX{5-60}, MIN{5-60}, QTLU{5-60}, QTLD{5-60}, RANK{5-60}, RSV{5-60}, IMAX{5-60}, IMIN{5-60}, IMXD{5-60}, CORR{5-60}, CORD{5-60}, CNTP{5-60}, CNTN{5-60}, CNTD{5-60}, SUMP{5-60}, SUMN{5-60}, SUMD{5-60}, VMA{5-60}, VSTD{5-60}, WVMA{5-60}, VSUMP{5-60}, VSUMN{5-60}, VSUMD{5-60}.
- **5 SEC fundamental features** — `earnings_yield`, `book_to_price`, `gross_profitability`, `roe`, `asset_growth` (Cooper-Gulen-Schill 2008 `pct_change(periods=252d)` post-Bug-5 fix).
- **3 PEAD features** — `days_since_earnings`, `pead_signal`, `pead_quintile_rank` (post-earnings drift).
- **3 SUE features** — `sue_signal`, `surprise_momentum`, `surprise_streak`.
- **3 news sentiment features** — `sentiment_pos_share`, `mean_sentiment`, `n_articles` (2026-05-18 shipped, regime-conditional gate live across 14 regimes).

## Momentum features added 2026-05-18 (Jegadeesh-Titman / 52w distance / sector momentum)

Per `doc/research/2026-05-18-model-regime-mismatch.md` finding ("model is mean-reversion in a momentum market"), 3 new momentum features were wired into the panel via pandas_ta_classic:
- Jegadeesh-Titman 12-1 month momentum
- 52-week distance from high
- Sector-cross-sectional momentum rank

These are part of the 172 feature count above where applicable; see commits `fc32385` / `a355a69`.

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
