# Plan: Relative (Stock/SPY) Indicators — IMPLEMENTED

## Motivation

Raw indicators reflect absolute momentum, volatility, and trend of a single stock. In a strong bull market, every stock looks like a buy; in a crash, everything looks like a sell. By normalizing indicators against SPY, the model learns **whether the stock is outperforming or underperforming the market** — not just whether it's going up or down.

This is widely used in institutional quant strategies (relative strength, pairs trading, market-neutral signals). The risk: in a sector rotation or stock-specific catalyst, the relative signal may lag or invert because SPY dilutes the signal.

---

## Core Idea

For each feature, replace the raw value with the stock-to-SPY ratio:

```
relative_indicator = stock_indicator / spy_indicator
```

For indicators that can cross zero or be negative (MACD hist, BBP, CCI), use the **difference** instead:

```
relative_indicator = stock_indicator - spy_indicator
```

---

## Feature Transformation Rules

| Feature | Raw Range | Transform | Rationale |
|---------|-----------|-----------|-----------|
| `price` | >0 | `close_stock / close_spy` | Relative price strength; rising = outperforming |
| `rsi` | 0–100 | `rsi_stock / rsi_spy` | Both always positive; ratio >1 = stock more overbought |
| `macd_hist` | any | `macd_hist_stock - macd_hist_spy` | Crosses zero; difference preserves sign meaning |
| `cci` | any | `cci_stock - cci_spy` | Crosses zero; difference better than ratio |
| `bbp` | ~-1 to +1 | `bbp_stock - bbp_spy` | Crosses zero; difference preserves interpretation |
| `adx` | 0–100 | `adx_stock / adx_spy` | Both always positive; ratio >1 = stock trending harder |
| `williams_r` | -100 to 0 | `williams_r_stock - williams_r_spy` | Both negative; difference simpler than ratio of negatives |
| `obv_slope` | any | `obv_slope_stock - obv_slope_spy` | Crosses zero; difference = relative volume momentum |

---

## Changes Required

### 1. Data Layer — fetch SPY alongside the stock

**File**: `Notebooks/test_001_nvda.ipynb` (Cell 1)

```python
df_stock = common.fetch_ohlcv(SYMBOL, start=START, end=END, provider=PROVIDER)
df_spy   = common.fetch_ohlcv("SPY",  start=START, end=END, provider=PROVIDER)

df_stock = common.compute_indicators(df_stock, INDICATOR_SPEC)
df_spy   = common.compute_indicators(df_spy,   INDICATOR_SPEC)
```

No changes to `common/data/` needed — `fetch_ohlcv` already supports any symbol.

### 2. New helper — compute relative features

**File**: `Notebooks/test_001_nvda.ipynb` (new Cell 1b, or inline in Cell 1)

Build a function that takes stock and SPY DataFrames and produces relative features:

```python
RATIO_FEATURES  = ["rsi", "adx"]               # always positive → use ratio
DIFF_FEATURES   = ["macd_hist", "cci", "bbp",   # cross zero → use difference
                    "williams_r", "obv_slope"]

def compute_relative_features(df_stock, df_spy, feature_columns):
    """Produce relative features: stock/SPY ratio or difference."""
    df = pd.DataFrame(index=df_stock.index)
    df["rel_price"] = df_stock["close"] / df_spy["close"]
    for col in feature_columns:
        if col in RATIO_FEATURES:
            df[col] = df_stock[col] / df_spy[col].replace(0, np.nan)
        else:
            df[col] = df_stock[col] - df_spy[col]
    return df.dropna()
```

The resulting DataFrame has the same column names as before (`rsi`, `macd_hist`, etc.) so downstream model code requires **zero changes** — only the values are different.

### 3. Notebook cell updates

| Cell | Current | After |
|------|---------|-------|
| Cell 0 (config) | `FEATURE_COLUMNS = [...]` | Add `REL_PRICE = True` flag; keep same FEATURE_COLUMNS |
| Cell 1 (data) | Fetch stock only | Fetch stock + SPY, compute indicators on both, build relative features |
| Cell 2 (Manual) | No change | Threshold values will need re-tuning (relative values have different scales) |
| Cell 3 (Classification) | No change | Trains on relative features automatically |
| Cell 4 (Q-Learning) | No change | Bin edges adapt to data automatically |
| Cell 5 (comparison) | Equity simulation uses `df["close"]` | Still uses raw `df_stock["close"]` for equity sim (we trade the actual stock, not the ratio) |
| Cell 6 (export) | No change | Exports as before |

### 4. Manual model threshold re-tuning

Current thresholds assume raw indicator ranges (e.g., RSI 40/65, CCI -100/+100). Relative features have different distributions:

- `rsi` ratio: centered ~1.0 (stock RSI / SPY RSI). Buy when <0.7 (stock oversold vs market), sell when >1.3
- `macd_hist` diff: centered ~0. Buy when stock hist exceeds SPY hist, sell when below
- `cci` diff: centered ~0. Thresholds might be ±50 instead of ±100
- `bbp` diff: centered ~0. Thresholds might be ±0.3 instead of ±0.8
- `adx` ratio: centered ~1.0. >1.5 means stock trending harder than market
- `williams_r` diff: centered ~0. Thresholds might be ±30 instead of ±80
- `obv_slope` diff: centered ~0. Direction (>0 / <0) still meaningful

**These will need empirical tuning** — run the notebook, inspect distributions, then set thresholds at reasonable percentiles (e.g., 20th/80th).

### 5. LEAN main.py impact

`main.py` must also compute SPY indicators and do the relative transform. This means:

- Add `self.spy_symbol = self.AddEquity("SPY", Resolution.Daily).Symbol` in `Initialize()`
- `_build_feature_frame()` fetches history for both symbols, computes indicators for both, then applies the same ratio/difference transform
- The model artifacts (trees, Q-table, rules) already encode relative features from training — no artifact format changes

### 6. Benchmark chart adjustment

The comparison chart currently normalizes stock price as "Buy & Hold". With relative indicators, we should show **both**:
- Stock Buy & Hold (raw)
- SPY Buy & Hold (raw)
- Model equity curves

This lets the user see whether the model captures alpha over the market, not just absolute returns.

---

## What Does NOT Change

- `common/` library — no modifications needed
- Model interfaces (`train`, `predict`, `save`, `load`) — unchanged
- JSON artifact format — unchanged
- Live trading runner — unchanged (uses same main.py logic)
- Indicator registry — unchanged (same indicators, just applied to two symbols)

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| SPY data gaps or misaligned dates | Low | Inner-join on date index before computing ratios |
| Division by zero (SPY indicator = 0) | Low | Replace 0 with NaN, then dropna |
| Relative features have unfamiliar distributions | Medium | Inspect histograms before setting Manual thresholds; ML models adapt automatically |
| Regime change (stock decouples from SPY) | Medium | Monitor correlation; relative signals weaken when correlation drops |
| Overfitting to stock/SPY correlation | Medium | Out-of-sample validation with separate date range |
| Sector-specific moves diluted by broad index | Low | Could use sector ETF (e.g., SMH for NVDA) instead of SPY in future |

---

## Implementation Order

1. Fetch SPY data alongside stock (trivial)
2. Build `compute_relative_features()` helper in notebook
3. Verify ML models (Classification, Q-Learning) train without errors on relative features
4. Inspect relative feature distributions, re-tune Manual thresholds
5. Update comparison charts to include SPY benchmark
6. Update LEAN `main.py` to compute relative features
7. Run LEAN backtest to validate
