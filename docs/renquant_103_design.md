# renquant_103 — Design Specification

**Status**: Design / Pre-implementation review  
**Author**: Ren Hao  
**Date**: 2026-04-10  
**Based on**: renquant_102 (multi-stock pre-trained scanner)

---

## 1. Motivation

renquant_102 is a single-mode momentum system: it always hunts volume-spike breakouts regardless of market conditions. It works in calm bull markets but structurally fails in the current environment (VIX oscillating 19–32, large single-day macro-driven reversals, tariff/geopolitical headline risk). The failures are:

1. **No regime awareness** — regime filter exists in code but is disabled; the system trades identically in trending and choppy markets
2. **Entry direction never adapts** — always buys strength (momentum), which is the wrong approach when the market is mean-reverting
3. **Crowded entry signal** — volume spike + up-close is one of the most widely used retail signals; edge is thin in liquid equities
4. **Model used as binary gate, not ranker** — the expensive ML model just says yes/no instead of driving candidate ranking
5. **No earnings awareness** — volume spikes on earnings day are noise; the models were never trained on earnings outcomes
6. **Correlated candidate selection** — sector guard limits concentration but doesn't optimize for diversification within the selected positions
7. **Lagging training data** — `sample_end: 2024-12-31`; models haven't seen 2025–2026 volatility regimes

renquant_103 addresses all of these while reusing the core architecture (pre-trained per-symbol models, 3-stage DETECT → CONFIRM → EXECUTE, BagLearner/classification/Q-learning tournament, JSON artifacts, shared LEAN pipeline).

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  REGIME DETECTION                        │
│                                                          │
│  Layer 1 (Hurst)     Layer 2 (Changepoint)  Layer 3 (GMM) │
│  "What regime?"      "Did it just change?"  "How sure?"  │
│       ↓                     ↓                    ↓       │
│            → composite regime signal + confidence ←      │
└─────────────────────────┬───────────────────────────────┘
                          │  regime + confidence
                          ↓
┌─────────────────────────────────────────────────────────┐
│              REGIME-ADAPTIVE PARAMETERS                  │
│   position_size, stop_loss, max_hold, cash_reserve       │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────┐
│           STOCK SELECTION PIPELINE (per bar)             │
│                                                          │
│  [Earnings filter]                                       │
│       ↓                                                  │
│  [Regime-conditional volume scan]                        │
│   BULL_CALM:      volume P85 + up-close                  │
│   BULL_VOLATILE:  volume P85 + down-close (capitulation) │
│   CHOPPY:         stock outperformed SPY last 5d         │
│   BEAR:           no new buys                            │
│       ↓                                                  │
│  [Relative strength ranking vs sector]                   │
│       ↓                                                  │
│  [Per-symbol model — continuous score, not binary gate]  │
│       ↓                                                  │
│  [Combined rank = 0.5×RS + 0.5×model_score]             │
│       ↓                                                  │
│  [Correlation-aware greedy selection]                    │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ↓
              EXECUTE (same as 102)
```

---

## 3. Layer 1 — Hurst Exponent (Slow Baseline)

### What it measures
Long-range memory of the SPY return series. Directly answers: **"Is the market in a momentum or mean-reversion regime?"**

- H > 0.55 → momentum regime (returns are positively autocorrelated — trends persist)
- H = 0.45–0.55 → ambiguous / random walk (no exploitable directional pattern)
- H < 0.45 → mean-reversion regime (returns are negatively autocorrelated — moves reverse)

### Computation
Rescaled Range (R/S) analysis on rolling 63-day (≈ 3 months) SPY daily returns.

```python
def compute_hurst(returns: pd.Series, max_lag: int = 40) -> float:
    """
    Rolling Hurst exponent via R/S analysis.
    Returns H in [0, 1]. 0.5 = random walk.
    """
    lags = range(2, min(len(returns) // 2, max_lag))
    rs_vals = []
    for lag in lags:
        chunks = [returns.iloc[i:i+lag].values
                  for i in range(0, len(returns) - lag, lag)]
        rs_chunk = []
        for chunk in chunks:
            mean = chunk.mean()
            devs = np.cumsum(chunk - mean)
            R = devs.max() - devs.min()
            S = chunk.std(ddof=1)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_vals.append(np.mean(rs_chunk))
    if len(rs_vals) < 2:
        return 0.5  # fail safe: assume random walk
    poly = np.polyfit(np.log(list(lags)[:len(rs_vals)]), np.log(rs_vals), 1)
    return float(np.clip(poly[0], 0.0, 1.0))
```

### Update frequency
Recomputed daily on rolling 63-day window. Slow-moving — changes meaningfully over weeks, not days. This is the ground-truth regime characterizer, not the trigger.

### Output
`hurst_regime ∈ {MOMENTUM, AMBIGUOUS, REVERSION}`

---

## 4. Layer 2 — Changepoint Detection (Fast Transition Trigger)

### What it measures
When the statistical distribution of SPY returns has shifted — i.e., a regime transition just occurred. Does NOT characterize the new regime, only signals that something changed.

### Why needed
Hurst takes 60+ days to respond to a new regime. Changepoint catches transitions within 2–5 days. Together: Hurst tells you what regime you're in; Changepoint tells you when you entered it.

### Algorithm: CUSUM (Cumulative Sum Control Chart)
CUSUM accumulates deviations from an expected mean. When the cumulative sum exceeds a threshold, a changepoint is flagged.

```python
def cusum_changepoint(returns: pd.Series,
                      threshold: float = 3.0,
                      drift: float = 0.5) -> bool:
    """
    Returns True if a changepoint was detected in the most recent window.
    threshold: sensitivity (lower = more triggers, higher = fewer)
    drift: allowance before accumulating (reduces noise sensitivity)
    """
    s_pos, s_neg = 0.0, 0.0
    mu = returns.mean()
    sigma = returns.std(ddof=1) if returns.std(ddof=1) > 0 else 1.0
    for r in returns:
        z = (r - mu) / sigma
        s_pos = max(0, s_pos + z - drift)
        s_neg = max(0, s_neg - z - drift)
        if s_pos > threshold or s_neg > threshold:
            return True
    return False
```

### Usage in strategy
- Run CUSUM on rolling 20-day SPY returns each bar
- If changepoint detected: flag `regime_transition = True` for current bar
- On transition: immediately re-evaluate regime (re-run Hurst on last 30d, re-query GMM)
- Transition flag also triggers position size reduction by 50% for next 3 bars (uncertainty window)

### Output
`regime_transition ∈ {True, False}`, `transition_uncertainty_bars_remaining`

---

## 5. Layer 3 — GMM Clustering (Continuous Confidence Scoring)

### What it measures
Classifies each market bar into a regime cluster and outputs a **probability vector** — not a hard label. This probability drives continuous position sizing instead of discrete parameter steps.

### Features (4-dimensional feature vector per bar, computed on SPY)
| Feature | Computation | Why |
|---------|------------|-----|
| `10d_return` | SPY 10-day log return | Recent trend direction |
| `20d_realized_vol` | Std of SPY daily returns × √252 | Vol regime |
| `spy_adx` | ADX(14) on SPY | Trend strength |
| `return_autocorr` | Autocorr of SPY 10d returns, lag=1 | Momentum persistence |

### Training (notebook, offline)
- Fit GMM with k=3 components on SPY daily feature vectors from 2018–2025
- Covers: 2018 vol spike, 2020 COVID crash/recovery, 2021 bull, 2022 bear, 2023–24 recovery, 2025 tariff vol
- Inspect and label each cluster post-hoc: `BULL_CALM`, `BULL_VOLATILE`, `BEAR`
- Serialize GMM parameters (means, covariances, weights) to JSON for LEAN

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, covariance_type='full',
                      random_state=42, n_init=10)
gmm.fit(feature_matrix)  # shape: (n_days, 4)

# Inspect clusters — assign labels by cluster mean vol and return
# Low vol + positive return → BULL_CALM
# High vol + mixed return → BULL_VOLATILE
# Negative return + high vol → BEAR

# Serialize
import json
gmm_artifact = {
    "means": gmm.means_.tolist(),
    "covariances": gmm.covariances_.tolist(),
    "weights": gmm.weights_.tolist(),
    "cluster_labels": ["BULL_CALM", "BULL_VOLATILE", "BEAR"],  # manual assignment
    "feature_order": ["10d_return", "20d_realized_vol", "spy_adx", "return_autocorr"]
}
with open("backtesting/renquant_103/spy-gmm-regime.json", "w") as f:
    json.dump(gmm_artifact, f)
```

### Usage in strategy (LEAN)
Each bar: compute feature vector → compute log-likelihood for each cluster → softmax → P(cluster).

```python
def gmm_predict_proba(self, feature_vec: list) -> dict:
    """
    Returns P(BULL_CALM), P(BULL_VOLATILE), P(BEAR) for current bar.
    Uses pre-loaded GMM parameters (means, covariances, weights).
    """
    x = np.array(feature_vec)
    log_probs = []
    for k in range(self._gmm_n_components):
        mu = self._gmm_means[k]
        sigma = self._gmm_covs[k]
        diff = x - mu
        sign, logdet = np.linalg.slogdet(sigma)
        inv_sigma = np.linalg.inv(sigma)
        mahal = diff @ inv_sigma @ diff
        log_p = -0.5 * (mahal + logdet) + np.log(self._gmm_weights[k])
        log_probs.append(log_p)
    # softmax
    log_probs = np.array(log_probs)
    log_probs -= log_probs.max()
    probs = np.exp(log_probs)
    probs /= probs.sum()
    return {label: float(p) for label, p in zip(self._gmm_labels, probs)}
```

### Output
`gmm_probs = {"BULL_CALM": 0.72, "BULL_VOLATILE": 0.21, "BEAR": 0.07}`

---

## 6. Combining the Three Layers

### Regime Resolution Logic
```
if hurst_regime == MOMENTUM and not regime_transition:
    base_regime = BULL_CALM
elif hurst_regime == REVERSION and not regime_transition:
    base_regime = CHOPPY
else:
    base_regime = AMBIGUOUS

# GMM refines base_regime with probability weights
dominant_gmm = argmax(gmm_probs)

if dominant_gmm == BEAR and gmm_probs["BEAR"] > 0.5:
    final_regime = BEAR  # GMM bear signal overrides (protective)
elif base_regime == AMBIGUOUS:
    final_regime = dominant_gmm  # fall back to GMM when Hurst uncertain
else:
    final_regime = base_regime   # Hurst is authoritative when confident

# Transition uncertainty: reduce confidence for 3 bars after changepoint
if transition_uncertainty_bars_remaining > 0:
    regime_confidence = 0.5  # half confidence during transition
else:
    regime_confidence = gmm_probs[final_regime]
```

### Regime Confidence → Continuous Position Scaling
```
effective_position_pct = base_position_pct × regime_confidence
```

This means the strategy never hard-switches between 30% and 15% positions — it glides smoothly as confidence changes. No flickering.

---

## 7. Regime-Adaptive Parameters

| Parameter | BULL_CALM | BULL_VOLATILE | CHOPPY | BEAR |
|-----------|-----------|---------------|--------|------|
| Max position | 30% | 20% | 15% | 0% (no buys) |
| Cash reserve | 0% | 20% | 30% | 100% |
| Stop-loss | 8% | 5% | 5% | 5% (existing) |
| Max hold days | 30 | 15 | 10 | 15 (existing) |
| Entry direction | Momentum | Capitulation | Divergence | Blocked |
| Min model score | 0.10 | 0.15 | 0.15 | — |

All of these are scaled by `regime_confidence` for continuous adaptation.

---

## 8. Regime-Conditional Entry Signal

### BULL_CALM — Momentum Entry (same as 102)
- Volume today ≥ P85 of rolling 20-day lookback
- Today's close > yesterday's close
- Signals: buy strength, momentum expected to continue

### BULL_VOLATILE — Capitulation Entry (new)
- Volume today ≥ P85 of rolling 20-day lookback (high volume required)
- Today's close **< yesterday's close** (down day — reversal of 102)
- Today's close must be in **bottom 30% of 5-day range** (not just any down day — proper dip)
- Rationale: in volatile markets, high volume down days represent panic selling / capitulation; mean reversion to follow

### CHOPPY — Divergence Entry (new)
- No volume spike required (volume signal is too noisy in choppy markets)
- Stock 5-day return **outperforms SPY 5-day return by > 1%** (relative strength despite market weakness)
- Stock RSI (relative to SPY RSI) shows improving trend over last 3 bars
- Rationale: in a choppy market, stocks showing hidden strength relative to the index are accumulation candidates

### BEAR — Blocked
- No new buys under any conditions
- Existing positions: tighten stop-loss to 5%, exit on any sell signal (min_hold ignored)

---

## 9. Stock Selection Pipeline (Within a Regime)

### Step 1: Earnings Calendar Filter
Exclude any stock within ±3 trading days of its earnings announcement date.

- Volume spikes near earnings are noise by construction
- The per-symbol models were never trained to predict earnings outcomes
- Implementation: maintain earnings date lookup from a pre-downloaded schedule (JSON artifact, updated weekly)

### Step 2: Regime-Conditional Volume/Signal Scan
Apply the regime-appropriate filter from Section 8. Produces a candidate list.

### Step 3: Relative Strength Ranking
Rank candidates by performance relative to their sector ETF over 20 days:

```
rs_score(stock) = stock_20d_return - sector_etf_20d_return
```

Sector ETFs used: XLK (tech), XLF (finance), XLV (healthcare), XLE (energy), XLI (industrials).

This measures institutional capital flows within sectors — less crowded than raw price/volume signals. Stocks at the top of their sector are where sector rotation money is flowing.

### Step 4: Per-Symbol Model — Continuous Score
Apply the pre-trained per-symbol model to each candidate. Instead of thresholding to buy/sell/hold, extract the **raw score** (classification: forest average, qlearning: Q-value difference between buy and hold actions).

Use model score as a continuous number, not a binary gate.

### Step 5: Combined Ranking
```
combined_rank = 0.5 × normalize(rs_score) + 0.5 × normalize(model_score)
```

Both components normalized to [0, 1] across the current candidate set. Sort descending.

### Step 6: Correlation-Aware Greedy Selection
Greedily select from the ranked list, skipping any candidate whose **30-day rolling correlation with any already-selected or already-held position exceeds 0.70**.

```python
def select_with_correlation_guard(ranked_candidates, held_positions,
                                   corr_matrix, n_slots, max_corr=0.70):
    selected = []
    existing = list(held_positions)
    for ticker, score in ranked_candidates:
        if len(selected) >= n_slots:
            break
        all_current = existing + selected
        if all(corr_matrix.loc[ticker, held] < max_corr
               for held in all_current if held in corr_matrix.columns):
            selected.append(ticker)
    return selected
```

Correlation matrix: computed in notebook on 60-day rolling returns, serialized to JSON, loaded in LEAN.

### Step 7: Wash Sale + Sector Guard
Same as 102 — applied after correlation selection.

---

## 10. Model Training Changes (Notebook)

### 10a. New Training Features
Add regime context as features so per-symbol models are regime-aware:

| New feature | Computation | Added to |
|-------------|------------|---------|
| `spy_realized_vol` | SPY 20d std × √252, normalized | All models |
| `spy_adx` | ADX(14) on SPY | All models |
| `spy_trend` | SPY close / SPY EMA50 | All models |
| `hurst_proxy` | Autocorr of SPY 10d returns, lag=1 | All models |

These are **market-context features** — they tell the model what regime it's operating in, so it can implicitly learn regime-conditional behavior.

### 10b. Shorter Lookahead: 5 Days (was 10)
A 10-day prediction horizon is too long when macro headlines can reverse the market in 48 hours. Reduce to 5 days:
- Faster, more responsive signals
- More realistic in the current regime
- More training labels per stock (each 5-day window is a sample vs 10-day)
- Tradeoff: slightly more trading frequency → more tax drag (acceptable given improved responsiveness)

### 10c. Extended Training Data
```
sample_start: "2019-01-01"   # was 2021-01-01
sample_end:   "2025-12-31"   # was 2024-12-31
training_years: 3            # was 2
```

The 2019-2021 period adds pre-COVID bull + crash dynamics. The 2025 extension adds the tariff vol regime the models have never seen. Using 3-year rolling window gives more regime diversity per training run.

### 10d. OOS Sharpe Floor Raised
From 0.8 → 0.85. With more training data and regime features, models that don't pass a higher bar should be excluded rather than kept as weak signals. Better to skip a symbol than include a poor model.

---

## 11. Watchlist Changes

### Removals (high-beta, correlated tech with weak OOS Sharpe)
| Ticker | Reason |
|--------|--------|
| ARKK | Extremely high-beta, OOS Sharpe 0.85 (barely passes), duplicate tech exposure |
| SHOP | OOS Sharpe 0.80 (floor), high correlation with other tech names |
| COIN | Most volatile (crypto proxy), poor in non-momentum regimes |

### Additions (defensive + uncorrelated)
| Ticker | Sector | Rationale |
|--------|--------|-----------|
| GLD | Commodity | Safe haven in BEAR/BULL_VOLATILE, low correlation to equities |
| TLT | Bond ETF | Counter-cyclical in risk-off; long bonds rally in fear regimes |
| XLV | Healthcare | Defensive sector, lower beta, performs in late-cycle |
| XLU | Utilities | Highest defensive quality, near-zero correlation to tech |

These additions give the strategy something to buy in non-BULL_CALM regimes rather than going entirely to cash.

### Revised Watchlist (24 symbols)
```json
["TSLA", "AMZN", "GOOG", "MSFT", "AMD", "NFLX",
 "CRM", "PLTR", "UBER",
 "JPM", "UNH", "XOM", "CAT", "BA",
 "XLE", "XLF",
 "NVDA", "LLY",
 "GLD", "TLT", "XLV", "XLU",
 "META", "AAPL"]
```

### Revised Sector Map
```json
{
  "TSLA": "tech", "AMZN": "tech", "GOOG": "tech", "MSFT": "tech",
  "AMD": "tech", "NFLX": "tech", "CRM": "tech", "PLTR": "tech",
  "UBER": "tech", "NVDA": "tech", "META": "tech", "AAPL": "tech",
  "JPM": "finance", "XLF": "finance",
  "UNH": "healthcare", "LLY": "healthcare", "XLV": "healthcare",
  "XOM": "energy", "XLE": "energy",
  "CAT": "industrial", "BA": "industrial",
  "GLD": "commodity",
  "TLT": "bond",
  "XLU": "utility"
}
```

`max_positions_per_sector`: 3 → keep for equity sectors; apply separately for defensives (GLD, TLT, XLU can hold simultaneously since they serve different defensive purposes).

---

## 12. Risk Management Changes

### Regime-Adaptive Stop-Loss (new)
Stop-loss is no longer a fixed 8%. It adapts:

```
BULL_CALM:      8% (same as 102 — give room to breathe)
BULL_VOLATILE:  5% (tighter — preserve capital in whipsaw)
CHOPPY:         5% (mean-reversion entries should be tight)
BEAR:           5% (defensive)
```

### Transition Uncertainty Window (new)
For 3 bars after a Layer 2 changepoint:
- Do not open new positions
- Tighten stop-loss on existing positions by 1%
- This prevents entering right at the moment of a regime flip when signals are most unreliable

### Portfolio Drawdown Circuit Breaker
Keep 15% from 102, but add a **regime-conditional reduction**:
- BULL_CALM: 15% halt threshold (same)
- BULL_VOLATILE: 10% halt threshold (earlier protection)
- CHOPPY: 8% halt threshold (very conservative)

### Trailing Stop (new, optional per regime)
In BULL_CALM: after a position gains >5%, switch from fixed stop to trailing stop at 5% below the high watermark of that position. Locks in profit while letting winners run.

---

## 13. Correlation Matrix Artifact

New artifact required by the correlation-aware selection (Step 6 of Section 9):

- Computed in notebook: 60-day rolling pairwise correlation of all watchlist symbol returns
- Serialized to `backtesting/renquant_103/watchlist-correlation.json`
- Loaded in LEAN at initialization
- Updated on each model retrain (same schedule as per-symbol models)

---

## 14. New Artifacts Summary

| Artifact | Path | Updated by |
|----------|------|------------|
| Per-symbol models | `models/{SYM}/*` | Notebook (same as 102) |
| SPY GMM regime model | `spy-gmm-regime.json` | Notebook (new) |
| Watchlist correlation matrix | `watchlist-correlation.json` | Notebook (new) |
| Earnings calendar | `earnings-calendar.json` | Script (weekly, new) |

---

## 15. What Is Unchanged from 102

- Pre-trained per-symbol model architecture (BagLearner / RTLearner / Q-learning tournament)
- JSON-only artifacts (no pickle, LEAN-compatible)
- `export_lean_watchlist.py` for LEAN data export
- `backtest_and_analyze.py` for results + notifications
- `live/runner.py` for live trading
- Wash-sale guard (30 days)
- Tax tracking (ST/LT rates)
- Telemetry charts and runtime statistics
- Model staleness guard (60 days)
- `new_strategy.py` scaffold script

---

## 16. Implementation Roadmap

### Phase 1 — Notebook (Research)
1. Add SPY GMM training to renquant_103 notebook
2. Add Hurst exponent and CUSUM functions to `common/indicators/`
3. Extend training data window (`sample_end: 2025-12-31`, `training_years: 3`)
4. Add regime features to per-symbol model training
5. Change lookahead from 10 → 5 days
6. Compute and serialize correlation matrix
7. Scrape/prepare earnings calendar artifact
8. Train all symbols, verify OOS Sharpe ≥ 0.85, export models

### Phase 2 — LEAN Execution (backtesting/renquant_103/main.py)
1. Scaffold strategy from 102 (`new_strategy.py` or manual copy)
2. Implement 3-layer regime detection in `main.py`
3. Implement regime-adaptive parameter switching
4. Replace entry logic with regime-conditional scan (Section 8)
5. Implement relative strength ranking (Section 9, Step 3)
6. Replace binary model gate with continuous model scoring
7. Implement correlation-aware greedy selection
8. Implement earnings calendar filter
9. Add trailing stop logic for BULL_CALM
10. Update `strategy_config.json` with all new parameters

### Phase 3 — Validation
1. `export_lean_watchlist.py --strategy renquant_103`
2. Run LEAN backtest: 2024-01-01 → 2026-03-26
3. Compare against 102 baseline on same period
4. Check regime telemetry: verify regime signals make intuitive sense (e.g., early 2025 tariff shock flagged as BEAR/BULL_VOLATILE)
5. Tune CUSUM threshold and GMM cluster assignments if needed
6. Paper trade via `live/runner.py --broker alpaca-paper`

---

## 17. Decisions Log

All open questions resolved on 2026-04-10.

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | CUSUM threshold sensitivity | **3.0** | Balanced sensitivity; tune down to 2.5 if too few triggers in validation |
| 2 | Hurst window | **63 days** | ~3 months, more responsive than 90d while still stable |
| 3 | GMM components | **k=3** | BULL_CALM / BULL_VOLATILE / BEAR — clean and interpretable; extend to k=4 only if validation reveals a distinct 4th state |
| 4 | Earnings calendar source | **yfinance** (`Ticker.calendar`) | Already in stack; refresh weekly via script |
| 5 | Correlation guard threshold | **0.70** | Standard threshold; tighten to 0.65 if portfolio still too correlated in validation |
| 6 | GLD/TLT entry signal | **Yes — counter-cyclical** | GLD and TLT use BEAR regime as a BUY trigger (inverse of equity logic); they are also eligible in BULL_VOLATILE as partial hedge positions |
| 7 | Run alongside 102 or replace? | **TBD — review backtest charts first** | Run LEAN backtest on 103, compare equity curves vs 102 on same period, then decide |
| 8 | Combined rank weights | **50/50** (RS score + model score) | Equal weight as starting point; tune after first backtest by checking which component is more predictive |
