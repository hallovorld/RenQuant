# renquant_103 — Design Specification

**Status**: Implemented and live (active daily strategy)
**Author**: Ren Hao  
**Last updated**: 2026-04-20  
**Based on**: renquant_102 (multi-stock pre-trained scanner)

> **Note**: This document started as a design spec. Sections marked with ⚠️ contain decisions that evolved during implementation — see inline notes for the actual values in the codebase.

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

renquant_103 addresses all of these while reusing the core architecture (pre-trained per-symbol models, champion-model-per-symbol tournament, JSON artifacts, shared LEAN pipeline).

### Implemented scoring note

The original design assumed `model_score` could be compared directly across model families. That turned out to be false in practice because Manual, Classification, Q-Learning, and XGBoost emit different raw score types. The live implementation now uses a two-part score contract:

- `raw_score`: the model's native output, kept for logging and debugging
- `rank_score`: a calibrated probability score `P(outperform SPY by threshold% in lookahead days)`, used for filtering, tier thresholds, and cross-symbol ranking

Calibration lives in `kernel/scoring.py` and selects its method by sample size:
- **n ≥ 300**: isotonic regression (step-function, rich data)
- **120 ≤ n < 300**: Platt scaling (logistic regression sigmoid — smooth, no tail overfitting)
- **n < 120**: constant base-rate

`score_calibration` is baked into `policy-metadata.json` at notebook training time and **refreshed daily** by `scripts/recalibrate_scores.py` after each model retrain. If the metadata has no `score_calibration`, the runner computes a runtime fallback from recent history. This preserves one champion model per symbol without introducing ad hoc multi-model ensembling at execution time.

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
│  [Regime gates + candidate scan]                         │
│   Current implementation: no separate volume DETECT gate │
│   Offensive regimes: model buy signal is the trigger     │
│   BEAR: defensives only                                  │
│       ↓                                                  │
│  [Relative strength ranking vs sector]                   │
│       ↓                                                  │
│  [Per-symbol champion model — raw score + calibration]   │
│       ↓                                                  │
│  [Combined rank = w_rs×RS + w_rank×rank_score]           │
│   weights from ranking.blend_weights in strategy_config  │
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

- H > 0.65 (hurst_trending_threshold) → momentum regime (returns are positively autocorrelated — trends persist)
- H = 0.45–0.55 → ambiguous / random walk (no exploitable directional pattern)
- H < 0.52 (hurst_reversion_threshold) → mean-reversion regime (returns are negatively autocorrelated — moves reverse)

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
elif final_regime == "CHOPPY":
    # GMM has two near-equal clusters (~48/48%) so its CHOPPY posterior is structurally
    # uninformative (~50% regardless of market state). Use Hurst distance instead:
    # lower H (stronger mean-reversion) → higher CHOPPY confidence.
    regime_confidence = (0.45 - hurst) / (0.45 - choppy_hurst_floor)  # floor=0.20
    regime_confidence = clamp(regime_confidence, 0.0, 1.0)
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

⚠️ **Actual values in `strategy_config.json` (evolved from design):**

| Parameter | BULL_CALM | BULL_VOLATILE | CHOPPY | BEAR |
|-----------|-----------|---------------|--------|------|
| Max position | 15% | 20% | 15% | 0% (offensive) |
| Cash reserve | 0% | 20% | 30% | 100% |
| Stop-loss | 15% | 5% | 5% | 5% (existing) |
| Max hold days | 500 | 500 | 40 | 500 (existing) |
| Trailing stop trigger | 20% gain | — | — | — |
| Trailing stop trail | 18% below HWM | — | — | — |
| Drawdown halt | 35% | 10% | 8% | 5% |
| Entry direction | Momentum | Capitulation | Divergence | Defensives only |
| Min model score | 0.10 | 0.15 | 0.15 | — |

BEAR regime allows 1 defensive position (GLD/TLT/XLV/XLU) at 15% of portfolio — offensive buys are blocked but defensives can be entered. Stop-loss of 15% in BULL_CALM was widened from the original 8% design to give high-beta tech names room to breathe through normal corrections.

---

## 8. Regime-Conditional Entry Signal

### Actual entry trigger in code

The current notebook, LEAN, and live runner no longer use a separate regime-conditional volume DETECT stage for renquant_103. Instead:

- offensive regimes use the per-symbol champion model's `buy` signal as the entry trigger
- BEAR blocks offensive buys and allows only the best-ranked defensive ticker
- candidate filtering uses a minimum calibrated `rank_score`, not heterogeneous raw scores
- final ranking uses `w_rank × rank_score + w_rs × relative_strength` where weights are data-driven (default 0.5/0.5, updated daily by `recalibrate_scores.py`)
- slot escalation (`tiered_thresholds`) also applies to calibrated `rank_score`

This was a deliberate correction after observing that raw scores from different model families were not comparable enough to support clean portfolio ranking.

---

## 9. Stock Selection Pipeline (Within a Regime)

### Step 1: Earnings Calendar Filter
Exclude any stock within ±3 trading days of its earnings announcement date.

- Volume spikes near earnings are noise by construction
- The per-symbol models were never trained to predict earnings outcomes
- Implementation: maintain earnings date lookup from a pre-downloaded schedule (JSON artifact, updated weekly)

### Step 2: Signal Scan
Apply the regime gates, then use the per-symbol champion model's `buy` signal to produce a candidate list. There is no separate volume DETECT stage in the implemented renquant_103 execution path.

### Step 3: Relative Strength Ranking
Rank candidates by performance relative to their sector ETF over 20 days:

```
rs_score(stock) = stock_20d_return - sector_etf_20d_return
```

Sector ETFs used: XLK (tech), XLF (finance), XLV (healthcare), XLE (energy), XLI (industrials).

This measures institutional capital flows within sectors — less crowded than raw price/volume signals. Stocks at the top of their sector are where sector rotation money is flowing.

### Step 4: Per-Symbol Model — Raw Score plus Calibration
Apply the pre-trained champion model to each candidate. The runner keeps the native `raw_score` (classification: forest average, qlearning: Q-value difference, xgboost: `P(buy)-P(sell)`, manual: vote count) for logging, then converts it into a calibrated `rank_score` for cross-model comparison.

### Step 5: Combined Ranking
```
combined_rank = w_rank × normalize(rank_score) + w_rs × normalize(rs_score)
```

Both components normalized to [0, 1] across the current candidate set. Sort descending.

Weights `[w_rank, w_rs]` are stored in `strategy_config.json` under `ranking.blend_weights` and default to `[0.5, 0.5]`. They are updated daily by `scripts/recalibrate_scores.py` using a logistic regression on `[norm(rank_score), norm(rs_score)]` against actual forward outperformance outcomes across all watchlist symbols. The positive coefficients are normalised into live blend weights.

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

### Step 7: Selection Loop Order
For each candidate (in combined-rank order), the checks run in this exact sequence:
1. **Tiered threshold on calibrated `rank_score`** — slot 1: 0.10, slot 2: 0.30, slot 3: 0.50
2. **Wash-sale guard** — skip if sold < 30 days ago
3. **Sector guard** — skip if sector already has ≥3 positions (defensives exempt)
4. **Correlation guard** — skip if pairwise correlation with any held or already-selected ticker ≥0.70

This order matters: wash-sale and sector guard are cheap filters applied before the more expensive correlation lookup.

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

### 10d. OOS Sharpe Floor
⚠️ Design proposed raising to 0.85. **Actual implementation uses 0.8** (matching renquant_102) — raising further reduced the number of qualifying models too aggressively on the 24-ticker watchlist.

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
BULL_CALM:      15% (wide — give high-beta tech names room to breathe through normal corrections)
BULL_VOLATILE:  5%  (tighter — preserve capital in whipsaw)
CHOPPY:         5%  (mean-reversion entries should be tight)
BEAR:           5%  (defensive)
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

### Trailing Stop (implemented)
In BULL_CALM: activates after a position gains ≥20% from entry, then trails at 18% below the position's rolling high-water mark. ⚠️ Design proposed 5%/5% — widened to 20%/18% to avoid premature exits on high-beta tech stocks during normal intraday/weekly corrections.

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
| SPY GMM regime model | `artifacts/spy-gmm-regime.json` | Notebook (new) |
| Watchlist correlation matrix | `artifacts/watchlist-correlation.json` | Notebook (new) |
| Earnings calendar | `artifacts/earnings-calendar.json` | Script (weekly, new) |

---

## 16. Strategy Kernel (`backtesting/renquant_103/kernel/`)

All strategy-specific logic is extracted into a self-contained Python package with **zero `common/` imports**. This solves the LEAN/notebook parity gap: LEAN Docker cannot access `common/`, so previously all logic was manually duplicated into `main.py`. Now kernel code is the single source of truth.

### Modules

| Module | Key exports |
|--------|-------------|
| `kernel/config.py` | `BULL_CALM`, `BULL_VOLATILE`, `CHOPPY`, `BEAR`, `REGIMES`, `artifact_path()` |
| `kernel/regime.py` | `RegimeState`, `detect_regime()`, `load_gmm_artifact()` |
| `kernel/indicators.py` | `compute_all()`, `build_feature_frame()` |
| `kernel/models.py` | `load_artifact()`, `score_artifact()`, `calibrate_score()`, `predict_classification()`, `predict_manual()` |
| `kernel/exits.py` | `HoldingState`, `ExitSignal`, `compute_exits()` (5-exit priority order) |
| `kernel/selection.py` | `CandidateResult`, `SelectionContext`, `score_candidates()`, `run_selection_loop()`, `is_wash_sale_blocked()`, `is_earnings_blocked()` |
| `kernel/sizing.py` | `compute_position_size()` (with oversize fallback at 25%) |

### How it's consumed

- **LEAN `main.py`**: Thin ~300-line wrapper. Imports kernel locally (`from kernel.x import ...` — same pattern as `from config import ...`).
- **`live/runner.py`**: Auto-detects kernel at startup (`_load_kernel()`): if `kernel/__init__.py` exists, adds strategy dir to `sys.path` and sets `config["_use_kernel"] = True`. All model loading, regime detection, and scoring then routes through kernel modules.
- **CI enforcement**: `tests/test_kernel_isolation.py` asserts every `kernel/*.py` file contains no `import common` or `from common` statement (AST-parsed, not regex).

### Artifact path

Strategy-level artifacts moved to `artifacts/` subdir to separate from `models/{SYM}/`:
```
backtesting/renquant_103/artifacts/
  spy-gmm-regime.json
  watchlist-correlation.json
  earnings-calendar.json
```

---

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

## 16. Post-Design Additions (Implemented)

The following features were added after the initial design based on backtesting results:

### SPY EMA50 Trend Gate
Blocks all new offensive buys when SPY is below its 50-day EMA. Prevents entering individual stocks during macro downtrends where all technical signals are overwhelmed by market-wide selling. Applied in both LEAN `main.py` and the notebook simulation.

### Fixed Training Cutoff + Expanding-Window Live Models
- **Backtest simulation**: Models trained on 2016–2023 (fixed cutoff `2024-01-01`). OOS evaluation on 2024+. Prevents training boundary from shifting with each new day's data, eliminating simulation variance between notebook runs.
- **Live trading**: After backtest export, each model is retrained on the last 4 years up to today and re-exported. Live runner always uses the most current data without contaminating backtest OOS metrics.

### Live Model Score Ranking
Simulation, LEAN, and live runner all rank buy candidates by today's actual model output (continuous confidence from `predict_score_bulk()`) rather than by static OOS Sharpe. Ensures the highest-conviction signal on a given day is executed first. Also applies a `min_model_score=0.10` threshold to filter out weak signals before ranking.

`predict_score_bulk()` is now implemented on all four model types: Classification (raw BagLearner output), Q-Learning (Q(buy)−Q(sell)), XGBoost (P(buy)−P(sell)), Manual (raw vote count). The live runner uses `_get_model_score()` helper with priority: `predict_score_bulk` → `predict_score` → string fallback.

### Tiered Thresholds
Each successive buy slot in a single day requires a progressively higher model score bar, preventing overcommitment on low-conviction multi-candidate days.

- Slot 1: `min_model_score = 0.10`
- Slot 2: `min_model_score = 0.30`
- Slot 3: `min_model_score = 0.50`

Configured in `tiered_thresholds` array in `strategy_config.json`. Logic is identical across LEAN (`main.py`), notebook simulation (Cell 21), and live runner (`run_once_multi`): `tier_idx = min(slots_filled, len(tiers) - 1)`.

### Single-Day Loss Gate
Exits a position when today's close drops ≥10% from yesterday's close (`max_single_day_loss_pct: 0.10` in BULL_CALM regime). Addresses the gap-risk limitation of daily-bar resolution: with a 15% cumulative stop, a stock can drop 20%+ in a single session before the cumulative stop sees the damage. The single-day gate fires first.

- **BULL_CALM**: enabled at 10% — meaningful because the cumulative stop is wide at 15%
- **BULL_VOLATILE, CHOPPY, BEAR**: disabled (0.0) — 5% cumulative stop is already tight

Logic is identical in all three components:
- LEAN: `(prev_close − today_close) / prev_close >= sdl_pct`; `_prev_closes` dict updated each bar after the sell loop
- Notebook: same formula using `ohlcv[ticker]["close"].iloc[_idx − 1]`
- Live runner: same formula using `dfs[symbol]["close"].iloc[-2]`

Exit priority order in all three (LEAN + notebook + live runner):
1. Trailing stop (BULL_CALM: 20% trigger, 18% trail)
2. Cumulative stop-loss (15% BULL_CALM, 5% others)
3. **Single-day loss gate** (10% BULL_CALM, disabled others) ← new
4. Max hold (500 days most regimes, 23 days CHOPPY)
5. Model sell streak (3 consecutive signals, min_hold gating)

### Oversize Fallback for High-Priced Stocks
If a stock's price exceeds the budget that would be allocated under the normal `max_position_pct × confidence` formula (e.g. LLY at $926 with a $752 budget computed from 15% × confidence), the live runner retries at up to 25% of portfolio value. Falls back only when 1 share fits within 25% of portfolio; otherwise the symbol is skipped and a warning is logged (`[oversize fallback: 25%]`).

- **Normal path**: `invest = min(cash - reserve, portfolio × max_pos_pct × confidence)`
- **Fallback trigger**: `shares = invest / price == 0` AND `price <= portfolio × 25%`
- **Fallback invest**: `min(portfolio × 25%, available_cash)`
- **Purpose**: prevents high-priced stocks from being silently excluded purely because their per-share cost exceeds the regime-confidence-scaled allocation

### Calibration Gap Fixes (2026-04-16)

Three structural weaknesses in the cross-model `rank_score` calibration were identified and fixed:

**1. Staleness** (`scripts/recalibrate_scores.py`): Models retrain daily but calibration curves were frozen at notebook-training time. A new script re-fits the calibration after each daily retrain and writes updated `score_calibration` back into each symbol's `policy-metadata.json`. Also computes fresh data-driven blend weights and saves them to `strategy_config.json` as `ranking.blend_weights`. Wired into `daily_103.sh` as Step 2b (non-fatal: failure falls back to prior calibration).

**2. Isotonic tail overfitting** (`kernel/scoring.py`): Isotonic regression is a piecewise step-function that can overfit on sparse tail samples. Method selection now depends on sample size: **isotonic** for n≥300, **Platt scaling** (logistic regression sigmoid — smooth and monotone) for 120≤n<300, **constant base-rate** for n<120.

**3. Arbitrary 50/50 blend**: The `0.5 × rank + 0.5 × RS` blend had no empirical basis. `recalibrate_scores.py` now fits a logistic regression on `[norm(rank_score), norm(rs_score)]` versus binary outperformance outcomes and converts the positive coefficients into blend weights. The runner reads `ranking.blend_weights` from config instead of hardcoding.

### Gap-Alignment Fixes (2026-04-14)
Six behavioral differences between notebook simulation and LEAN were identified and fixed:

1. **Trailing stop trigger (LEAN bug)**: Was using `current_gain` to check if the trigger level was crossed — stop disarmed after pullback. Fixed to use `peak_gain = (HWM − entry) / entry` so the stop stays armed once ever triggered.
2. **BEAR defensive ranking (notebook)**: Was sorting by static OOS Sharpe. Fixed to use `oos_raw_scores.loc[today]` (live model confidence), matching LEAN.
3. **Transition uncertainty window (notebook missing)**: Added 3-bar block after each CUSUM changepoint using `changepoint_dates` already computed in Cell 5.
4. **Earnings filter (notebook missing)**: Added `_is_earnings_blocked()` helper and check in candidates loop, loading `earnings-calendar.json` to match LEAN.
5. **Sell streak during min_hold (LEAN)**: Was accumulating sell streak inside the min_hold window, which could trigger an exit on exactly day 20 even if no fresh sell signals occurred. Fixed to skip the model signal check entirely during min_hold, matching notebook behavior.
6. **Q-Learning score formula (LEAN)**: Was using `Q(buy) − Q(hold)` = `q_vals[0] − q_vals[2]`. Fixed to `Q(buy) − Q(sell)` = `q_vals[0] − q_vals[1]`, matching `predict_score_bulk()` in `kernel/models.py`.

### Cross-Sectional Rotation (2026-04-20)
Held positions now compete with new candidates on the same calibrated `rank_score` every bar — the standard mainstream-quant rotation rule. Without this, a held stock with a marginal score blocks a far-better unowned candidate from ever entering the portfolio.

- **Where**: new `RotationJob` (`kernel/pipeline/job_rotation.py` + `task_rotation.py`) sits between `RankingJob` and `SelectionJob` in Phase 3. The notebook simulation cell mirrors the same logic via `kernel.rotation.find_rotation_pairs()`.
- **Pure primitive**: `kernel/rotation.py` — `find_rotation_pairs(held_scores, held_meta, candidates, today, rotation_cfg, tax_cfg) → list[RotationPair]`. Stdlib only.
- **Tax-adjusted swap margin**: `effective_swap_margin = base_margin + tax_drag(unrealized_pnl, hold_days, ST/LT rate)`. Positions within `lt_protection_days` of the long-term threshold sitting on a gain are pinned (`+inf`) — forced swap would burn the upcoming LT discount.
- **Guards**: each pair re-checked against wash-sale, sector cap, and correlation guards on the **virtual post-swap holdings set** before being emitted.
- **Output**: emits `ExitSignal(exit_type="rotation")` for the sold ticker (handled by the existing exits path) plus a sized buy order. Telemetry counter `Rotation Exits` reported in LEAN runtime stats.
- **Skip conditions**: rotation block is skipped entirely in BEAR regime (defensive branch already restricts buys), when `rotation.enabled=false`, or when there are no holdings/candidates.
- **Config** (`strategy_config.json`):
  ```json
  "rotation": {
    "enabled": true,
    "swap_margin": 0.10,
    "min_rotation_hold_days": 30,
    "lt_protection_days": 30,
    "max_rotations_per_bar": 2
  }
  ```
- **Tests**: 22 unit tests in `tests/test_kernel_units.py` (`TestTaxDrag`, `TestEffectiveSwapMargin`, `TestFindRotationPairs`); 13 paired-alignment tests in `tests/test_policy_alignment.py::TestRotationAlignment`.

> **Open issue (2026-04-20)**: `swap_margin` is in calibrated rank-score (probability) units while `tax_drag` is in fraction-of-position units — these are added as if they were the same unit. A future fix should translate both to expected-forward-return units using the calibration's `lookahead`/`threshold`. Until then, `swap_margin = 0.10` literally means "10 percentage-point edge in P(beat SPY by 3% over 5 days)".

### Unit Tests (`tests/`)
464 unit tests covering every major policy (run with `python -m pytest tests/ -v`) → **544 after kernel extraction** (80 new kernel unit tests):

- `tests/test_policy_alignment.py` — **222 paired NB/LEAN alignment tests**: 17 policy classes (TrailingStop, CumulativeStopLoss, SingleDayLoss, MaxHold, MinHold, ConsecutiveSellStreak, SPYEMA50, VelocityCrash, TransitionWindow, Earnings, TieredThresholds, CorrelationGuard, SectorGuard, WashSale, MinModelScore, CombinedRanking, PositionSizing), each with 6 `test_nb_*` + 6 `test_lean_*` + 1 cross-check. Meta-test enforces equal NB/LEAN count per class.
- `tests/test_simulation_policies.py` — end-to-end simulation tests for min_score filter, sector guard, SPY velocity/EMA50 filters, BEAR defensive buying, ranking, wash sale, consecutive sells, stop-loss, trailing stop, correlation guard, position cap
- `tests/test_lean_policies.py` — pure-Python replicas of each LEAN policy function, `predict_score_bulk()` correctness, regression tests for all gap fixes, GMM scaler parity (6 tests), live cash accounting (5 tests), below-floor model rejection (6 tests), wash-sale reconcile from prior days (6 tests), min_hold_days, wash-sale guard, SPY regime-context feature injection
- `tests/test_runner_ranking.py` — live runner model-score ranking, calibration (Platt + isotonic), tiered thresholds, regression guards; EXIT 3 max-hold enforcement; oversize-fallback 25% cap; wash-sale re-check in selection loop (48 tests)

## 17. Implementation Roadmap (Status)

### Phase 1 — Notebook (Research) ✅ Complete
1. ✅ SPY GMM training in renquant_103 notebook
2. ✅ Hurst exponent and CUSUM functions in `common/indicators/`
3. ✅ Training data: `sample_start: 2016-01-01`, `training_years: 3`
4. ✅ Regime features added to per-symbol model training
5. ✅ Lookahead changed 10 → 5 days
6. ✅ Correlation matrix computed and serialized
7. ✅ Earnings calendar via `scripts/fetch_earnings_calendar.py`
8. ✅ All symbols trained, OOS Sharpe floor 1.0, models exported
9. ✅ Fixed training cutoff (2024-01-01) for stable OOS simulation
10. ✅ Expanding-window live model refresh (last 4 years) in export cell
11. ✅ Live model score ranking (predict_score_bulk) replacing static Sharpe ranking

### Phase 2 — LEAN Execution (backtesting/renquant_103/main.py) ✅ Complete
1. ✅ Strategy scaffolded from 102
2. ✅ 3-layer regime detection implemented
3. ✅ Regime-adaptive parameter switching
4. ✅ Regime-conditional entry (momentum/capitulation/divergence/BEAR-defensive)
5. ✅ Relative strength ranking
6. ✅ Continuous model scoring via `_get_raw_model_score()`
7. ✅ Correlation-aware greedy selection
8. ✅ Earnings calendar filter
9. ✅ Trailing stop (20% trigger, 18% trail in BULL_CALM)
10. ✅ SPY velocity crash filter
11. ✅ SPY EMA50 trend gate
12. ✅ BEAR defensive buying (1 slot for GLD/TLT/XLV/XLU)
13. ✅ XGBoost model support in LEAN
14. ✅ `strategy_config.json` updated with all parameters

### Phase 3 — Validation ✅ Complete
1. ✅ `export_lean_watchlist.py --strategy renquant_103`
2. ✅ LEAN backtest: 2024-01-01 → 2026-03-26
3. ✅ Strategy outperforms SPY in OOS period
4. ✅ Regime telemetry verified in charts
5. ✅ Live trading active via three launchd agents (NYSE-holiday-aware):
   - `com.renquant.open103.plist` — 6:32 AM PT: sell-only pass using today's opening price
   - `com.renquant.preclose103.plist` — 12:44 PM PT: intraday stop-breach sell check
   - `com.renquant.daily103.plist` — 1:55 PM PT: retrain + full buy+sell pass after close
6. ✅ 564 unit tests passing — 562 passed + 2 skipped (`python -m pytest tests/ -v`)

### Phase 4 — Pipeline Re-architecture ✅ Complete
1. ✅ Strategy kernel extracted to `backtesting/renquant_103/kernel/` (9 self-contained modules, zero `common/` imports)
2. ✅ `InferencePipeline` (7 jobs) + `SellOnlyPipeline` + `TrainingPipeline` (per-ticker parallel)
3. ✅ `LeanAdapter` + `RunnerAdapter` bridge LEAN/broker state to `InferenceContext`
4. ✅ `main.py` slimmed to ~200 lines; `live/runner.py` uses `RunnerAdapter + InferencePipeline`
5. ✅ 114 kernel unit tests + 14 pipeline tests; 562 total passing (+ 2 skipped)
6. ✅ Flat `kernel/pipeline/` layout — `pp_inference.py`, `pp_training.py`, `job_*.py`, `task_*.py` at the same level so pipelines share jobs and jobs share tasks

---

## 18. Pipeline Architecture (renquant_103)

renquant_103 uses two parallel pipelines sharing the same 3-phase pattern: **global sequential → per-ticker parallel → global sequential**. Full spec: [`doc/pipeline_design.md`](pipeline_design.md).

### Inference Pipeline

Used by both LEAN (`main.py` via `LeanAdapter`) and the live runner (`live/runner.py` via `RunnerAdapter`).

```
LeanAdapter.make_context(data)       RunnerAdapter.make_context()
                       ↓
            InferenceContext (~50 fields)
                       ↓
         InferencePipeline.run(ctx)
┌─────────────────────────────────────────────────────────┐
│  Phase 1: Global sequential                             │
│    RegimeJob    → ctx.regime, ctx.confidence            │
│    DrawdownJob  → ctx.hwm, ctx.skip_buys                │
│    BuyGatesJob  → ctx.buy_blocked, ctx.bear_only        │
│                                                         │
│  Phase 2a: Parallel (ThreadPoolExecutor, per held)      │
│    TickerSellJob[AAPL], TickerSellJob[GOOG], ...        │
│    → collect ctx.exits                                  │
│                                                         │
│  Phase 2b: Parallel (per candidate)                     │
│    TickerCandidateJob[AMD], TickerCandidateJob[CAT], …  │
│    → collect ctx.candidates                             │
│                                                         │
│  Phase 3: Global sequential                             │
│    RankingJob   → ctx.ranked                            │
│    SelectionJob → ctx.orders                            │
└─────────────────────────────────────────────────────────┘
                       ↓
LeanAdapter.commit(ctx)              RunnerAdapter.commit(ctx)
(Liquidate/SetHoldings)              (broker.place_order + live_state.json)
```

`SellOnlyPipeline` (intraday): Phase 1 (Regime → Drawdown) + parallel `TickerSellJob` only.

**File map** (`kernel/pipeline/` — flat layout: `pp_*` orchestrators, `job_*` jobs, `task_*` atomic steps at the same level):

| File | Contents |
|------|----------|
| `context.py` | `InferenceContext`, `TickerInferenceContext` |
| `pipeline.py` | `Task`, `Job`, `TickerJob` ABCs + `run_parallel()` |
| `pp_inference.py` | `InferencePipeline`, `SellOnlyPipeline` (+ ticker-context builders) |
| `pp_training.py` | `TrainingPipeline` + all training jobs/tasks |
| `job_regime.py` | `RegimeJob` |
| `job_drawdown.py` | `DrawdownJob` |
| `job_gates.py` | `BuyGatesJob` |
| `job_sell.py` | `TickerSellJob` (per-ticker, runs `compute_exits()`) |
| `job_candidates.py` | `TickerCandidateJob` (per-ticker, scores + RS) |
| `job_ranking.py` | `RankingJob` |
| `job_selection.py` | `SelectionJob` |
| `task_*.py` | Atomic tasks per concern (regime, drawdown, gates, sell, candidates, ranking, selection) |

**Adapters** (`adapters/`): `lean.py` → `LeanAdapter`; `runner.py` → `RunnerAdapter`. Both are isolated from `kernel/` isolation rules (can import broker libs).

**LEAN `main.py`**: ~200 lines. `OnData` = `make_context → pipeline.run → commit → plot`.

### Training Pipeline

Runs in the notebook and daily automation. Parallel per-ticker phase covers the expensive work.

```
TrainingContext (global)
        ↓
Phase 1 (global): DataFetchJob → RegimeFitJob
        ↓
Phase 2 (parallel, ThreadPoolExecutor — one thread per ticker):
  TickerFeatureJob → TickerTournamentJob → TickerExportJob → TickerCalibrationJob
  [orchestrated by FeatureJob.run(ctx) which calls run_ticker_parallel()]
        ↓
Phase 3 (global): CorrelationJob
```

All four per-ticker stages run in sequence **within each ticker's own thread** — tickers are fully independent. Results are collected into `TrainingContext` after all threads complete.

`TournamentJob`, `ExportJob`, `CalibrationJob` are kept as skip-if-populated no-ops for backward notebook cell compatibility. Calling them after `FeatureJob` is a no-op.

**Logging**: every worker logs `[TICKER|thread-name] job START/DONE elapsed=Xs` for traceable async output.

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
| 8 | Combined rank weights | **Data-driven** (default 50/50) | Initially equal weight; now automatically computed by `recalibrate_scores.py` via Pearson correlation of each signal vs actual outperformance outcomes across all watchlist symbols |
