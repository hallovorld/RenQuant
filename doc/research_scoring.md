# Research: Per-Stock Modeling & Calibrated Scoring

Written 2026-04-20. Revised 2026-04-20 after architectural review.

This doc evaluates the current renquant_103 modelling + scoring stack and researches
whether we can do meaningfully better. The `rank_score` this stack produces is what
decides **which stock enters the portfolio, and which held position gets rotated out**,
so its quality caps the whole strategy's ceiling.

---

## TL;DR

1. The current per-stock tournament + per-stock calibration is a **data-starved,
   fragmented setup** — each of ~30 models is trained on 500–1000 rows, and the
   calibration that makes scores cross-comparable is fitted on only the stock's
   own ~500 OOS rows. Extreme tickers fall back to a constant probability.
2. Nothing fancy is used. Calibration is **isotonic → Platt → constant base-rate**
   by sample size. Expected-return regression is **isotonic → OLS → constant**.
   All per-stock, no cross-learning. Separately, 5-day labels overlap ~80%
   day-over-day and we don't correct for it in either training or evaluation —
   the effective sample size is N/5, not N.
3. The biggest levers, ranked by expected lift vs. engineering cost:
   - **(A) Cross-sectional panel model with LTR objective** — pool all tickers,
     train with `rank:pairwise` / `lambdarank`, group=date. Solves data starvation
     *and* makes scores cross-comparable by construction.
   - **(B) Sample weighting by label concurrency** (AFML ch.4) — paired with (A),
     prevents the tree from treating 5 overlapping rows as 5 independent samples.
   - **(C) Beta-neutral, cross-sectionally Gaussianized labels** — residualize
     forward return against SPY + sector ETF, then rank-normalize per date and
     apply inverse-normal CDF. Kills the arbitrary 3%/5d threshold and removes
     outlier chasing.
   - **(D) Partial feature neutralization** (Numerai-style) on momentum/trend
     features only. Forces the panel model to find idiosyncratic alpha rather
     than collapse into a beta-tracker.
   - **(E) Cross-sectional factor features** (size, value, quality, 12-1
     momentum) — fundamental anchors that pure technicals lack.
   - **(F) Purged K-fold CV with embargo** — replaces the current single-split
     holdout with a distribution of OOS metrics.
   - **(G) NGBoost for uncertainty** — `score = μ − λσ`; use σ as a sizing
     multiplier, not as input to a Markowitz optimizer.
4. Concrete recommendation: **Stage 1 ships (A)+(B)+(C)+(D)+(E)+(F) as a single
   architectural shift**, trained **weekly** with daily calibration/inference.
   Stacking meta-learners move to Stage 3 (or get killed if the panel + factors
   already saturate). Tabular Q-learning is removed entirely. TFT / graph models
   stay out of scope until the panel LTR baseline plateaus.

---

## 1. What the current system actually does

### 1.1 Per-stock pipeline

For each ticker in the 37-stock watchlist:

1. **Feature frame** (`training/features.py::build_training_features`):
   - 7 technical indicators relative to SPY: `rsi`, `adx` as ratios;
     `macd_hist`, `cci`, `bbp`, `williams_r`, `obv_slope` as diffs.
   - Relative price `close = stock_close / spy_close × 100`.
   - Trend features: `trend = close/ema50`, `trend_long = close/ema200`.
   - Relative momentum: `rel_mom_20d`, `rel_mom_60d`.
   - SPY regime context (rolling per bar): `spy_realized_vol`, `spy_adx`, `spy_trend`, `hurst_proxy`.
   - Label: sign of `stock_fwd_return − spy_fwd_return` over 5 days, thresholded at 3% → {−1, 0, +1}.

   **Important mismatch**: `feature_columns` in `strategy_config.json` currently uses only
   11 of these (7 technicals + 4 SPY context). `trend`, `trend_long`, `rel_mom_20d`,
   `rel_mom_60d` are computed and written to the frame but **never trained on**. Dead code.

2. **Tournament** (`training/tournament.py::run_tournament`):
   - Train 4 algorithms on data before cutoff (now rolling `today − 2y`):
     - **Classification** — BagLearner(RTLearner) ×15 bags, leaf_size=25, raw score = BagLearner mean.
     - **QLearning** — tabular Q over discretized first 5 features, raw score = Q(buy) − Q(sell).
     - **Manual** — hand-coded vote on indicator thresholds, raw score = vote count.
     - **XGBoost** — 200 trees, depth 4, two classifiers (buy-vs-rest + sell-vs-rest), raw score = P(buy) − P(sell).
   - Evaluate each on OOS by annualised **long-only Sharpe** of its signals.
   - Pick the winner.

3. **Retrain-on-full-window** (`training/export.py::retrain_live_models`):
   - Rebuild the chosen algorithm and train on the last 4 years of features.
   - This is what LEAN/live load. (As of yesterday, also has walk-forward holdout Sharpe.)

4. **Calibration** (`training/scoring.py::fit_probability_calibration`):
   - Fit `raw_score → P(stock outperforms SPY by ≥ 3% over next 5 days)` on OOS rows of that stock.
   - Method by sample size: **n ≥ 300 → isotonic**, **120 ≤ n < 300 → Platt (logistic on standardised raw)**,
     **n < 120 → constant base-rate fallback**.
   - Also fit `fit_expected_return_calibration`: `raw_score → E[stock_return − SPY_return]` on same OOS
     — isotonic / linear OLS / constant. Used by `RotationJob` to compare candidate ER vs held ER.

### 1.2 How scores are used downstream

- **Scan phase** (`task_candidates.py`): each watchlist stock with a model passes through
  `extract_raw_score` → `calibration.calibrate(raw) → rank_score ∈ [0, 1]`.
  A separate `rs_score = stock_20d_return − sector_ETF_20d_return` is computed.

- **Ranking** (`task_ranking.py::BlendScoresTask`): each bar, the candidate list's
  `rank_score` and `rs_score` are min-max normalised across the current bar's
  candidates, then blended: `score = w_rank·norm(rank_score) + w_rs·norm(rs_score)`.
  `blend_weights` is fitted daily by `scripts/recalibrate_scores.py` via logistic
  regression on `[norm(rank_score), norm(rs_score)]` → positive-coef normalisation.
  **Current weights: `[1.0, 0.0]`** — `rs_score` contributes nothing in production.

- **Selection** (`kernel/selection.py::run_selection_loop`): tiered-threshold greedy
  slot-filling on `rank_score` (not blend) — slot 1 ≥ 0.10, slot 2 ≥ 0.30, slot 3 ≥ 0.50.
  Wash-sale, sector, correlation guards applied per slot.

- **Rotation** (`kernel/rotation.py::find_rotation_pairs`): uses the `expected_return`
  head directly (not `rank_score`). Net advantage = candidate ER − held ER − tax drag −
  txn cost; swap when advantage ≥ `min_expected_advantage_pct`. LT-protection pins
  positions near the 1-year holding mark.

### 1.3 Is anything fancy used?

**No.** It's textbook supervised ML with conservative defaults:

- Labels: binary forward-return threshold (3% in 5d).
- Calibration: isotonic / Platt — standard sklearn.
- Model selection: single-fold holdout Sharpe, argmax across 4 candidates.
- Expected-return head: OLS or isotonic.
- Features: hand-picked technical indicators.
- No uncertainty quantification, no multi-task learning, no cross-sectional structure,
  no fundamentals, no alternative data.

This is a reasonable v1 — but the ceiling is low.

---

## 2. Concrete weaknesses

These are grouped by impact. Each weakness names the file/line where the behaviour
lives so we can point at it when discussing fixes.

### 2.1 Data starvation (high impact)

- Per-stock models see **~500–1000 rows** each (4-year train window, daily bars).
- XGBoost with 200 trees on 500 rows of 11 features is severely overfit-prone.
  `n_estimators=200, max_depth=4` is a huge hypothesis class for that much data.
- Calibration's `isotonic` path requires n ≥ 300, `Platt` requires n ≥ 120. Tickers
  recently added to the watchlist (AVGO, KLAC, LRCX, MRVL, TXN based on git status)
  likely fall into Platt or even the constant-base-rate bucket on their first few
  runs — meaning their `rank_score` is **literally a constant** across all bars.
- QLearning tabular Q-tables are the worst offenders: discretized state space needs
  tens of thousands of transitions to converge.

### 2.2 No cross-learning (high impact)

- AMZN and META share a huge amount of structural behaviour (tech sector, beta to
  NVDA/QQQ, same macro sensitivities). Both models learn this independently —
  wasteful, and the low-sample model can't benefit from the high-sample one.
- Sector peers (CAT / GE / RTX / LMT in industrials; JPM / MA in financials) show
  the same problem.
- No mechanism for newly-added tickers (e.g. AVGO) to benefit from patterns learned
  on structurally similar names (NVDA).

### 2.3 Calibration fragility (medium–high impact)

- Each model's raw-score distribution is different. The calibration mapping is what
  puts them on a common `rank_score ∈ [0, 1]` scale. But:
  - Each calibrator is fit on ~500 OOS rows of **one stock** — small, noisy.
  - OOS window is the last 2y (after the rolling cutoff change), so the calibrator
    reflects one recent regime only; on a regime shift it mis-maps.
  - Two stocks with calibrated `rank_score = 0.60`: we treat these as comparable,
    but one might come from a well-calibrated isotonic fit (say AAPL with
    n=600 OOS) and the other from a constant-prob fallback (new tickers) —
    numerically equal, informationally not.

### 2.4 Wrong model-selection criterion (medium impact)

- Tournament picks winner by **long-only Sharpe** (`oos_sharpe` in tournament.py).
- But in production, the score is used for **cross-sectional ranking**. These are
  different objectives:
  - Sharpe is path-dependent (sequence of returns matters).
  - Ranking needs the score to correctly order candidates on any given bar.
- The right model-selection metric for ranking is the **Information Coefficient (IC)**
  = Spearman(score, forward_return) on each bar, averaged over time. It's what the
  score is actually doing; it's much less noisy than Sharpe of a single-stock
  long-only strategy.
- Picking the wrong model on ~500 OOS rows is very likely — Sharpe is noisy, and
  the four algorithms often land within 0.2 of each other.

### 2.5 Dead features in config (low impact but silly)

`features.py` builds `trend`, `trend_long`, `rel_mom_20d`, `rel_mom_60d` and SPY
context columns, but only 11 of them appear in `feature_columns` in `strategy_config.json`.
At least `trend`, `trend_long`, `rel_mom_20d`, `rel_mom_60d` are computed, stored,
and thrown away at training time. Either wire them in or remove the code.

### 2.6 No uncertainty estimate (medium impact)

- `rank_score` is a point estimate. A 0.60 from a tight isotonic fit on 600
  well-behaved samples is not the same as a 0.60 from a Platt fit with
  ±0.10 confidence.
- Without uncertainty, we can't:
  - Scale position size by score confidence.
  - Down-weight new tickers until we have enough data.
  - Reject candidates whose score is within the noise floor of the next candidate.

### 2.7 Binary label + arbitrary threshold (medium impact)

- `threshold = 0.03` over 5 days means different things for LLY (vol ~30%) vs
  XLU (vol ~12%). LLY clears 3% on noise alone; XLU rarely clears it.
- Label distributions are skewed by ticker volatility, which means the per-stock
  model learns a different base rate of `label=1` for each ticker.
- Beyond volatility normalisation, the label also contains raw market beta (a
  bull-market 3% move is common for everything); without neutralizing SPY /
  sector beta, the model learns to predict market direction instead of
  idiosyncratic outperformance.

### 2.8 No proper cross-validation (medium impact)

- Single fixed cutoff (now rolling 2y). Any single-fold metric has massive variance.
- Overlapping labels (each bar's label looks 5 days forward) leak across train/OOS
  boundary unless you purge + embargo — current code does neither.
- López de Prado's **Purged K-Fold CV with embargo** is the industry-standard
  remedy; the recent trend is **Combinatorial Purged CV** which gives a full
  distribution of OOS Sharpes instead of a single noisy number.

### 2.9 The rs_score channel is dead (low impact)

Current `blend_weights = [1.0, 0.0]`. `rs_score` (20d stock return minus sector ETF
return) is computed but multiplied by zero in ranking. The recalibration script keeps
finding it un-useful, which suggests it's a noisy duplicate of information already in
the model score (`rel_mom_20d` was in features.py but not in feature_columns, and the
relative-price target of the classifier already captures short-term relative perf).

### 2.10 Overlapping-label training bias (high impact — missed in v1 of this doc)

- 5-day forward-return labels overlap 80% between consecutive bars.
- Trees don't know that `label_t` and `label_{t+1}` carry the same information —
  they treat every row as independent. Effective N is N/5, not N.
- This isn't just a CV concern (2.8). It biases **training**: splits that
  happen to land on overlap regions look disproportionately "signal-rich", which
  is why XGBoost + 200 trees + 500 rows is so prone to spurious structure.
- Fix: weight each row inversely proportional to label concurrency on that date
  (AFML ch.4). XGBoost and LightGBM both accept `sample_weight`. Cheap.

### 2.11 Feature-level beta leakage (medium impact)

- `rel_mom_20d`, `rel_mom_60d`, `trend`, `trend_long` all load on market/sector
  beta. A panel LTR model handed these features will happily become a glorified
  beta-tracker: the strongest in-sample signal is "stock followed SPY up the
  most in the last 20 days, so rank it highest", which is zero alpha.
- Mean-reversion features (RSI, BBP, Williams %R) don't have this problem — they
  measure distance from equilibrium, which is already cross-sectional.
- Fix: partial neutralization (Numerai-style) — regress momentum features
  against sector momentum, feed residuals to the model. Only neutralize the
  features that actually contain beta; leave mean-reversion indicators alone.

---

## 3. Better approaches

Ordered roughly by impact-per-complexity, with short descriptions and code-level
sketches where relevant.

### 3.1 Cross-sectional panel model with Learning-to-Rank (highest impact)

**What.** Replace 30+ per-stock models with **one** model trained on all tickers at
once. Input features include everything we already compute *plus* a ticker identifier
(hash, target encoding, or embedding). Use an LTR objective (`rank:pairwise` or
`lambdarank` in XGBoost/LightGBM) with `group_id = bar_date` so the model learns to
rank stocks against each other on the same day.

**Why this is the headline change.**
- Training data goes from ~500 rows to **~500 × N_tickers ≈ 15000–20000 rows**.
  Overfitting risk drops orders of magnitude.
- The LTR objective directly optimises what we use the score for — cross-sectional
  ranking on each bar.
- `rank_score` becomes **comparable across stocks by construction**. No per-stock
  calibration needed for ranking (you still need calibration if you want a
  probability, but for selection/rotation you just need the order).
- Adding a new ticker costs zero training data problem. It inherits the shared
  parameters (subject to the minimum-history gate in 3.12).
- Standard Kaggle stack for finance competitions (Jane Street, Numerai, the Two
  Sigma problems). Mean-IC of 0.10–0.15 is plausible; papers report Sharpe 1.5–2.0
  net-of-cost on US equities with this approach
  ([Building Cross-Sectional Systematic Strategies By Learning to Rank, 2020](https://arxiv.org/pdf/2012.07149)).
- Recent production work shows a LightGBM panel model hitting
  [annualised return 31.4%, Sharpe 2.08 with IC ~0.153](https://arxiv.org/html/2507.07107)
  — same ballpark of watchlist size as ours.

**Sketch**:

```python
# training/panel_model.py (new)
def build_panel_frame(feature_frames, watchlist):
    frames = []
    for ticker, df in feature_frames.items():
        d = df.copy()
        d["ticker"] = ticker
        d["sector"] = SECTOR_MAP[ticker]
        d["date"] = d.index
        frames.append(d)
    panel = pd.concat(frames).sort_values(["date", "ticker"])
    # per-date group for LTR
    group_sizes = panel.groupby("date").size().values
    return panel, group_sizes

def train_panel_ltr(panel, group_sizes, feature_cols, label_col="label",
                    sample_weight_col="w"):
    import xgboost as xgb
    dtrain = xgb.DMatrix(
        panel[feature_cols],
        label=panel[label_col],
        weight=panel[sample_weight_col],   # from 3.9
    )
    dtrain.set_group(group_sizes)
    model = xgb.train(
        params={
            "objective": "rank:pairwise",   # or "rank:ndcg" / "rank:map"
            "eta": 0.05, "max_depth": 6,
            "lambda": 1.0, "alpha": 0.5,    # strong L1/L2 for panel data
            "nthread": -1,
        },
        dtrain=dtrain,
        num_boost_round=400,
        verbose_eval=50,
    )
    return model
```

**Inference**: on each bar, build one DMatrix of all candidates, score, sort
descending. No per-stock calibration required for ordering; optionally apply one
**global** calibrator fitted on all OOS rows (10× more data → isotonic fit much
tighter).

**Effort**: 2–3 days for the core, but Stage 1 bundles this with 3.9–3.12 so
budget 5–6 days total.

### 3.2 Purged K-fold CV with embargo (high impact on model selection)

**What.** Instead of one train/OOS split, run K folds (e.g. K=5) of 20%-OOS each,
with:
- **Purge**: remove training rows whose 5-day forward label overlaps any test row.
- **Embargo**: after each test fold, skip a few bars before the next train fold
  (guards against post-event leakage).

Reports the *distribution* of OOS Sharpe/IC per model, not a single noisy number
([Lopez de Prado, AFML ch.7](https://reasonabledeviations.com/notes/adv_fin_ml/);
[skfolio CPCV impl](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html)).

**Why valuable.** Model selection, hyperparameter tuning, and
"is this actually improving?" decisions all become much less noisy.

**Effort**: 1 day. Plug `skfolio.model_selection.CombinatorialPurgedCV` into the
panel training loop, mean-IC as the selection score.

### 3.3 Stacking meta-learner (demoted — Stage 3, if at all)

**What.** Feed multiple base-model predictions into a meta-learner.

**Why demoted from v1.** Stacking correlated base models (4 trees on the same
feature set) mostly memorizes noise and adds IC lift of maybe ~5% while doubling
the inference complexity. The real gains come from structurally orthogonal bases
(microstructure, fundamentals-only tree, sentiment) — which we don't have yet.
Revisit only if Stage 1 plateaus and we have a genuinely orthogonal second model.

### 3.4 Distributional output via NGBoost + σ-aware scoring (medium impact, Stage 2)

**What.** [NGBoost](https://arxiv.org/abs/1910.03225) outputs the full parameters of
a distribution (e.g. Normal(μ, σ)) over future returns. You get:
- `E[R - SPY]` — what rotation needs, no separate ER regression.
- `Var[R - SPY]` — uncertainty, unavailable today.
- P(R > threshold) — analytically computed from the distribution, no calibration.

**How to use σ** — and specifically, **what not to do**:
- **Ranking/selection score**: `score = μ − λσ`, `λ ∈ [0.5, 1.5]`. Replaces
  calibrated `rank_score`.
- **Position sizing**: scale existing confidence-based `max_position_pct` by
  `σ_p50 / σ_i`. High-uncertainty candidates get smaller allocations.
- **Keep the greedy selector + correlation guard.** Don't build a
  Markowitz / Black-Litterman optimizer for N=8 concurrent positions.
  At that scale, mean-variance optimization chases corner solutions; the
  covariance matrix is rank-deficient without aggressive regularization, and
  once you regularize enough to get diversification, you've approximated what
  the correlation guard already does. σ-in-score captures ~80% of the
  uncertainty benefit for ~10% of the complexity.

**Effort**: 2 days. New `NGBoostPanelModel` class (supports the same API as
XGBoost). Persistence via sklearn pickle or JSON schema. Test that the calibrated
probability recovers correct base rates.

**Caveat**: NGBoost doesn't support rank objectives natively. If we go with 3.1
(LTR), we keep LightGBM/XGBoost for ranking and use NGBoost only for the
uncertainty/ER head. Two artifacts per panel, same features.

### 3.5 Cross-sectional factor features (promoted to Stage 1)

**What.** Add per-date cross-sectional z-scores of:
- Size (log market cap)
- Value (earnings yield, book-to-price)
- Quality (ROE, gross profitability)
- Momentum (12m - 1m return)
- Short-interest (if available)
- Beta to SPY (rolling 60d)
- Residual momentum (momentum orthogonal to SPY)

**Why promoted.** LTR models *thrive* on cross-sectional comparisons. Technical
indicators alone, even in a panel, are weak without fundamental anchors — they
tell you what has happened to the price, not what the business is. The
Fama-French / Carhart factor literature is clear these exposures have persistent
risk premia, and at daily frequency stocks still trade on their factor
exposures cross-sectionally.

**Effort**: 1–2 days. yfinance provides size + beta + 12-1 momentum. Fundamentals
(earnings yield, quality) need OpenBB or quarterly statement caching. Even
without full fundamentals, the technical factors alone (size, momentum, beta,
residual momentum) are worth adding.

### 3.6 Beta-neutral, cross-sectionally Gaussianized labels (Stage 1)

**What.** Pipeline:
1. Compute `fwd_return_5d` per (ticker, date).
2. Residualize against rolling 60d regression on `[SPY_return, sector_ETF_return]`.
   **Purged** — the regression uses only data strictly before the current bar.
3. Rank each date's cross-section of residual returns → `[0, 1]`.
4. Apply inverse-normal CDF → final label `~ N(0, 1)` per date.

**Why.** Two issues fixed at once:
- Beta noise removed — the model can't trivially win by predicting market
  direction (which is what the panel LTR would otherwise latch onto).
- Heavy-tailed per-ticker returns are replaced by a Gaussian cross-section,
  which is what tree-based LTR objectives consume best. No more threshold
  tuning (3%/5d is arbitrary); the label is continuous and orderable.

**Effort**: ~1 day, mostly care in purging the rolling beta regression (easy to
leak current-bar data if you're not careful). **Supersedes** the simpler
volatility-normalised-label idea from an earlier draft.

### 3.7 Learning-to-rank objective, even without panel model (backup option)

If (3.1) is delayed, a cheap halfway step: keep per-stock models but switch the
tournament's evaluation metric from Sharpe to **cross-sectional IC averaged over
OOS bars**. Better model selection without touching the training code.

**Effort**: 2 hours. Change `oos_sharpe` in tournament.py to an IC computation
across tickers per bar, averaged.

### 3.8 Graph / relational models (TFT-GNN, etc.) — out of scope

Recent papers (e.g. [TFT-GNN 2025](https://www.mdpi.com/2673-9909/5/4/176),
[Multi-Sensor TFT 2025](https://www.mdpi.com/1424-8220/25/3/976)) report meaningful
lifts from attention models that explicitly represent asset relationships. These
are 5–10× the engineering effort of (3.1) and require GPU or patient training.
Revisit only after the panel LTR baseline is solid.

### 3.9 Sample weighting for overlapping labels (Stage 1, paired with 3.1)

**What.** Weight each training row inversely proportional to the number of
concurrent active labels on that date:
`w_i = 1 / mean(concurrency_i over bars where label_i is active)`, where
`concurrency_i = count of labels whose 5-day forward window includes date i`.

**Why.** Closes the training-time overlapping-label bias from 2.10. Without it,
a panel model with 5-day labels across ~30 tickers treats 5 overlapping
rows per bar per ticker as 5 independent samples, which massively over-counts the
available information and drives overfitting.

**Effort**: < 1 day. Precompute a `weight` column in the panel frame;
XGBoost/LightGBM accept `sample_weight` in the DMatrix.

### 3.10 Partial feature neutralization (Stage 1)

**What.** For each date, regress each of `{rel_mom_20d, rel_mom_60d, trend,
trend_long}` against sector-momentum features. Feed the **residuals** into the
model. Keep mean-reversion indicators (RSI, BBP, Williams %R) un-neutralized —
those signals *are* distance-from-equilibrium, and projecting out sector beta
destroys what makes them alpha.

**Why.** Tree ensembles naturally over-weight momentum / beta exposures because
those are the most predictive features in-sample. Neutralizing forces the model
to find idiosyncratic alpha. Numerai-style neutralization reports ~10–20% IC
lift on crowded factor sets.

**Effort**: < 1 day. Rolling linear regression per feature, per bar, computed
once at panel construction.

### 3.11 Handling newly-listed tickers in the panel (Stage 1 design pattern)

The panel model collapses the v1 cold-start problem but introduces a new one:
how do we handle tickers whose history is shorter than the longest lookback
window (e.g. 252d for 12-1 momentum)?

**Layered defence:**

1. **Minimum-history gate** (hard floor): a ticker enters the panel only after
   252 trading days of OHLCV. Below that, it stays out of the candidate pool
   entirely — no buys. Longest-horizon factors are unreliable below that.
2. **Missingness indicators**: for each feature that can be NaN on young
   tickers, emit a binary `{feature}_is_missing` column. Trees learn the
   "new-ticker" pattern as its own regime.
3. **Sector-median imputation for fundamentals**: for missing factor features
   (P/E, earnings yield, quality), fill with the date's sector-bucket median
   — not global mean/median. A missing P/E on a live ticker almost always
   means "unreleased yet, probably close to peers", not zero.
4. **NaN-native tree defaults**: XGBoost and LightGBM learn optimal default
   split directions for NaN; we don't pre-impute technicals at all.
5. **Sample-weight decay on young history**: `w_early = min(1, history_days/504)`,
   stacked multiplicatively on top of 3.9's concurrency weight. Prevents
   "every IPO looks like AVGO in year 1" bias.
6. **Expanding-window neutralization warmup**: for the first year past the 252d
   gate, the neutralization regression (3.10) uses an expanding window from
   listing date, not a rolling 252d — a ticker newly past the gate has zero
   lookback for its own residuals.

**What to avoid.**
- Cross-sectional rank imputation on the *feature itself* (e.g. fill missing
  feature with 0.5 after rank-normalization) — creates artificial clustering
  at the median that trees latch onto as a spurious signal.
- Hierarchical Bayesian partial pooling — statistically cleanest approach, but
  the implementation and debugging cost dwarfs the expected lift at our scale.

**Effort**: ~1 day bundled with panel feature construction.

---

## 4. What I'd ship, in order

**Stage 1 — "panel rewrite"** (~5–6 days, single atomic migration):

Ship all of the following together. They're mutually reinforcing; skipping one
weakens the rest.

1. **Cross-sectional panel LTR model** (3.1) replacing per-stock tournament.
   `xgb.rank:pairwise`, `group_id = date`. Single JSON artifact.
2. **Sample weighting by label concurrency** (3.9). Precomputed in the panel frame.
3. **Beta-neutral, cross-sectionally Gaussianized labels** (3.6). Purged rolling
   beta regression + rank + inverse-normal CDF.
4. **Partial feature neutralization** (3.10) on momentum/trend features only.
5. **Cross-sectional factor features** (3.5). At minimum: size, 12-1 momentum,
   rolling 60d beta, residual momentum. Fundamentals (P/E, ROE) if OpenBB is
   already plumbed, otherwise deferred to 3.5 follow-up.
6. **Newly-listed ticker imputation** (3.11) — 252d gate, missingness indicators,
   sector-median fill, sample-weight decay.
7. **Purged K-fold CV with embargo** (3.2) for hyperparameter tuning and
   "is this better?" decisions. Use IC (not Sharpe) as the selection metric.
8. **Retraining cadence change**: **weekly** full panel refit (Sunday cron),
   **daily** score calibration + blend-weight recalibration. Frozen weights
   between weekly refits. Daily panel refits create excessive turnover and
   waste compute.

Keep the per-stock tournament code path alongside as a `ranking.model_type`
config flag so we can A/B — but flip the live default to the panel model once
OOS IC beats the per-stock baseline for 4 consecutive weekly rebuilds.

**Stage 2 — "uncertainty head"** (~2–3 days):
9. **NGBoost panel model for μ, σ** (3.4). Second artifact, same features.
10. **Replace calibrated `rank_score` with `μ − λσ`** for selection.
11. **σ-based position-size multiplier** inside the sizing step.

Keep the greedy selector + correlation guard. No Markowitz, no Black-Litterman.

**Stage 3 — "alpha expansion + optional stacking"** (timeline open):
12. **Full fundamental factor set** — quality, value, short interest, analyst
    revisions (3.5 follow-up). Requires OpenBB or similar data plumbing.
13. **Orthogonal second model** — microstructure (intraday), sentiment
    (news/SEC filings), or fundamentals-only tree. Only *after* we have one,
    consider stacking (3.3). Stacking four trees on the same feature set is
    theater.
14. **(Speculative) TFT / graph model** (3.8) if Stage 1+2+3 plateaus.

---

## 5. What *not* to do

- **Don't add more model types to the tournament.** Five/six isn't better than
  four — with ~500 rows each, the variance in model selection swamps the diversity
  gain. Fix the data problem first (3.1).
- **Don't keep tabular Q-learning** after the panel migration. Discretizing the
  state space is a massive information loss; with the panel fixing data
  starvation, the Q-table's value proposition is gone. Remove the code path.
- **Don't build a Markowitz / Black-Litterman optimizer for N=8 positions.**
  Covariance estimation is rank-deficient at that scale; unconstrained MV
  concentrates into 2–3 names; constrained MV approximates what the
  correlation guard already does. Use NGBoost σ in the scoring function and
  the sizing multiplier instead.
- **Don't stack the existing four models.** They share a feature set; a meta-learner
  will just memorize noise. Stacking only pays off when the base models are
  structurally orthogonal.
- **Don't skip the sample weighting** in Stage 1. It's the cheapest piece and
  closes the biggest methodological hole (2.10). Panel LTR without concurrency
  weights is a panel LTR overfit to 80%-overlap redundancy.
- **Don't refit the panel daily.** Weekly refits with frozen weights between
  refits is the industry standard. Daily panel refits waste compute and produce
  noise-driven turnover.
- **Don't hand-tune `buy_threshold`/`sell_threshold` per stock.** These live in
  `strategy_config.json → model_params` and are shared. If we want per-stock
  thresholds, derive them from score quantiles on calibration data, not by hand.
  (Better: 3.6 Gaussianized labels remove the need for thresholds entirely.)
- **Don't keep the `rs_score` blend channel "just in case".** It's zero-weighted
  by the daily recalibrator — evidence it's redundant. Either remove it to
  simplify, or replace it with something genuinely orthogonal (short-interest
  change, analyst revisions, residualised industry-relative momentum).

---

## 6. Resolved design decisions (post architectural review)

These were open questions in the v1 draft. Architectural-review discussion has
closed them.

| Question | Decision | Reason |
|----------|----------|--------|
| Model architecture | Cross-sectional panel LTR (XGBoost/LightGBM) | Data starvation → panel; ranking objective → LTR |
| Model-selection metric | Mean-IC (not Sharpe) | Ranking needs cross-sectional IC |
| Label design | Beta-neutral + cross-sectionally Gaussianized | Kills threshold arbitrariness + beta noise |
| Training-time overlap handling | AFML sample weighting | Effective N is N/5 without it |
| Feature neutralization | Partial — momentum/trend only | Preserve mean-reversion signal |
| Uncertainty | NGBoost for μ, σ, and `score = μ − λσ` | Unifies ER + P + σ into one head |
| Portfolio optimization | Greedy + correlation guard + σ sizing multiplier | Markowitz overkill at N=8 |
| Retraining cadence | Weekly panel refit + daily calibration | Balance fit freshness against turnover |
| Q-learning | Removed | Panel renders it obsolete |
| Stacking | Deferred to Stage 3, contingent on orthogonal base models | Correlated bases = noise memorization |

## 7. Remaining open questions

1. **Fundamental data source.** Stage 1 can launch with technicals-only + size
   + beta + 12-1 momentum (all from yfinance). Moving to full fundamentals
   (earnings yield, ROE, short interest) requires OpenBB or a paid source.
   What's the budget?
2. **Compute budget.** Weekly panel refit on ~15k rows × ~20 features is ~minutes
   on CPU — no GPU needed for Stage 1+2. Stage 3 TFT/GNN would need GPU.
3. **A/B experimental setup.** Keep notebook-only side-by-side comparisons until
   the panel beats per-stock on OOS IC for 4 consecutive weekly rebuilds, then
   flip the live default via `ranking.model_type` config flag?
4. **Factor universe for the neutralization regression.** Sector ETFs are the
   obvious choice (XLK, XLF, XLE, XLI, XLV, XLU, XLC, XLP, XLY, XLRE, XLB).
   Do we also neutralize against a separate size factor (IWM / SPY spread),
   or is that captured well enough by the size feature in 3.5?

---

## Sources

Cross-sectional / learning-to-rank:
- [Building Cross-Sectional Systematic Strategies By Learning to Rank (Poh et al., 2020)](https://arxiv.org/pdf/2012.07149)
- [Machine Learning Enhanced Multi-Factor Quantitative Trading: A Cross-Sectional Portfolio Optimization Approach (2025)](https://arxiv.org/html/2507.07107)
- [XGBoost Learning to Rank documentation](https://xgboost.readthedocs.io/en/latest/tutorials/learning_to_rank.html)
- [Public sentiment-based stock selection strategy using BERT + LightGBM (Wang et al., 2025)](https://journals.sagepub.com/doi/abs/10.1177/14727978251355788)
- [Identifying the best performing stocks based on their semi-correlation (arXiv 1906.08636)](https://arxiv.org/pdf/1906.08636)
- [Combined machine learning for stock selection strategy with dynamic weighting (2025)](https://arxiv.org/html/2508.18592v1)

Deep learning stock ranking:
- [A Novel Hybrid TFT-Graph Neural Network Model for Stock Market Prediction (2025)](https://www.mdpi.com/2673-9909/5/4/176)
- [Multi-Sensor Temporal Fusion Transformer for Stock Performance Prediction — Adaptive Sharpe Ratio Approach (MDPI 2025)](https://www.mdpi.com/1424-8220/25/3/976)
- [Comparing Transformer Models for Stock Selection in Quantitative Trading (SpringerLink)](https://link.springer.com/chapter/10.1007/978-3-032-00891-6_19)
- [Quantformer: from attention to profit with a quantitative trading approach (arXiv 2404.00424)](https://arxiv.org/pdf/2404.00424)

Distributional regression:
- [NGBoost: Natural Gradient Boosting for Probabilistic Prediction (Duan et al., 2020)](https://arxiv.org/abs/1910.03225)
- [NGBoost Stanford ML Group page](https://stanfordmlgroup.github.io/projects/ngboost/)
- [Forecasting Probability Distributions of Financial Returns with Deep Neural Networks (arXiv 2508.18921, 2025)](https://arxiv.org/html/2508.18921v1)

Purged cross-validation & sample weighting:
- [Advances in Financial Machine Learning (López de Prado) — ch.4 sample weighting, ch.7 purged CV](https://reasonabledeviations.com/notes/adv_fin_ml/)
- [Purged cross-validation — Wikipedia](https://en.wikipedia.org/wiki/Purged_cross-validation)
- [skfolio CombinatorialPurgedCV](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html)
- [QuantInsti — Cross Validation in Finance: Purging, Embargoing, Combinatorial](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/)

Feature neutralization:
- [Numerai — Feature Neutralization documentation](https://docs.numer.ai/numerai-tournament/models/feature-neutralization)
