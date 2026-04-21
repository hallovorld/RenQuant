# Research: Per-Stock Modeling & Calibrated Scoring

Written 2026-04-20. Draft for discussion — not a commitment to implement.

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
   by sample size. Expected-return regression (used by rotation) is **isotonic →
   OLS → constant**. All per-stock, no cross-learning.
3. The biggest levers, ranked by expected lift vs. engineering cost:
   - **(A) Cross-sectional panel model with learning-to-rank objective** — pool all
     tickers into one model; train with `rank:pairwise` / `lambdarank`, group=date.
     Solves data starvation *and* gives you cross-sectional comparability for free.
   - **(B) Purged K-fold CV with embargo** for tournament evaluation — the current
     single fixed split (now rolling 2y) is a noisy estimator.
   - **(C) Stacking meta-learner** on top of the 4 existing model heads — gains
     10–30% over best-single-model in typical Kaggle quant.
   - **(D) Distributional output** (e.g. NGBoost) so `rank_score` carries
     uncertainty, not just a point probability. Enables confidence-aware sizing.
   - **(E) Cross-sectional factor features** (size, value, momentum, quality) —
     pure alpha additions, orthogonal to technicals.
4. My concrete recommendation: **do (A) first.** It replaces the tournament, adds
   no fundamental data dependency, and makes scores cross-comparable by construction.
   Everything else can layer on top afterward.

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
- A volatility-normalised target (e.g. `forward_return / 20d_realised_vol`) would
  make labels directly comparable.

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
  parameters.
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

def train_panel_ltr(panel, group_sizes, feature_cols, label_col="label"):
    import xgboost as xgb
    dtrain = xgb.DMatrix(panel[feature_cols], label=panel[label_col])
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

**Effort**: 2–3 days of work including:
- New `build_panel_frame` + `train_panel_ltr` module.
- Inference path that loads a single artifact and scores all candidates at once
  (`kernel/panel_model.py` — lightweight json load + xgb.Booster.predict).
- Keep current per-stock tournament alongside as a baseline; add a config flag
  `ranking.model_type: "panel" | "per_stock"` to A/B.
- A few tests for group construction and rank invariance.

### 3.2 Purged K-fold CV with embargo (high impact on model selection)

**What.** Instead of one train/OOS split, run K folds (e.g. K=5) of 20%-OOS each,
with:
- **Purge**: remove training rows whose 5-day forward label overlaps any test row.
- **Embargo**: after each test fold, skip a few bars before the next train fold
  (guards against post-event leakage).

Reports the *distribution* of OOS Sharpe/IC per model, not a single noisy number
([Lopez de Prado, AFML ch.7](https://reasonabledeviations.com/notes/adv_fin_ml/);
[skfolio CPCV impl](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html)).

**Why valuable.** Tournament model selection becomes much less noisy. Currently, a
random seed change can flip the winner between Classification and XGBoost; with
5-fold purged CV averaging, the winner is more stable and reflects real skill
rather than luck on a single OOS window.

**Effort**: 1 day. Plug `skfolio.model_selection.CombinatorialPurgedCV` into
`run_tournament`, loop folds, take mean-IC or mean-Sharpe as the selection score.

### 3.3 Stacking meta-learner (medium-high impact, low effort)

**What.** Keep the 4 base models. Instead of `argmax(Sharpe)`, feed their 4
predictions (`rank_score_cls`, `rank_score_ql`, `rank_score_xgb`, `rank_score_manual`)
plus a few regime features (`regime`, `spy_20d_vol`, `hurst`) into a simple
meta-learner (LightGBM with ~50 trees, or even a logistic regression). The
meta-learner outputs the final `rank_score`.

**Why.**
- No model is always best. The tournament throws away information from the 3
  losers. Stacking uses it.
- In quant Kaggles, stacking 3–5 base models consistently adds 10–30% to IC over
  best-single.
- Regime features let the meta-learner implicitly pick "use XGBoost in CHOPPY,
  QLearning in BULL_CALM" — current tournament makes a single static pick.

**Effort**: 1 day. Fit a LightGBM on `[base_model_scores] + [regime_features]` →
`label`. Persist the meta-model. Score at inference = meta-model prediction.

### 3.4 Distributional output via NGBoost (medium impact, medium effort)

**What.** Replace XGBoost classifier + separate calibration with
[NGBoost](https://arxiv.org/abs/1910.03225), which outputs the full parameters of
a distribution (e.g. Normal(μ, σ)) over future returns. You get:
- `E[R - SPY]` — what rotation needs, no separate ER regression.
- `Var[R - SPY]` — uncertainty, unavailable today.
- P(R > threshold) — analytically computed from the distribution, no calibration.

Built on scikit-learn, similar runtime to sklearn GBM, published by Stanford ML group.

**Why.**
- Unifies probability + expected-return + uncertainty into one head. Today these
  are three separate fitted objects.
- P(outperform) is computed from the predicted distribution, not from a post-hoc
  calibration on tiny samples. Solves 2.3 (calibration fragility) for free.
- Uncertainty → confidence-aware sizing: reduce `max_position_pct` when σ is high,
  skip candidates whose Sharpe CI includes zero.

**Effort**: 2 days. New `NGBoostPanelModel` class (could combine with 3.1 since
ngboost supports the same API). Persistence is a simple pickle → JSON schema.
Test that the calibrated probability recovers correct base rates.

Caveat: NGBoost doesn't support rank objectives natively. If we go with 3.1 (LTR),
we'd keep LightGBM/XGBoost for ranking and use NGBoost only for the uncertainty/ER
head. That's fine — two artifacts per model, both trained on the same panel.

### 3.5 Cross-sectional factor features (medium impact, cheap)

**What.** Add per-date cross-sectional z-scores of:
- Size (log market cap)
- Value (earnings yield, book-to-price if we can pull from yfinance.info / OpenBB)
- Quality (ROE, gross profitability)
- Momentum (12m - 1m return)
- Short-interest (if available)
- Beta to SPY (rolling 60d)
- Residual momentum (momentum orthogonal to SPY)

The Fama-French / Carhart factor literature is clear these exposures have
persistent risk premia. They're cheap to compute and orthogonal to pure technicals.

**Why.** Current features are purely technical/relative. The 5-day horizon is short
enough that fundamentals don't dominate, but stocks do trade on their factor
exposures even at daily frequency, especially cross-sectionally. Adds diversification
to the feature set.

**Effort**: 1–2 days depending on data source. yfinance provides enough for size +
beta + momentum. Fundamentals (earnings yield) need OpenBB or quarterly statement
caching.

### 3.6 Volatility-normalised labels (low-medium impact, trivial)

**What.** Change label from `fwd_return > 0.03` to `fwd_return / realised_vol > k`.
Same k across tickers. LLY's 3% move is ~0.5σ; XLU's 3% move is ~1.5σ — they're not
the same signal.

**Why.** Label distribution becomes comparable across tickers. Per-stock base rate
of `y=1` stabilises. Downstream calibration works better. Also composable: if we
later drop the threshold entirely and use regression, no change required.

**Effort**: 30 minutes. One line change in `features.py`.

### 3.7 Learning-to-rank objective, even without panel model (low effort, medium impact)

If (3.1) is too invasive, a cheap halfway step: keep per-stock models but switch the
tournament's evaluation metric from Sharpe to **cross-sectional IC averaged over
OOS bars**. Better model selection without touching the training code.

**Effort**: 2 hours. Change `oos_sharpe` in tournament.py to an IC computation
across tickers per bar, averaged.

### 3.8 Graph / relational models (TFT-GNN, etc.) — out of scope for now

Recent papers (e.g. [TFT-GNN 2025](https://www.mdpi.com/2673-9909/5/4/176),
[Multi-Sensor TFT 2025](https://www.mdpi.com/1424-8220/25/3/976)) report meaningful
lifts from attention models that explicitly represent asset relationships. These
are 5–10× the engineering effort of (3.1) and require GPU or patient training.
Revisit only after the panel LTR baseline is solid.

---

## 4. What I'd ship, in order

**Stage 1 — "better foundations"** (highest lift, ~4–5 days total):
1. **Switch tournament selection metric from Sharpe to IC** (3.7) — 2 hours.
2. **Add purged K-fold CV** around tournament (3.2) — 1 day.
3. **Replace tournament with cross-sectional panel LTR model** (3.1) — 2–3 days.
   Keep per-stock as a fallback via config flag so we can A/B.
4. **Volatility-normalised labels** (3.6) inside 3.1 — trivial.

At this point we should see measurable IC improvement and much more stable
model-selection behaviour.

**Stage 2 — "uncertainty + stacking"** (~3 days):
5. **Stacking meta-learner** over current heads + regime features (3.3) — 1 day.
6. **NGBoost for ER + uncertainty head** alongside LTR ranker (3.4) — 2 days.

**Stage 3 — "alpha expansion"** (~1 week, ongoing):
7. **Cross-sectional factor features** (3.5).
8. (Speculative) TFT / graph model if Stage 1+2 plateaus.

---

## 5. What *not* to do

- **Don't add more model types to the tournament**. Five/six isn't better than
  four — with ~500 rows each, the variance in model selection swamps the diversity
  gain. Fix the data problem first (3.1).
- **Don't hand-tune `buy_threshold`/`sell_threshold` per stock**. These live in
  `strategy_config.json → model_params` and are shared. If we want per-stock
  thresholds, derive them from score quantiles on calibration data, not by hand.
- **Don't keep the `rs_score` blend channel "just in case"**. It's zero-weighted by
  the daily recalibrator, which is evidence it's redundant. Either (a) remove it
  to simplify, or (b) replace it with something that actually adds orthogonal
  signal — residualised industry-relative momentum, short-interest change, or
  analyst revisions. `norm(20d_return − sector_20d_return)` appears to carry no
  information the model doesn't already have.

---

## 6. Open questions before implementation

These shape the priority list and I'd like your view:

1. **Compute budget**. Is GPU / multi-hour training acceptable? If yes, TFT /
   Transformer options open up. If no, stick to CPU LightGBM/XGBoost panel.
2. **Fundamental data**. Are you willing to pull from OpenBB / a paid source for
   value & quality factors? Affects whether (3.5) is worth doing.
3. **Retraining cadence**. Currently daily. If we move to a panel model, daily
   full retrain might be overkill — weekly full + daily incremental is common.
   Do you want the panel fit once a week and frozen, or refit daily?
4. **A/B experimental setup**. Do we keep a "research" and "production" model
   side-by-side with notebook-only comparisons, or swap the live pipeline once a
   new approach beats the baseline on holdout?

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

Purged cross-validation:
- [Purged cross-validation — Wikipedia](https://en.wikipedia.org/wiki/Purged_cross-validation)
- [Advances in Financial Machine Learning — Reasonable Deviations notes](https://reasonabledeviations.com/notes/adv_fin_ml/)
- [skfolio CombinatorialPurgedCV](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html)
- [QuantInsti — Cross Validation in Finance: Purging, Embargoing, Combinatorial](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/)
