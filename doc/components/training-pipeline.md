# Model Training — Design + References

**Last updated**: 2026-05-23 (weekly unique staging + production-semantic WF config derivation)

> **2026-05-20 update — training paths**:
>
> **1. Production HF PatchTST panel scorer** (PRIMARY since 2026-06-05):
> - Active production config uses `ranking.panel_scoring.kind="hf_patchtst"`.
> - The active checkpoint/calibrator still use shadow-named paths pending artifact registry cleanup.
> - The previous XGBoost primary remains the readonly shadow / rollback baseline.
>
> **1b. XGBoost panel-LTR** (previous primary / rollback):
> - `scripts/daily_retrain_alpha158_fund.sh` is the wrapper for the alpha158+fund+sentiment retrain pipeline (172 features = alpha158 + 5 fund + 3 PEAD + 3 SUE + 3 sentiment).
> - Weekly `weekly_wf_promote.sh` writes unique scorer/calibrator staging artifacts, runs the strict 3-cut WF + sanity + trade-ledger gate, and swaps active production only after `wf_gate_metadata.passed=True`.
> - Label: `fwd_60d_excess` (60-day forward excess return), `lookahead_days=60`
>
> **2. NGBoost head** (PROMOTED 2026-05-17, σ-wire dormant):
> - `scripts/train_ngboost_proper.py` (best-by-val_IC selection + XGB-baseline quality gate refusing save when val_IC < +0.0294)
> - Artifact: `artifacts/prod/ngboost-head.alpha158_fund.json` (val_IC +0.0352, σ-calib +0.274)
> - σ-wire stays OFF per 3-condition A/B all NULL/negative (2026-05-17)
>
> **3. Calibrator** (Platt scaling, switched from isotonic 2026-05-18):
> - `scripts/fit_panel_calibrator.py --method platt`
> - Monthly cron `monthly_calibrator_refresh.sh` with H2a (non-collapse) + H2b (IC-regression) hard gates + auto-rollback (commit `637594e`)
> - ER clip [-0.20, +0.20] at train-site + load-time guard (P0 2026-05-15)
>
> **4. HF PatchTST training path** (shipped 2026-05-19, primary since 2026-06-05):
> - `scripts/patchtst_hf.py` — HF `transformers.Trainer` + `PatchTSTModel` backbone + dual head (rank_head + dist_head Student-t df/loc/scale)
> - Margin Ranking loss (CIKM 2025) + Student-t NLL multi-task
> - `PerRegimeICCallback` selects best epoch by min-across-regime IC (PRIME DIRECTIVE in code)
> - `load_best_model_at_end=True` + cosine LR + warmup
> - Optional `--film-regime-cond` FiLM regime conditioning (Perez 2017)
> - **Known issue (project_patchtst_hf_save_mismatch memory)**: pre-refactor checkpoints saved LAST epoch not best — fixed post-2026-05-19 via HF Trainer's `load_best_model_at_end`. Old checkpoints loaded by `hf_patchtst_scorer.py` with `head.*` → `rank_head.*` rename map.
> - Drivers: `eval_hf_trainer_5cut_5seed.py`, `eval_hf_film_5cut_5seed.py`
>
> **5. DLinear baseline** (§5.12 must-have, 2026-05-19):
> - `scripts/dlinear_baseline.py` — single-matmul trend+seasonal decompose, Margin Ranking loss
> - Driver: `eval_dlinear_5cut_5seed.py`
> - Gate: if PatchTST cannot beat DLinear by ≥+0.005 min-regime IC, architecture is NOT the bottleneck
>
> **Per-regime IC tracking**: `kernel/hmm_regime_labels.py` provides stateless 4-regime taxonomy (BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR) via SPY OHLCV thresholds. Used by `PerRegimeICCallback` and downstream analysis scripts.
>
> **Backends registered in model_registry.py**: `hf_patchtst` (primary), `xgb` (previous primary / readonly shadow), `patchtst` (legacy custom, pre-2026-05-19 refactor), `regime_router` (FROZEN as dormant baseline per arXiv 2603.13252).
>
> **alpha158_linear** (E29): walk-forward NO-GO 2026-05-07; dormant.

## Overview

renquant_104 trains **two layers** of machine-learning models:

1. **Per-ticker models** (Layer 1): one model per symbol, predicts whether
   the next-bar action is BUY/HOLD/SELL based on per-ticker features.
   Trained via tournament selection from 4 model types.

2. **Cross-sectional panel model** (Layer 2 — XGBoost / LightGBM /
   Transformer): one model across the whole watchlist, predicts the
   relative ranking of tickers based on neutralised cross-sectional
   features. Output is the `panel_score` and (optionally) Kelly-sizing
   `μ, σ` from a NGBoost head.

Both layers feed the rotation algorithm (see `doc/components/rotation.md`).

---

## Round-7 (2026-04-26) — acceptance-gated retrain flow

`scripts/train_104.py` wraps `FullTrainingPipeline` in an 11-gate
`ModelAcceptanceGate` (`kernel/model_acceptance.py`):

```
1. Snapshot active panel-ltr.json → .pre-train.json (rollback safety)
2. Run FullTrainingPipeline (writes new artifact to panel-ltr.json)
3. Move new content → panel-ltr.staging.json
4. Restore prior content from .pre-train.json → panel-ltr.json (so the
   gate sees prior=active, candidate=staging)
5. ModelAcceptanceGate(config=acceptance_cfg).evaluate(staging, active)
   — runs G1-G11 and returns AcceptanceVerdict
6. PASS  → promote(staging, active):
              JSON validation (audit fix #2) → atomic os.replace via
              .incoming.json (audit fix #12, no missing-active window)
              prior preserved at panel-ltr.previous.json
   FAIL  → reject(staging, _acceptance_log/, verdict):
              archives staging + verdict.txt
              ntfy push to "renquant"
              sys.exit(2)
7. finally: clean up .pre-train.json snapshot (success OR rejection;
   audit fix #9 always-clean)
```

Operator overrides (dangerous):
- `--skip-acceptance` flag → bypass gates for one run
- `acceptance.enabled: false` in strategy_config → disable globally
- `--skip-acceptance` AND `--skip-baseline --skip-recalibrate` is the
  pattern used by `strategy_config.lgbm_macro.json` and similar
  experimental configs (writes directly to artifacts/ — be ready to
  restore from `.<backend>.bak.json` afterwards)

Full SOP: [`model-selection.md`](model-selection.md).

---

## Layer 1 — Per-Ticker Models (`training/models.py`)

All four implement `BaseModel` ABC:
```
.train(features, labels) -> dict      # train metadata (sharpe, etc.)
.predict(features) -> str | float     # action / score
.predict_bulk(df) -> pd.Series        # vectorised
.predict_score_bulk(df) -> pd.Series  # continuous score (calibrated downstream)
.save(path) / .load(path)             # JSON-only artifact
```

### 1.1 ManualModel

Hand-written multi-indicator threshold voting. Score = sum of (signed
indicator votes). Used as deterministic baseline + cold-start fallback.

**Refs**:
- Wilder J. W. 1978. *New Concepts in Technical Trading Systems*. RSI/ADX foundation.

### 1.2 ClassificationModel

Forward-return-labeled features → BagLearner of RTLearner (random tree).
- Labels: `+1` if `forward_return > +threshold`, `−1` if `< −threshold`, `0` otherwise.
- Forward return is computed against SPY (relative outperformance) to
  prevent bull-market always-buy bias.
- Forest of K random trees with bootstrapped samples.

**Refs**:
- Breiman L. 2001. "Random Forests." *Machine Learning* 45 (1): 5–32.
- Geurts P., Ernst D., Wehenkel L. 2006. "Extremely Randomized Trees." *Machine Learning* 63 (1): 3–42.

### 1.3 QLearningModel

Tabular Q-learning over discretised state space.
- States = bins of (RSI, MACD, momentum, etc.) — typically 5–10 bins per dimension.
- Actions = {BUY, HOLD, SELL}.
- Reward = next-bar return − transaction cost.
- Trained over N epochs with ε-greedy exploration.
- Score = `Q(buy) − Q(sell)`.

**Refs**:
- Watkins C. J. C. H. 1989. "Learning from Delayed Rewards." PhD thesis,
  Cambridge. Original Q-learning formulation.
- Sutton R. S., Barto A. G. 2018. *Reinforcement Learning: An Introduction*.
  2nd ed. MIT Press. Chapter 6 (TD methods) + Chapter 11 (off-policy).

### 1.4 XGBoostModel

Two `XGBClassifier` models per ticker:
- `xgb_buy`: P(BUY-vs-rest) — binary classifier on `+1`-labeled rows.
- `xgb_sell`: P(SELL-vs-rest) — binary classifier on `−1`-labeled rows.
- L1/L2 regularisation, max_depth ~ 6, learning_rate ~ 0.05.
- Score = `P(buy) − P(sell)` (continuous).

**Refs**:
- Chen T., Guestrin C. 2016. "XGBoost: A Scalable Tree Boosting System."
  *KDD '16*: 785–794.
- Friedman J. H. 2001. "Greedy Function Approximation: A Gradient
  Boosting Machine." *Annals of Statistics* 29 (5): 1189–1232.

## Tournament Selection (`training/tournament.py`)

For each ticker, train all 4 model types on the same 70/30 walk-forward
split, then export the winner by **OOS Sharpe ratio** (after-tax).

- Floor: `OOS Sharpe ≥ 1.0` (raised from 0.8 in renquant_103). Below floor
  → ticker excluded from the universe.
- Tie-break: in-sample Sharpe.
- Each model uses `abs(hash(ticker)) % 2^32` as seed for reproducibility.

**Refs**:
- Sharpe W. F. 1966. "Mutual Fund Performance." *Journal of Business* 39 (1): 119–138.
- Bailey D. H., López de Prado M. 2014. "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
  *Journal of Portfolio Management* 40 (5): 94–107. → motivates the
  Sharpe ≥ 1.0 floor + multi-model tournament (variance reduction by
  ensembling).

## Score Calibration (`common/models/scoring.py`)

Per-model raw scores are NOT cross-model comparable (e.g., XGBoost
returns probabilities, QLearning returns Q-value differences). To
enable cross-ticker ranking, each model's score is calibrated via
`score_calibration` metadata stored in `policy-metadata.json`:

| Sample size | Method                       | Reference                     |
|------------:|------------------------------|--------------------------------|
| n ≥ 300     | **Isotonic** (PAV algorithm) | Zadrozny-Elkan 2002             |
| 120 ≤ n < 300 | **Platt** (sigmoid)        | Platt 1999                      |
| n < 120     | **Constant** (base rate)     | fall-back                       |

Output: probability that next-`H`-bar return beats SPY by `θ%` —
**cross-model comparable** rank score.

**Refs**:
- Zadrozny B., Elkan C. 2002. "Transforming Classifier Scores into
  Accurate Multiclass Probability Estimates." *KDD '02*.
- Platt J. C. 1999. "Probabilistic Outputs for Support Vector Machines
  and Comparisons to Regularized Likelihood Methods." *Adv. in Large-
  Margin Classifiers*.
- Brodersen K. H., Ong C. S., Stephan K. E., Buhmann J. M. 2010. "The
  Balanced Accuracy and Its Posterior Distribution." *ICPR '10*.
  Theoretical justification for sample-size-aware method dispatch.

## Layer 2 — Cross-Sectional Panel-LTR

Single learning-to-rank model trained on the **whole watchlist panel**:

- Inputs: 24 features per (ticker, date) — 16 neutralised indicators +
  4 technical factor z-scores + 4 fundamental z-scores + earnings_surprise +
  insider_net_buy.
- Labels: beta-neutralised + sector-/size-neutralised forward excess
  returns (10-day lookahead).
- Per-row weights: `concurrency × age` (penalise highly-redundant + stale rows).
- Loss: NDCG-style learning-to-rank.

### Backend Dispatch (`panel_ltr.backend`)

Three backends share the same fit/predict interface (`PanelLTRModel`,
`PanelLGBMModel`, `PanelTransformerModel`):

#### XGBoost (`training_panel/ltr_model.py`) — current golden

- Objective: `rank:pairwise`.
- Group sizes: per-date row counts.
- Monotone constraints on 6 economically-signed factors:
  `beta_60d_z: -1, mom_12_1_z: +1, resid_mom_z: +1, earnings_yield_z: +1,
   roe_z: +1, gross_profitability_z: +1, short_pct_float: -1,
   insider_net_buy_90d: +1`.
- L1/L2 regularisation, `max_depth: 6`, `eta: 0.02`, early stopping.

**Why XGBoost wins on this panel** (per 2026-04-23 + 2026-04-25 A/B):
- L1/L2 + tree splits cheaply ignore noisy features.
- Level-wise growth is naturally regularised vs LightGBM's leaf-wise.
- Monotone constraints encode economic priors directly (Catania-Politis 2020).

**Refs**:
- Burges C. J. C. 2010. "From RankNet to LambdaRank to LambdaMART: An
  Overview." Microsoft Research Tech Report MSR-TR-2010-82.
- Liu T.-Y. 2009. "Learning to Rank for Information Retrieval."
  *Foundations and Trends in Information Retrieval* 3 (3): 225–331.

#### LightGBM (`training_panel/lgbm_ltr.py`) — shelved (2026-04-25)

- Objective: `lambdarank` with `ndcg_at: [5, 10]`.
- Per-row weights normalized to mean=1.0 (LGB-WEIGHT-NORM fix — pre-fix
  weights were ~3e-4 → gradient signal vanished → train_ic stuck at iter 1).
- T2-1 retest result: OOS scorer_mean_ic=0.0269 vs XGBoost 0.0476 (-44%).

**Why LightGBM loses here**:
- Leaf-wise growth more aggressive → memorises train distribution faster.
- Train→OOS gap = 5.4× (vs XGBoost ~3×).

**Refs**:
- Ke G., Meng Q., Finley T. et al. 2017. "LightGBM: A Highly Efficient
  Gradient Boosting Decision Tree." *NeurIPS '17*.
- Catania L., Politis D. 2020. "Empirical Asset Pricing with ML."
  *J. Risk*. Documents the LGBM-vs-XGB asymmetry on financial cross-sectional ranking.

#### Transformer (`training_panel/transformer_model.py`) — shelved (2026-04-23)

- Architecture: 3-layer encoder, d_model=128, 4 heads, ListNet loss.
- Self-attention within date-group (cross-sectional, not temporal).
- Device: MPS (Apple Silicon).
- A/B 2026-04-23: OOS IC = +0.0063 vs XGBoost +0.0309 → 5x worse.

**Why Transformer loses here**:
- Dataset too small (1,256 dates ≪ ImageNet-equivalent 1M+ samples).
- Noisy features dilute attention.
- Same regularisation budget can't compete with XGBoost's cheap L1/L2.

**Refs**:
- Vaswani A., Shazeer N., Parmar N. et al. 2017. "Attention Is All You
  Need." *NeurIPS '17*.
- Chen Y., Pelger M., Zhu J. 2024. "Deep Learning in Asset Pricing."
  *Management Science*. Cross-sectional transformer needs >5,000 dates
  to beat trees.

### Cross-Validation (`training_panel/purged_cv.py`)

Purged K-Fold + CombinatorialPurgedCV (CPCV) per López de Prado.
- Embargo: `cv_embargo_days: 5` (bars between train and test).
- Lookahead: `lookahead_days: 10` (label horizon).
- 15-fold CPCV for stability metric (`pool_ic` is the mean across folds).

**Refs**:
- López de Prado M. 2018. *Advances in Financial Machine Learning*.
  Wiley. Chapter 7 (cross-validation in finance) + Chapter 11 (CPCV).
- López de Prado M. 2019. "A Robust Estimator of the Efficient Frontier."
  *SSRN 3469961*. Theoretical basis for CPCV variance reduction.

### Calibration Layer (`training_panel/global_calibrator.py`)

After the panel scorer, a global isotonic calibrator maps raw scores
to probability of `forward_return > 0.03`. Calibrator is trained
post-hoc on `panel.scorer_oos_predictions × panel.labels`.

**Refs**:
- Niculescu-Mizil A., Caruana R. 2005. "Predicting Good Probabilities
  with Supervised Learning." *ICML '05*. Demonstrates calibration
  necessity for tree-based models.

### NGBoost μ,σ Head (Stage 2 — `training_panel/ngboost_head.py`)

Optional second head fitted on the same panel, predicts `Normal(μ, σ)`
per row. Used by:
- Kelly sizing: `f* = μ/σ²` per ticker, capped at `max_concentration`.
- Score combination (`score_mode = mu_minus_lambda_sigma`): override
  `panel_score = μ − λσ` for risk-aware ranking.

**Refs**:
- Duan T., Avati A., Ding D. Y. et al. 2020. "NGBoost: Natural Gradient
  Boosting for Probabilistic Prediction." *ICML '20*.
- Kelly J. L. 1956. "A New Interpretation of Information Rate." Foundation
  of `f* = μ/σ²` for continuous Gaussian.
- Thorp E. O. 2006. *Handbook of Asset and Liability Management*. Half-Kelly
  variance reduction → `kelly_sizing.fractional = 0.5`.

## Feature Engineering (`training_panel/factors.py`, `kernel/fundamentals.py`)

### Technical Factors (z-scored cross-sectionally per date)

1. **Beta** (`beta_60d_z`): rolling 60-day OLS β vs SPY; sign-monotonic neg.
2. **Momentum 12-1** (`mom_12_1_z`): 252-21d return; **the** anomaly.
3. **Residual momentum** (`resid_mom_z`): excess of 21d return over β-implied.
4. **Realised vol** (`realized_vol`): 20d std of returns.
5. **Drawdown from peak** (`drawdown_from_peak`).
6. **Volume shift** (`volume_shift`): 20d vs 60d log-volume.
7. **Price-to-high** (`price_to_high`): close / 252-day high.
8. **Amihud illiquidity** (`amihud_illiquidity`): `|ret| / dollar_volume`.

**Refs**:
- Jegadeesh N., Titman S. 1993. "Returns to Buying Winners and Selling
  Losers." *Journal of Finance* 48 (1): 65–91. **Momentum**.
- Asness C. S., Moskowitz T. J., Pedersen L. H. 2013. "Value and Momentum
  Everywhere." *Journal of Finance* 68 (3): 929–985.
- Frazzini A., Pedersen L. H. 2014. "Betting Against Beta." *Journal of
  Financial Economics* 111 (1): 1–25. → low-β anomaly.
- Amihud Y. 2002. "Illiquidity and Stock Returns." *Journal of Financial
  Markets* 5 (1): 31–56.
- Blitz D., Huij J., Martens M. 2011. "Residual Momentum." *Journal of
  Empirical Finance* 18 (3): 506–521.

### Fundamental Factors (`kernel/fundamentals.py`)

1. **Earnings yield** (`earnings_yield_z`): EPS / price.
2. **ROE** (`roe_z`): net income / equity.
3. **Gross profitability** (`gross_profitability_z`): gross profit / assets.
4. **Book-to-price** (`book_to_price_z`): book equity / market cap.

Cached at `data/fundamentals/{SYMBOL}.parquet` via OpenBB. Sector-median
fill for missing values, then cross-sectional z-score.

**Refs**:
- Fama E. F., French K. R. 1992. "The Cross-Section of Expected Stock
  Returns." *Journal of Finance* 47 (2): 427–465.
- Novy-Marx R. 2013. "The Other Side of Value: The Gross Profitability
  Premium." *Journal of Financial Economics* 108 (1): 1–28.
- Hou K., Xue C., Zhang L. 2015. "Digesting Anomalies: An Investment
  Approach." *Review of Financial Studies* 28 (3): 650–705. → Q-factor
  model bridge.

### Earnings Surprise (`kernel/earnings_surprise.py`)

`earnings_surprise_cum`: trailing-4Q cumulative surprise %, sourced from
yfinance `.earnings_dates`. Cached at `data/earnings_surprise/{SYM}.parquet`.

**Refs**:
- Bernard V. L., Thomas J. K. 1989. "Post-Earnings-Announcement Drift:
  Delayed Price Response or Risk Premium?" *Journal of Accounting Research*
  27 (Supplement): 1–36. The "PEAD" anomaly.

### Insider Trades (`kernel/insider_trades.py`)

`insider_net_buy_90d`: trailing-90d net executive buy (USD) parsed from
SEC EDGAR Form 4 filings. Executive-only filter (officer/director codes).

**Refs**:
- Lakonishok J., Lee I. 2001. "Are Insider Trades Informative?" *Review
  of Financial Studies* 14 (1): 79–111.
- Cohen L., Malloy C., Pomorski L. 2012. "Decoding Inside Information."
  *Journal of Finance* 67 (3): 1009–1043.

## Neutralisation (`training_panel/neutralization.py`)

Cross-sectional regression on (sector dummies, log market cap), then
take residuals. Removes "size + sector" from raw factor returns so the
panel-LTR learns *idiosyncratic* signal, not factor exposure.

**Refs**:
- Daniel K., Titman S. 1997. "Evidence on the Characteristics of Cross
  Sectional Variation in Stock Returns." *Journal of Finance* 52 (1): 1–33.
- Fama E. F., MacBeth J. D. 1973. "Risk, Return, and Equilibrium:
  Empirical Tests." *Journal of Political Economy* 81 (3): 607–636.

## Training Schedule

Configurable via `training.cadence` + `training.allowed_weekdays`:

- **renquant_104 current**: `cadence: "custom"`, `allowed_weekdays: [1, 3, 6]`
  (Tue / Thu / Sun PT). Each weekday training runs full pipeline (~30 min).
- `model_ttl_days: 1` per-ticker cache gate — skip if metadata's
  `trained_date` is within TTL.

Per-ticker training in parallel via `multiprocessing.Pool` (capped at
`mp.cpu_count() - 1`).

## Test Coverage

- `tests/test_training_modules.py` (16 tests): features, tournament, export.
- `tests/test_panel_training_pipeline.py` (15 tests): full pipeline E2E.
- `tests/test_panel_bugfixes.py` (6 tests): calibration order + z-score wiring.
- `tests/test_panel_orthogonal_factors.py` (9 tests): Round-3 factors.
- `tests/test_ngboost_head.py` (12 tests): NGBoost μ,σ fit/predict/save/load.
- `tests/test_fundamentals_cache.py` (9 tests): OpenBB cache.
- `tests/test_earnings_surprise.py` (9 tests): yfinance surprise cache.
- `tests/test_insider_trades.py` (11 tests): SEC EDGAR Form 4 parser.
- `tests/test_panel_transformer.py` (12 tests): transformer fit/predict.
- `tests/test_panel_hourly_wiring.py` (8 tests): hourly bar features.
- `tests/test_regime_calibrator.py` (10 tests): regime-conditional calibrator.
- `tests/test_training_cadence.py` (8 tests): cadence + TTL gates.
- `tests/test_universe_alignment.py` (18 tests): universe admission floors.

## Comparison Tables

### Backend Comparison (this session)

See `doc/experiments/panel-backend-comparison.md` for the latest 3-way head-to-head.

### Per-Ticker Model Type Distribution (latest training)

| Model type     | Count | Notes |
|----------------|------:|-------|
| Manual         | ~14   | Cold-start fallback; rarely wins tournament |
| Classification | ~13   | Random-forest based; good for noisy tickers |
| QLearning      | ~12   | Best when state-action transition is regular |
| XGBoost        | ~10   | Wins for tickers with strong feature interactions |

Tournament floor `Sharpe ≥ 1.0` excluded ~50/108 tickers in latest run
(below-floor list in `live.runner` MODEL SUMMARY at startup).

## Roadmap

### Tier 1.5 (Current — XGBoost panel + NGBoost head)
- ✅ Round 1-5 factor expansion (8 technical + 4 fundamental + 2 EDGAR)
- ✅ NGBoost μ,σ head + Kelly sizing
- ✅ Hourly features (Plan G)
- ✅ Regime-conditional calibrator (Plan F)

### Tier 2 (Next — Backend Optimization)
- ⏳ T2-1 LightGBM swap (shelved 2026-04-25 — confirmed inferior)
- ⏳ T2-2 noise feature pruning
- ⏳ T2-3 training window tuning
- ⏳ T2-4 Boyd convex MPC (Phase 3 rotation)
- ⏳ T2-5 Transformer revisit (deferred — needs >5k dates per Chen 2024)

### Tier 3 (Roadmap)
- Multi-task learning (joint return + volatility heads)
- Online updates (rolling window vs full retrain)
- Reinforcement learning at portfolio level (Q-learning over allocation)
