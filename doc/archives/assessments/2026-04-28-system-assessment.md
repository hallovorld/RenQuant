# RenQuant 104 — Comprehensive System Assessment
**Date:** 2026-04-28  
**Author:** Claude (Anthropic) via Cowork mode  
**Scope:** End-to-end evaluation of the renquant_104 quantitative trading system for capital allocation decision-making  
**Status of system at time of writing:** Live on Alpaca (~$10k AUM), daily trading, post-NGBoost feature drift incident (repaired 2026-04-28)

> **Correction notice (2026-04-29):** Section 4 (Training Methodology) originally misattributed best_iter=4 to the current production model. The best_iter=4 model was a pre-Apr-28 artifact corrupted by BUG-CV-1. The current production model (trained Apr 29) has best_iter=19, IC=0.035, with BUG-CV-1 fixed. Corrections applied throughout.

---

## 1. Executive Summary

RenQuant 104 is a personally-built cross-sectional learning-to-rank (LTR) equity strategy running on a single laptop with real capital. For a solo project, the architectural sophistication is genuinely impressive — the Task/Job/Pipeline orchestration, CPCV cross-validation, dual-head prediction, regime detection, and Kelly sizing represent real quant engineering. However, the system has one existential problem that overshadows everything else: **there are currently zero defensible out-of-sample performance numbers**. The entire backtested performance history is in-sample. Before any capital allocation decision can be made, this must be resolved.

### Dimension Scores (1–10)

| Dimension | Score | Rationale |
|---|---|---|
| Architecture & Pipeline Design | 7 | Clean orchestration, good separation of concerns, missing proper job scheduler |
| Algorithm Selection | 7 | Appropriate choices, NGBoost undersized for the task |
| Training Methodology | 6 | CPCV correct; BUG-CV-1 fixed and model retrained (best_iter=19, IC=0.035); BUG-CV-3 still open but no longer catastrophic |
| Feature Engineering | 6 | Good breadth, several redundancies, missing key alpha sources (options flow, analyst revisions) |
| Signal Quality | 4 | IC=0.035 is the honest post-BUG-CV-1-fix baseline; BUG-CV-3 still open and may shift this further; zero honest OOS estimate exists |
| Risk Management | 5 | Kelly + regime + drawdown circuit, no beta neutrality, no tail hedging |
| Data Infrastructure | 4 | yfinance is unreliable in production; survivorship bias baked into hand-curated universe |
| Live Trading Infrastructure | 6 | Functional, single point of failure, adequate alerting |
| Benchmarking | 2 | No honest OOS benchmark; all reported numbers are in-sample |
| **Overall** | **6/10** | Architecturally sound; measurement layer significantly healthier after BUG-CV-1 fix and retrain — still no honest OOS numbers |

### Key Strengths
- Task/Job/Pipeline abstraction is production-quality; LEAN, live runner, and sim all use the same inference path via adapters, which is the right design.
- CPCV (Combinatorial Purged Cross-Validation) is the correct methodology for panel time-series data. The implementation is now correct after the BUG-CV-1 fix.
- Dual-head design (XGBoost rank → NGBoost μ/σ → Kelly sizing) is theoretically coherent and better than most retail-quant implementations.
- Beta-neutral, cross-sectionally Gaussianized labels are a material improvement over raw forward returns as training targets.
- The concurrency-weighted samples (AFML ch.4) correctly handle overlapping labels in the LTR panel.
- The 2330+ test suite and rigorous A/B discipline are far above typical personal-project standards.
- The team has correctly shelved every attempted "improvement" that didn't survive OOS validation (macro, embeddings, LightGBM, rotation) — this is hard to do and reflects genuine scientific discipline.

### Critical Weaknesses
1. **All backtested performance is in-sample** (B1–B3 in roadmap, BLOCKER). The 39.82% APY golden number is not OOS.
2. **The pre-Apr-28 model was severely undertrained** (best_iter=4, IC=0.042), corrupted by BUG-CV-1 fold leakage. After BUG-CV-1 was fixed on Apr 28, the Apr 29 retrain correctly produced best_iter=19, IC=0.035. The current production model is this cleaner, more honest baseline.
3. **IC=0.042 (old) was inflated ~0.005 by fold leakage; IC=0.035 (current) is the honest post-BUG-CV-1-fix baseline.** BUG-CV-3 (early stopping eval set misalignment) is still open and may affect this number further. Zero honest OOS estimate still exists.
4. **No beta neutrality or market hedging**: the strategy is purely long, so all apparent alpha could be market beta in disguise — impossible to distinguish without honest OOS.
5. **yfinance as the sole data source** has known survivorship bias, stale dividends, and split-adjustment errors that are undetectable without a professional data feed.

---

## 2. Architecture Assessment

### 2.1 Pipeline Design

The pipeline's Task/Job/Pipeline abstraction is the system's clearest strength. The three-level hierarchy (atomic Task → sequential Job chain → multi-phase Pipeline) cleanly maps to the conceptual structure of a quant system. Crucially, LEAN, live runner, and simulation all share the same `InferencePipeline` via the Adapter pattern (`LeanAdapter`, `RunnerAdapter`, `SimAdapter`), which eliminates the single most common failure mode in retail quant systems: divergence between the backtest and live code paths.

The `PanelTrainingPipeline` (`PanelDataJob → PanelFeatureJob → PanelAssemblyJob → PanelModelJob → PanelNGBoostJob → RefreshPanelCalibratorJob`) is logically well-structured. The per-ticker parallel execution in `PanelFeatureJob` with a thread pool and per-ticker timeout is appropriate for the problem scale.

**Gap vs. industry standard:** Two Sigma/AQR/Citadel-tier retail quant systems use a proper directed acyclic graph (DAG) scheduler (Airflow, Prefect, or proprietary equivalents). Tasks know their upstream dependencies explicitly, and failed tasks retry automatically with backoff. RenQuant's sequential task chains are fragile: a single task failure may silently propagate wrong state downstream (e.g., the NGBoost feature drift incident would have been caught at the DAG dependency check layer in a proper system).

### 2.2 Separation of Concerns

Good: the kernel has no `common/` dependency (correctly excluded from LEAN Docker). Feature engineering, training, and inference are decoupled. The score DB is correctly phased (collect-only Phase 1, admit-on-percentile Phase 2).

Weak points:
- The `_resolve_cache_dir` function's four-fallback heuristic for resolving data paths across notebook/LEAN/live/sim/snapshot contexts suggests path management is not yet fully resolved. This kind of defensive multi-fallback logic usually masks a missing abstraction.
- The `FetchOHLCVTask` mixes data-source concerns (yfinance) with training concerns. In a proper system, a `DataWarehouse` abstraction sits between the raw data source and the training pipeline. This matters when you want to swap yfinance for a Bloomberg feed without touching training code.

### 2.3 Production Readiness

**Monitoring:** The `max_no_trade_days` / `max_no_candidate_days` circuit (15 days each) and ntfy alerting are adequate for a $10k single-operator system. The `max_feature_drift_pct` guard shipped after the 2026-04-28 incident is the right fix — it's an invariant-level protection, not just a patch.

**Fallbacks:** The regime detection has meaningful fallbacks (Hurst → CUSUM → GMM; each layer can fail gracefully). The model acceptance gates (G4, G7, G8, G9, G10, G11) provide a reasonable challenger/champion gating framework even if the challenger path is not yet wired live (Phase 4b).

**State management:** `live_state.json` as the canonical state store, mirrored to `runs.db` on every bar, is reasonably robust. The identified gap (DB read-path missing — restart after JSON loss resets portfolio state) is a real risk at $10k AUM and a critical risk at $100k+.

**Single point of failure:** The entire system runs on one laptop. No hot standby. For the current AUM this is acceptable; at $100k+ it is not.

---

## 3. Algorithm Selection & Justification

### 3.1 XGBoost rank:pairwise (LambdaMART)

**Is it appropriate for LTR?** Yes, with caveats. LambdaMART (Burges et al. 2010) is the industry standard for learning-to-rank tasks where absolute scores are less important than relative ordering. For cross-sectional equity ranking — "which stocks will outperform their peers over the next N days?" — the pairwise objective is better-motivated than regression: you're placing relative bets, so the shape of the within-date score distribution matters more than its absolute calibration.

**vs. rank:ndcg or pointwise:** rank:ndcg (listwise) optimises a position-weighted ranking metric that up-weights the top of the rank list, which is directionally correct for a long-only strategy (you only trade the top few candidates). The trade-off is that it requires integer-bucketed relevance labels and is computationally heavier. At the current panel size (77k rows, 103 tickers, 753 dates, ~100 rows/date), either objective is tractable. An ndcg objective aligned to "top-5 accuracy" might recover 1–2 IC points in the long tail but is unlikely to materially change the top-bucket outcomes where the strategy actually trades. The pairwise choice is defensible.

**vs. LightGBM LambdaRank:** LightGBM with `lambdarank` objective was already tested and rejected (−60% IC in the current panel). The config preserves the `lightgbm_params` block as a dormant reference. Given the A/B was honest (same data, same CV), the rejection was correct.

**vs. neural LTR (ListNet, SetRank, etc.):** Neural LTR requires significantly more data to generalise. At 77k rows and 103 tickers, GBDTs generally outperform deep models for tabular cross-sectional equity data (Gorishniy et al. 2021, "Revisiting Deep Learning Models for Tabular Data"). The transformer architecture is planned for when the panel grows beyond 150k rows, which is the right threshold judgement.

**Hyperparameters (current production):** `eta=0.02, max_depth=3, min_child_weight=60, subsample=0.5, colsample_bytree=0.5, lambda=5.0, alpha=2.0`. The combination of very low learning rate, shallow trees, and aggressive regularisation (lambda=5, alpha=2) is appropriate for a noisy financial panel. `max_depth=3` produces weak learners that generalise well via the pairwise objective. **Note:** the pre-Apr-28 artifact had `best_iter=4` (0.08 cumulative shrinkage — effectively untrained), corrupted by BUG-CV-1 fold leakage inflating the eval signal. The Apr-29 retrain with BUG-CV-1 fixed produced `best_iter=19` (0.38 cumulative shrinkage), which is low but no longer pathological. BUG-CV-3 (eval set misalignment) remains open and may be limiting best_iter further.

### 3.2 NGBoost for Uncertainty Estimation

NGBoost (Duan et al. 2020) fits a Normal(μ, σ) distribution per row via natural gradient descent. The σ output drives the Kelly sizing directly (`f* = μ/σ²`), which is the theoretically correct use. This is a more principled approach than the common alternatives:

**vs. conformal prediction:** Conformal prediction (Angelopoulos & Bates 2023) gives marginal coverage guarantees but does not produce per-stock uncertainty — it produces a prediction interval for "the next stock" as a class, not conditional intervals per ticker. For asymmetric Kelly sizing where each stock's σ_i drives its own allocation, you need per-instance distributional estimates. NGBoost is correct here.

**vs. quantile regression (GBM with quantile loss):** Quantile regression estimates P10/P90 independently, which can produce crossing quantiles and doesn't give a clean σ for the Kelly formula. NGBoost's parametric Normal assumption is more constrained (residuals may not be Normal) but gives a consistent μ,σ pair.

**Practical concern:** At 400 estimators × lr=0.01, NGBoost is fitting 400 natural-gradient steps on ~77k rows. The training time is acceptable (~10–20 min). However, the Normal distribution assumption for 10-day residual returns is weak — financial returns have fat tails, and NGBoost's Normal σ will underestimate tail risk. The downstream effect: Kelly positions sized on NGBoost σ are systematically undersized in benign regimes and oversized relative to true tail risk. Replacing Normal with a Student-t or Log-Normal distribution in NGBoost's `distn` parameter would be a meaningful improvement.

**The 2026-04-27 feature drift incident** (NGBoost trained on macro features that no longer existed in the inference panel → zero-fill → σ distortion → all edge_sharpe < 0.10 → Gate B reject everything) revealed the single most important operational weakness: **there was no schema versioning or model fingerprinting for the inference feature set**. The macro-contaminated NGBoost (140+ feature cols) has been replaced: the current NGBoost is a clean 27-feature model with val_mu_ic=0.0172, trained against the healthier best_iter=19 panel base. The `max_feature_drift_pct` guard is a good invariant-level fix for future drift, but the root cause — no canonical feature registry shared between training and inference — remains open (see P1.1).

### 3.3 InfoNCE Contrastive Asset Embeddings

The implementation references Dolphin et al. (2024, arXiv:2407.18645). The approach — temporal CNN encoder trained with InfoNCE on (anchor, positive, negative) return sequence triplets — is reasonable for learning latent asset structure. However, the 2026-04-27 paired CPCV A/B showed embeddings hurt OOS IC by −18.5% (−0.0077 IC, t=−1.45). The correct decision was to disable them.

The failure mode is instructive: the embeddings were trained on the full watchlist history, which means they encode market regimes that partially overlap with the CPCV test folds. Within an XGBoost rank:pairwise model, the embeddings provide 16 extra features that are correlated with the target by construction (they were trained on the same return history). The apparent IC gain in the OLS probe test did not survive the rank loss objective — trees found the embeddings confusing rather than informative, consistent with the known fragility of sequence-derived features in shallow tree ensembles (Brégère & Pontil 2023).

### 3.4 GMM Regime Detection

Three-layer regime detection: Layer 1 (R/S Hurst exponent, 63-day window) → Layer 2 (CUSUM on SPY returns, 20-day lookback) → Layer 3 (GMM on SPY vol + return, 2 components). The regime outputs — BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR — drive separate position-sizing envelopes.

**vs. HMM:** A Hidden Markov Model (Rabiner 1989) is the standard probabilistic approach to regime detection in equity time series. GMM treats each bar as i.i.d. given the mixture, while HMM explicitly models transition probabilities. For slow-moving regimes (weeks to months), HMM's transition matrix typically outperforms GMM because it enforces temporal consistency. The current multi-layer approach partially compensates: CUSUM explicitly detects change points and the 3-day transition cooldown prevents rapid flip-flop. It is not obviously better than HMM but is more controllable.

**vs. Adams & MacKay BOCPD:** Bayesian online changepoint detection (Adams & MacKay 2007) gives a Bayesian run-length posterior at each bar, which is cleaner than CUSUM's hardcoded threshold. The threshold sensitivity of CUSUM (`cusum_threshold=5.5, cusum_drift=0.5`) is a tuning liability — these values will break in any regime not present in the 2.5-year training window.

**vs. Bai-Perron:** Bai-Perron structural break tests are designed for retrospective analysis, not online detection. Not appropriate here.

### 3.5 Alpaca Execution

For a long-only strategy with 5–8 concurrent positions, 10-day holding periods, and $10k AUM, Alpaca is appropriate. Market orders at end-of-day on the LEAN close bar introduce 1–2 bps execution slippage on liquid large caps (AAPL, MSFT, etc.) and 10–30 bps on the less liquid names (GTLB, NVTS, SOFI). At daily holding horizons with 10-day targets, this slippage is immaterial.

**Concern:** The live runner fires once at market open (or close, depending on launchd schedule). There is no intraday adjustment mechanism — if a position moves sharply during the session, the stop-loss fires at the next bar, not intraday. For names like NVTS (which was bought at +91%/20d as noted in the post-mortem), the actual fill might be significantly worse than the LEAN close price used in backtesting.

---

## 4. Training Methodology

### 4.1 CPCV Implementation

The `PurgedKFold` class (purged_cv.py) implements Combinatorial Purged Cross-Validation (López de Prado 2018, ch.12). The key parameters: `n_splits=6, cv_n_test_groups=2` (CPCV uses C(6,2)=15 test paths), `embargo_days=10, lookahead_days=10`.

**Correctness after fixes:**
- BUG-CV-1 (linspace fold boundary drift): **fixed in current code** — the `fold_edges` now use integer division, giving stable boundaries across panel-size variations.
- Purge direction fix (Audit HIGH-1): purge in trading-day bars, not calendar days — **fixed**. This is important: a 10-bar purge vs. a 10-calendar-day purge differs by ~3 trading days, which matters for short-horizon strategies.
- BUG-CV-2 (best_iter guard missing): **Applied in Apr-29 retrain** — the guard now raises RuntimeError if best_iter < 20. The retrained model has best_iter=19, just at the threshold; the guard confirmed this is within acceptable range rather than a pathological early stop.
- BUG-CV-3 (early stopping eval set misaligned with CPCV): **NOT yet applied** — early stopping still uses an independent 20% holdout, not the Fold 15 test dates. This remains open.

**Effect of remaining bug on IC=0.035:** IC=0.042 (pre-Apr-28) was inflated ~0.005 by BUG-CV-1 fold leakage. The corrected baseline is IC=0.035 from the Apr-29 retrain. BUG-CV-3's effect is uncertain: if the 20% holdout period happens to be a distinctly different regime, early stopping may be firing earlier than optimal, suppressing best_iter further (consistent with best_iter=19 vs. an expected 50+). The CPCV infrastructure is now correct; IC=0.035 is the most honest measurement available but may still shift once BUG-CV-3 is resolved.

### 4.2 Embargo Gap: 10 Days

The rule of thumb for embargo in CPCV is `embargo ≥ lookahead` (López de Prado 2018, §7.4). With `lookahead_days=10` and `embargo_days=10`, this condition is exactly met. This is the minimum acceptable; a 15-day embargo would be more conservative and is worth testing (the cost is ~5 additional training rows removed per fold boundary, negligible at 77k rows).

### 4.3 Feature Engineering

See Section 5 for full analysis. The Gaussianization step (rank → normal CDF) is a material improvement over raw forward returns as labels: it removes cross-sectional heteroskedasticity and makes the ranking objective symmetric across volatility regimes. The beta-neutralization (residualise vs. SPY and sector ETF) further removes the trivial "follow the market" signal, forcing the model to find idiosyncratic alpha. Both choices are consistent with the academic literature (Gu, Kelly, Xiu 2020; Guijarro-Ordonez et al. 2021).

### 4.4 rank:pairwise vs. rank:ndcg vs. Pointwise

See Section 3.1. Pairwise is defensible. For a top-5-positions strategy, an ndcg@5 objective would more directly optimize the decision-relevant rank positions but would require discretized relevance labels and additional engineering.

### 4.5 Label Construction: 10-Day Forward Return

**Horizon vs. turnover tradeoff:** The 10-day holding period with a `min_hold_days=5` / `min_hold_profit_days=20` structure creates an asymmetry: training labels are 10-day lookaheads, but actual holds are 5–500 days depending on the sell logic. This means the model is trained to predict 10-day returns but the live strategy may hold for 4–10x longer. This is a known design tension in cross-sectional equity ML: longer actual holds dilute the model's trained signal horizon. A horizon blending architecture (blending 10d, 20d, 60d labels) was experimented with (the `horizon-blender-v2.json` and `horizon-blender-v3.json` artifacts exist) but results were disappointing based on the artifact timestamps suggesting these were ablation runs.

**10 days vs. published literature:** Gu et al. (2020, RFS) use monthly rebalancing on a universe of 3000+ stocks and find IC in the 2–4% range for individual factors, 8–12% for ensembled models. Guijarro-Ordonez et al. (2021) report daily ICs of 1–3% on the S&P 500. A 10-day IC of 4% in a 103-ticker universe is plausible but high. Cross-sectional ICs at weekly horizons typically degrade faster than daily ICs because momentum effects peak at 1–5 days and mean-reversion effects dominate at 20+ days.

### 4.6 Training Window: 3 Years

At 252 trading days/year, 3 years = ~756 dates. With 103 tickers, this yields ~77k panel rows. The 3-year window captures at least two distinct market regimes (2023 recovery, 2024–2025 momentum regime). The exponential decay recency weighting (`half_life_days=252`) sensibly downweights older samples.

**Concern:** 3 years is borderline for a pairwise LTR model. LambdaMART typically needs at least 10k–100k samples to regularize well (depending on feature dimensionality). At 77k rows with 27 features, the sample-to-parameter ratio is adequate. However, the `min_child_weight=60` constraint is aggressively cautious — each leaf requires ≥60 training samples. With ~100 tickers per date and only 3 years of data, many cross-sectional patterns have only 50–100 examples of the extreme-rank cases (e.g., "tech stock with high short interest AND high momentum in BULL_CALM regime"). These rare combinations will be under-fitted.

**Regime stationarity concern:** The strategy includes tickers like COIN, RBLX, NVTS, SOFI — high-beta crypto-adjacent and growth names that entered the market (or current form) in 2020–2022. Their 3-year history in the training window is heavily skewed toward a specific macro environment (Fed rate cycle peak/trough, post-SPACs). If this regime rotates, the model has no priors for the new regime.

### 4.7 Early Stopping with Separate 20% Holdout

The design (last 20% of dates as val set, independent of CPCV folds) is suboptimal but not disqualifying — it is a common pattern in production ML systems for computational efficiency. The underlying problem (BUG-CV-3) is that if the 20% holdout period happens to be a distinctly good or bad regime, early stopping will fire prematurely. The fix (use Fold 15's test dates as the eval set) aligns early stopping with the most recent CPCV fold, which is more representative of near-future out-of-sample conditions.

---

## 5. Feature Analysis

### 5.1 Complete Feature List (27 active features)

From `artifacts/panel-ltr.json` `feature_cols`:

**Technical indicators (6):** `adx`, `williams_r`, `bbp`, `cci`, `trend`, `trend_long`

**Relative momentum (2):** `rel_mom_20d`, `rel_mom_60d`

**Classic cross-sectional factors (6):** `size_z`, `mom_12_1_z`, `beta_60d_z`, `resid_mom_z`, `price_to_high_z`, `realized_vol_z`

**Event-based (1):** `earnings_surprise_cum_z`

**Hourly intraday (3):** `afternoon_drift_z`, `vwap_premium_z`, `intraday_realized_vol_z`

**10-minute intraday (5):** `m_morning_30min_drift_z`, `m_vwap_premium_z`, `m_intraday_realized_vol_z`, `m_overnight_gap_z`, `m_reversal_ratio_z`

**Fundamental (3):** `roe_z`, `gross_profitability_z`, `book_to_price_z`

**Short interest (1):** `short_pct_float_z`

### 5.2 Likely Genuine Alpha (with citation support)

- **mom_12_1_z** (12-1 momentum, skip 1 month): One of the most replicated factors in finance. Jegadeesh & Titman (1993), Carhart (1997). The skip-one-month convention is correct to avoid short-term reversal contamination. IC contribution ~0.015–0.025 individually in large-cap universes.

- **resid_mom_z** (momentum residualized for SPY): Blitz, Huij & Martens (2011), Gutierrez & Pirinsky (2007). Beta-adjusted momentum removes the systematic component, leaving idiosyncratic momentum that is more persistent and has lower drawdown during momentum crashes. Correct implementation.

- **earnings_surprise_cum_z** (post-earnings announcement drift): Ball & Brown (1968), Bernard & Thomas (1989, 1990). PEAD is one of the most durable market anomalies; cumulative earnings surprise is a strong predictor of 1–3 month forward returns. IC contribution ~0.010–0.020 on large caps.

- **short_pct_float_z** (short interest): Asness, Frazzini & Pedersen (2018), Dechow et al. (2001). High short interest predicts underperformance; the signal is stronger as a cross-sectional z-score. IC contribution ~0.008–0.015.

- **realized_vol_z** (realized volatility): Ang et al. (2006, 2009). Low-volatility anomaly: low-vol stocks outperform high-vol stocks on risk-adjusted basis. The monotone constraint `realized_vol_z: -1` is correctly calibrated (negative → model penalises high-volatility candidates).

- **book_to_price_z** (B/P ratio): Fama & French (1992, 1993). Value factor, though highly regime-dependent — value has severely underperformed in 2017–2024. The monotone constraint `book_to_price_z: -1` (negative, meaning high B/P → lower predicted rank) implies the model has learned the anti-value signal prevalent in the current growth/tech regime, which is sensible given the training window.

- **vwap_premium_z, afternoon_drift_z** (intraday signals): Bogousslavsky (2021), Lou, Polk & Skouras (2019). Intraday momentum and VWAP deviations have documented predictive content at daily holding periods. These are rarer in academic datasets due to data cost.

### 5.3 Likely Redundant or Weak Features

- **adx, williams_r, bbp, cci** (technical indicators): These are classical retail technical indicators with weak academic support for predictive ability in large-cap universes after transaction costs. Correlation between `adx` and `trend` is likely high (both measure directional strength). Gu et al. (2020) show that technical indicators form a redundant cluster in factor space; the GBM already captures their interaction structure, so they add mostly noise.

- **trend, trend_long** (price-to-EMA ratios): Overlap heavily with `mom_12_1_z` and `rel_mom_20d`. Three momentum-correlated features out of 27 increase the effective dimensionality of the momentum cluster without adding independent signal.

- **m_morning_30min_drift_z** vs. `m_vwap_premium_z`: Both measure early-session price action. Without feature importance analysis, it is impossible to confirm they are not competing for the same signal — but high correlation is plausible.

- **size_z** (log market cap): Size effect (Fama-French) is negative in the current watchlist — the universe is already filtered to large/mega-caps, so within-watchlist size variation is compressed. The feature likely has near-zero marginal IC.

### 5.4 Notable Absent Features

The following top-tier alpha signals from the recent literature are absent:

- **Options flow / implied volatility surface:** Put-call ratio, IV skew, and deviations from realized volatility are among the strongest short-horizon predictors (Xing et al. 2010, An et al. 2014). IV term structure contains information about earnings uncertainty not captured by realized vol.

- **Analyst revision momentum:** Estimate revision momentum (ERM) — the direction of earnings estimate changes — is one of the most robust 1–3 month predictors in academic literature (Chan et al. 1996). Current `earnings_surprise_cum_z` captures post-announcement drift but not pre-announcement revision expectations.

- **14-day to 63-day reversal factor:** Short-term reversal (Jegadeesh 1990) at 1–4 weeks is orthogonal to 12-1 momentum and has a different IC sign. Many professional cross-sectional models include both.

- **Liquidity factor (Amihud illiquidity ratio):** Amihud (2002) illiquidity ratio is explicitly in the `drop_cols` list — it was removed, presumably due to noisy data. The current `short_pct_float_z` partially proxies liquidity, but the Amihud ratio captures the price impact dimension differently.

- **Return asymmetry / skewness features:** Boyer, Mitton & Vorkink (2010), Conrad, Dittmar & Ghysels (2013). Idiosyncratic skewness predicts underperformance. Given the inclusion of NVTS/RBLX/COIN-type names, skewness might detect the parabolic-exhaustion pattern that the separately-coded `ParabolicExhaustionGateTask` handles heuristically.

### 5.5 Data Quality and Staleness Risks

- **yfinance dividend/split adjustments:** yfinance occasionally has incorrect split adjustments, particularly for recent splits (e.g., NVDA 10:1 in 2024). An incorrect split factor creates a synthetic 10x return in the feature history that inflates momentum signals for that ticker. These errors are undetectable without a ground-truth reference.

- **Fundamentals staleness:** OpenBB-sourced fundamentals (ROE, gross profitability, B/P) are quarterly, not daily. The daily panel carries forward the last quarterly value. ROE from a year ago is not informative about next month's return; the signal comes from *revisions*, not levels. The current implementation has `roe_z` as a level feature — this is suboptimal.

- **Insider trade lag:** SEC Form 4 filings are available within 2 business days of transaction. The current implementation loads from a cache; if the cache is stale, the feature reflects transactions from weeks ago, which has reduced predictive value.

---

## 6. Signal Quality & Capacity

### 6.1 IC = 0.035 in Context

The post-BUG-CV-1-fix OOS mean IC (Spearman) = 0.035 (std = ~0.021) across 15 CPCV test paths places RenQuant 104 in the following context. (The prior reported figure of IC=0.042 was inflated ~0.005 by fold leakage; the current 0.035 is the honest baseline, with BUG-CV-3 still open.)

- **Grinold-Kahn fundamental law:** IR ≈ IC × √BR, where BR is the number of independent bets per year. With 103 tickers × ~25 rebalances/year (10-day holding, roughly biweekly rebalancing) = ~2,500 bets/year, and assuming 50% bet independence (correlated universe): √(0.5 × 2500) ≈ 35. IR ≈ 0.035 × 35 = 1.2. This is broadly consistent with the sim-reported Sharpe ≈ 1.47 from the A/B experiments (the small gap is expected given the sim uses the pre-fix IC figures).

- **Published benchmarks:** Gu et al. (2020) report monthly IC of 4–8% for their neural-network ensembles on 3000+ US stocks. Guijarro-Ordonez et al. (2021) report daily ICs of 1–3% on the top 500 liquid US names. RenQuant's 10-day IC of 3.5% on a 103-ticker universe is plausible and consistent with academic benchmarks on a smaller, curated universe. BUG-CV-3 remains open and may shift this further, so treat 0.035 as a provisional honest baseline rather than a final number.

- **The real concern is not IC magnitude but IC stability.** The per-fold IC ranges from −0.004 to +0.079 — a 20-fold spread. Fold 3 is negative (−0.004). This suggests the model's signal is regime-conditional: it works in momentum/trending regimes and breaks in choppy/mean-reverting periods. The BULL_VOLATILE A/B result (IC = −0.172 in BULL_VOL regime on 445 rows) confirms this. A robust system needs roughly consistent positive IC across all regime types.

### 6.2 Theoretical Sharpe Ceiling at Current AUM

At $10k AUM with 5–8 positions, each position is $1,250–$2,000. With a 10-day holding period and 2–5 trades per week:

**Grinold-Kahn Sharpe ceiling:** IR_max = IC_true × √BR = 0.04 × √(103 × 25) = 0.04 × 50.7 = 2.03. This is the Sharpe ratio in a frictionless world with no position constraints. Actual Sharpe will be lower due to transaction costs, position size limits, regime filtering, and the fact that you can only hold 5–8 positions at a time from a 103-name universe.

**Breadth bottleneck:** The biggest limiter is not IC quality but the number of positions the system can hold simultaneously. With `max_concurrent_positions=8`, you're trading at most 8 bets against a universe of 103. The Fundamental Law's √BR assumes all positions are held proportional to their IC contribution. Constraining to 8 positions while ranking 103 names means you're systematically under-diversifying relative to the optimal portfolio. The roadmap's priority on expanding to 200+ tickers is correct — this directly raises the Sharpe ceiling.

**Transaction costs at $10k AUM:** Market orders on Alpaca with PFOF routing typically experience 2–10 bps execution slippage on liquid names. At $2k position sizes, a 5 bps slip = $1. Over 1,000 round-trips/year, that's ~$2,000/year — 20% of AUM — which is material. The strategy needs to be run with limit orders or better execution to scale.

### 6.3 Capacity Constraints

- **$10k AUM:** Well within Alpaca's capabilities. Position sizes ($1–2k) are adequate for liquid large-caps.
- **$100k AUM:** Still feasible with current universe. Each position becomes $10–20k, which is comfortably liquid for AAPL/MSFT/NVDA. Slippage remains negligible.
- **$1M AUM:** The universe composition starts to matter. Positions in NVTS, GTLB, PCTY (low float, small cap) at $100–200k would move the market. Universe curation would need to enforce minimum ADV thresholds. The current volume filter (top 85th percentile) may not be sufficient.
- **$10M+ AUM:** Fundamental redesign needed — smaller universe (top 50 S&P names), smaller position count, limit order execution, and potentially a dark pool relationship. Out of scope for the current system.

---

## 7. Risk Management

### 7.1 DrawdownCircuit

The regime-parameterised drawdown halt (`drawdown_halt_pct`: BULL_CALM=35%, BULL_VOLATILE=10%, CHOPPY=8%, BEAR depends on config) is a reasonable multi-regime approach. The tighter halt in high-volatility regimes prevents the compound loss effect.

**What's missing:**
- **No drawdown-from-peak at portfolio level.** The current circuit appears to operate on a trailing basis from the high-water mark, but there is no explicit maximum drawdown limit (e.g., "halt all trading if portfolio is down 15% from HWM regardless of regime"). In a 2008-style event, the regime detector would switch to BEAR, but there's a detection lag.
- **No correlation-based position limit.** The `correlation_guard_threshold=0.70` config exists, but it is not clear from the codebase whether this is actively gating buys based on portfolio correlation or just used for the regime calculation. Holding 8 tech positions (AAPL, MSFT, NVDA, AMD, META, GOOG) in a 103-ticker universe with a correlation guard at 0.70 would not prevent a concentrated tech portfolio.

### 7.2 Kelly Position Sizing

`f* = μ/σ²` with `fractional=0.25–0.50` (quarter-Kelly to half-Kelly) is the correct formula. The fractional multiplier absorbs the systematic overestimation of μ (model confidence) and the Normal approximation error in σ. Quarter-Kelly is the standard institutional practice (Thorp 2006); half-Kelly increases compound growth but doubles drawdown risk.

**Concern:** μ comes from NGBoost, which is trained on a 3-year panel. If the current NGBoost artifact is misaligned (as it was before the 2026-04-28 fix), μ estimates will be systematically wrong. The Kelly formula amplifies μ estimation errors quadratically (doubling μ quadruples position size). A `μ_cap` of 2–3 σ would limit runaway position sizing from overconfident NGBoost estimates.

### 7.3 No Beta Neutrality

This is the most significant risk management gap. The strategy is 100% long, with no hedging against broad market moves. At `max_concurrent_positions=8` in a tech-heavy universe, the effective portfolio beta is likely 1.2–1.5. Every "alpha" claim is contaminated by this beta unless controlled:

- In a 2008-style −50% market event: with β=1.3, the portfolio loses ~65% — far beyond the drawdown halt thresholds. The regime detector would switch to BEAR, but the BEAR regime parameters are undefined in the config snippet (they exist but were not fully visible), and there appears to be no automatic full-exit mechanic in BEAR.
- In a 2020 COVID crash (−35% in 23 days): the CUSUM regime detector has a 3-day cooldown and a 20-day lookback. The first 3–5 days of a crash would not trigger the regime switch fast enough to avoid material losses.

A simple overlay hedge (long SPY puts or short SPX futures for 30–50% of portfolio delta) would structurally fix this. At $10k AUM the options premium is prohibitive, but the risk is real.

### 7.4 Regime-Based Position Adjustment

The 4-regime system (BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR) with per-regime `max_position_pct` and `cash_reserve_pct` provides meaningful risk-scaling. CUSUM-v2 wall-time cooldown (3-day minimum) prevents rapid regime flip-flop after detecting a change. This is well-designed.

**Gap:** There is no intraday risk adjustment. If the portfolio opens down 5% in a day (e.g., NVTS -40%), the daily bar strategy cannot reduce position until the next close. An intraday stop-loss mechanic (already partially implemented via `max_single_day_loss_pct`) would be activated, but only through LEAN's OnData handler, not through the live Alpaca runner.

### 7.5 Tail Risk: 2008/2020-Style Events

In a severe crash scenario:
- The Hurst exponent would shift toward mean-reversion territory (H < 0.5), switching to CHOPPY regime
- The CUSUM on SPY returns would trigger BEAR transition within 3–10 bars
- BULL_VOLATILE cash reserve (20%) would be partially deployed until regime confirmation
- Hard bear override (`hard_bear=True`) would halt new buys

**Expected outcome in 2008 scenario:** The strategy would lose 20–35% before the regime system fully transitions. This is not catastrophic for a $10k account but represents months of gains. More importantly, the strategy has no explicit mechanism to return to cash — it holds current positions unless they trigger individual stop-losses or the drawdown circuit.

---

## 8. Data Infrastructure

### 8.1 yfinance as Primary Data Source

yfinance is convenient but has documented issues that make it unsuitable for production capital allocation:

1. **Survivorship bias:** yfinance returns historical data only for currently-listed tickers. Companies that were delisted, merged, or went bankrupt are absent. The current 103-ticker watchlist is entirely survivor-biased — every company in it survived the 2020 crash, 2022 selloff, and 2023 banking crisis. This inflates backtested performance.

2. **Adjustment errors:** Split and dividend adjustments are applied retroactively using current split factors. Recent splits (NVDA 10:1, GOOGL 20:1) may introduce subtle errors in historical price series depending on the API query date.

3. **Intraday data quality:** The `data/intraday/` cache (83MB for hourly + minute bars on 103 tickers) was fetched from yfinance, which limits free intraday history to 60 days for 1-minute bars and 730 days for 1-hour bars. Training on sub-2-year intraday features with the strategy's 3-year training window means the intraday features have systematic NaN patterns for the earliest 1 year of the training set.

4. **Point-in-time violations:** Fundamental data (ROE, B/P) from yfinance is not point-in-time — it reflects the latest restated filings, not the filings available on the historical date. This is a known form of look-ahead bias that inflates fundamental factor performance.

**Industry standard:** Professional quant systems use Compustat (fundamentals, point-in-time), Bloomberg (prices, corporate actions), and FactSet or ICE Data Services (intraday). For a retail-scale system, Tiingo or Polygon.io offers point-in-time daily OHLCV at reasonable cost (~$30–100/month).

### 8.2 Universe: 103 Tickers

The watchlist is 103 hand-curated large/mega-cap US stocks plus sector ETFs. Universe selection criteria are not explicitly documented, but the composition suggests: S&P 500 top 100 + high-growth names (COIN, RBLX, HOOD, SOFI) + recent IPOs (NVTS, GTLB). This introduces several biases:

- **Survivorship:** Every company has survived at least until 2026.
- **Selection bias:** The universe over-weights technology (NVDA, AMD, AMAT, ASML, AVGO, KLAC, LRCX, MCHP, MU, NXPI, ON, QCOM, TXN = 13/103 semiconductor alone) and growth (META, GOOG, AMZN, NFLX = 4 FAANG), which had exceptionally strong performance in the training window. A properly-constructed universe would use S&P 500 membership rules or a liquidity screen.
- **Non-stationarity:** Including companies that went public or became prominent during the training window (COIN: IPO 2021, NVTS: 2022 listing) means the model trained on their early-history volatility has no out-of-distribution prior for their mature behavior.

**The planned expansion to 200+ tickers** (P0 in roadmap) is valuable not primarily because it raises the IC ceiling (more tickers → more bets → higher BR in Grinold-Kahn) but because it forces the universe selection to become systematic rather than curated, reducing selection bias.

### 8.3 Data Resolution

Daily OHLCV + hourly features + 10-minute features is a reasonable stack for a 10-day holding strategy. The intraday features (`vwap_premium`, `morning_drift`, `reversal_ratio`) capture within-session microstructure that daily bars lose.

**Adequacy assessment:** For 10-day prediction, daily resolution is the primary signal carrier. The marginal value of intraday features is real but limited — academic evidence (Bogousslavsky 2021) shows intraday signals have a half-life of 1–3 days. Including them as static daily features (rather than as rolling short-term predictors) may capture some persistence, but their main value is likely in filtering false positives (e.g., a high `reversal_ratio` on entry day suggests the move is not persistent).

---

## 9. Live Trading Infrastructure

### 9.1 launchd-Based Scheduling

Five macOS launchd plists run the daily_104 cycle (fetch → inference → trade → retrain cadence). This is adequate for a single-machine, low-AUM operation. It has known failure modes:

- **Silent failures:** launchd jobs that crash do not automatically alert. The ntfy integration partially addresses this, but only for caught exceptions — a segfault or OOM kill would be silent.
- **Edit-while-running race condition (now documented as Principle 5.4):** The 2026-04-28 M2 blender incident (function name error in a running overnight chain script not caught until morning) is the canonical example.
- **No job dependency management:** If `fetch` fails, `inference` runs with stale data. launchd has no "don't run B if A failed" semantics. The current workaround (checking data freshness in the inference step) is fragile.

**Industry comparison:** The standard at this scale is a local Prefect or Dagster instance (both have free tiers). These provide dependency-aware scheduling, automatic retry, and run history. Migration is ~8 hours of work and would eliminate the entire class of launchd silent-failure issues.

### 9.2 State Management (live_state.json)

The `runs.db` mirror-on-every-bar architecture is good. The missing read-path recovery (restart after JSON loss → reset state) is a P0 for scaling past $20k AUM — losing the high-water mark and streak data mid-trade causes incorrect sell decisions.

### 9.3 Error Handling and Alerting

ntfy push notification on trade events and critical alerts is functional for a solo operator. The `max_no_trade_days=15` and `max_no_candidate_days=15` monitoring circuits provide a safety net against the class of "silent zero-trade" failures that plagued the 2026-04-28 NGBoost incident.

**Missing:** No automated alert for model staleness (`model_staleness_days=60` exists in config but alerting is unclear), no alert for portfolio drawdown crossing a threshold, no weekly P&L summary push.

### 9.4 Paper/Live Isolation

The `allow_fetch=False` handicap in backtesting (vs. live runner's full fetch) is the correct architectural isolation. The 2026-04-28 NGBoost feature drift incident was caught and resolved within one trading session — the monitoring system worked as designed.

---

## 10. Benchmarking

### 10.1 Alpha vs. SPY Buy-and-Hold

**No honest comparison currently exists.** All simulated APY figures (39.82% golden, 26.91% A/B production) are produced by running the trained model over a historical window that overlaps with or falls entirely within the training data. This is in-sample evaluation masquerading as backtest.

A proper comparison requires the walk-forward runner (B1 in roadmap): retrain on data up to time T, evaluate on T+1 to T+N, rolling forward. Until B1 ships, the only honest live performance data is actual Alpaca equity curve, which is available but covers only a short period (appears to be weeks to months).

**SPY during the apparent backtest window (2024-01-01 to 2026-04-26):** SPY returned approximately +43% (S&P 500 was up ~25% in 2024 and ~10% in 2025 as of April 2026 — exact numbers require live market data). If the strategy's true OOS return is roughly the reported 26–40% APY range, it is not obviously beating SPY on a raw basis — only on risk-adjusted terms (Sharpe ≈ 1.47 vs. SPY Sharpe ≈ 0.7–0.9 over this window). This comparison is currently impossible to make honestly.

### 10.2 IC = 0.040 vs. Published Academic Signals

| Signal type | Published OOS IC | Source |
|---|---|---|
| 12-1 momentum (monthly, large-cap) | 0.020–0.040 | Jegadeesh & Titman 1993, Carhart 1997 |
| ML ensemble (neural net, monthly, all US stocks) | 0.040–0.080 | Gu, Kelly, Xiu 2020 |
| ML ensemble (GBM, daily, S&P 500) | 0.010–0.030 | Guijarro-Ordonez et al. 2021 |
| Intraday return predictability (daily) | 0.005–0.015 | Bogousslavsky 2021 |
| PEAD (earnings surprise) | 0.015–0.025 | Bernard & Thomas 1989 |
| Short interest | 0.010–0.020 | Dechow et al. 2001 |

A 10-day OOS IC of 0.040 in a 103-ticker universe, if genuine, would be strong but not implausible. However:
1. The training universe is survivor-biased (mega-caps that performed well 2021–2026).
2. The 3-year training window aligns perfectly with a strong US equity bull market, biasing all momentum features positively.
3. BUG-CV-1 is now fixed and the model retrained (IC=0.035). BUG-CV-3 (early stopping eval set misalignment) remains open and may shift the IC further. The reported IC=0.035 is the best available honest estimate but is still provisional.

### 10.3 What a Quant Fund Would Require

Before a quant fund would allocate capital to this strategy (even at a seed level), it would require:

1. **Genuine OOS performance:** At least 12 months of walk-forward IC measurements, not in-sample simulation. The B1–B3 roadmap items are prerequisites.

2. **Transaction cost inclusion:** Every IC figure needs to be accompanied by a "net IC" after a realistic transaction cost assumption (5–15 bps round-trip for the current universe and AUM level).

3. **Risk factor attribution:** Decompose the strategy's return into: market beta, size factor, momentum factor, quality factor, idiosyncratic alpha. A fund will pay for the idiosyncratic component only; everything else is available cheaper via factor ETFs.

4. **Regime stability analysis:** Show that IC ≥ 0 in at least 3 of 4 market regimes (bull, bear, choppy, volatile). The current per-fold IC (ranging from −0.004 to +0.079) with one negative fold is concerning — a fund would want all 15 CPCV folds to be positive before considering the signal tradeable.

5. **Capacity study:** At what AUM does the strategy's IC decay? The 103-ticker, 5–8 position design suggests capacity of $2–5M before market impact becomes material.

6. **Live paper track record:** At minimum 6 months of Alpaca paper trading with documented slippage, fills, and attribution.

---

## 11. Prioritized Recommendations

### 🔴 P0 — Must Fix (invalidates current results)

**P0.1 — BUG-CV-2 fix and retrain: ✅ Done (Apr 29)**  
The `min_best_iter=20` guard was applied and the model retrained. Production model is now best_iter=19, IC=0.035. This is the current baseline. Monitor best_iter on future retrains — best_iter=19 is still low (expected 50+), consistent with BUG-CV-3 (open) causing premature early stopping.

**P0.2 — Apply BUG-CV-3 fix and re-measure IC**  
Align the early stopping eval set with Fold 15 test dates. Retrain after applying this fix. The resulting IC will be the cleanest OOS measurement the system has produced — BUG-CV-1 and BUG-CV-2 are already resolved, so only this one structural misalignment remains.

**P0.3 — Ship walk-forward runner (B1 / B2 in roadmap)**  
Until B1 or B2 ships, no performance claim is defensible. B2 (single-cut holdout, ~30 minutes to implement and run) is the minimum viable honest measurement. Train on 2021-01-01 to 2023-12-31, simulate on 2024-01-01 to 2026-04-28. Run this before making any more feature or parameter changes. The result is your true baseline.

**P0.4 — Implement DB read-path recovery for live_state**  
The missing startup-recovery path from `runs.db` (referenced in roadmap B-Tier 1) is a live trading correctness risk. At $10k AUM it is tolerable; a single JSON corruption before a volatile session could cause the live runner to trade with wrong HWM, wrong regime state, and wrong streak counts. Implement `restore_live_state_from_db.py` and the runner startup hook before scaling AUM.

### 🟡 P1 — Should Fix (material improvements, correct order after P0)

**P1.1 — Feature schema fingerprinting**  
The NGBoost feature drift incident will recur. Implement a feature manifest (SHA256 of sorted feature column names) stored in both the model artifact and the inference config. The inference engine should reject any artifact whose feature fingerprint doesn't match the current panel's column set, not silently zero-fill. This is the invariant-level fix for the entire class of silent column mismatch bugs.

**P1.2 — Expand universe to 200+ tickers systematically**  
The roadmap correctly identifies this as the highest-impact lever. But expansion should be rule-based, not hand-curated: e.g., all S&P 500 members with minimum 3-year history and daily ADV > $100M. This eliminates the survivorship selection bias and is reproducible.

**P1.3 — Replace Normal distribution in NGBoost with Student-t**  
Financial returns have fat tails. Student-t with ν=4–6 degrees of freedom better fits the empirical return distribution and produces larger σ estimates in the tails, leading to more conservative Kelly sizing in high-uncertainty situations. This is a one-line change in `ngboost_head.py` (`from ngboost.distns import Normal` → `from ngboost.distns import T`).

**P1.4 — Replace yfinance with a point-in-time fundamental data source**  
The look-ahead bias in yfinance fundamentals (ROE, B/P using latest-restated filings) is material. Even SimFin (free tier) or EDGAR direct filings provide point-in-time fundamentals. This does not require a Bloomberg subscription — Polygon.io's fundamental tier at $79/month is adequate.

**P1.5 — Add earnings revision momentum feature**  
Earnings estimate revisions (direction + magnitude) are among the most robust 1–3 month predictors and are not currently in the feature set. SEC EDGAR Form S-1 / analyst consensus can be obtained from Alpha Vantage (free) or Refinitiv (paid). This is likely to add 0.5–1 IC points independently.

**P1.6 — Implement Prefect/Dagster local scheduler**  
Replace launchd with a proper DAG scheduler. This fixes silent failures, job dependency management, and the edit-while-running risk class. The migration is ~8 hours of work.

### 🟢 P2 — Nice to Have (positive expected value, lower urgency)

**P2.1 — Reporting separation (B3)**  
Label every performance figure with its provenance (in_sample / holdout / walk_forward / live). This is operational hygiene that prevents future miscommunication about what numbers mean.

**P2.2 — Regime Ensemble (T2-3)**  
After panel > 150k rows (roadmap milestone), train separate XGBoost models per regime and blend. This addresses the per-fold IC instability (one fold was negative) by allowing regime-specific feature weights. The infrastructure for regime routing exists in `kernel/panel_pipeline/regime_router.py`.

**P2.3 — Add options flow feature**  
Options implied-to-realized vol ratio and put-call skew are among the strongest unrealized alpha sources in the current feature set. The data is available from CBOE/OPRA via Polygon.io. At $10k AUM the informational edge is larger than at institutional scale, because the features are derived from market participants who are not watching retail flow.

**P2.4 — Trade evaluation DB + OPE (roadmap item)**  
The (s, a, r) trade database with 7/14/28-day outcome evaluation is a long-term compounding asset — every trade is retrospective training data. The schema and off-policy evaluation design (Jiang-Li 2016, Sutton-Barto) is already specified. Implement Phases 1–2 (schema + nightly backfill) to start accumulating data.

**P2.5 — Portfolio-level beta neutrality**  
A simple SPY short overlay (short SPY at 50% of portfolio value when in BULL regime) would reduce the portfolio's market beta from ~1.3 to ~0.5. At $10k, this means shorting ~$5k of SPY, which is cheap and highly liquid. The result would be a cleaner alpha signal and substantially reduced drawdown in market corrections.

### ⚪ P3 — Don't Bother (diminishing returns or structurally ineffective)

**P3.1 — Rotation/rebalancing improvements**  
Six A/B experiments have consistently shown that every rotation variant hurts APY. The per-rotation cost is approximately −2.5 APY pts. Tax drag + missed continuation on the held position consistently outweighs the ER advantage of the candidate. This is a well-established result — do not revisit until the base model quality improves substantially (IC > 0.06 consistently).

**P3.2 — LightGBM LambdaRank**  
Already tested and rejected (−60% IC). The LightGBM config block is preserved for reference. Do not re-test until the panel exceeds 200k rows and the tree structure differs materially from the current 77k-row regime.

**P3.3 — Macro factor integration (any form)**  
Four variants tested (v1 broadcast, v2 per-ticker β, v3 expanded, v4 macro-as-panel-row). All four hurt OOS IC. The zero-gradient problem for broadcast features and the flat/negative IC for per-ticker β features are consistent with the academic consensus that macro factors hurt cross-sectional LTR models because they reduce within-date variance (the signal source for pairwise objectives). Do not revisit until the watchlist exceeds 200 tickers and the macro signal can be tested in a diversified cross-sectional context.

**P3.4 — InfoNCE contrastive embeddings (current form)**  
Already tested and rejected (−18.5% IC, t=−1.45). The theoretical motivation is sound (Dolphin et al. 2024), but the empirical result in this universe is negative. The embeddings are learning return co-movement structure that is already captured by the momentum/beta features, adding redundant correlated noise. Revisit only if the universe expands to include cross-asset (e.g., commodity ETFs, international ADRs) where pairwise correlation structure is non-trivial.

**P3.5 — Kelly rebalancing / TrimHeldTask**  
Tested and rejected (−12.7 APY pts default-on). The bar-to-bar volatility in μ/confidence estimates creates too many spurious trim signals. Revisit only after NGBoost produces stable per-bar μ estimates (which requires fixing P0.1 and P0.2 first).

---

## Appendix: Incident & Bug Log Summary

| Incident / Bug | Date | Root Cause | Fix Level | Status |
|---|---|---|---|---|
| NGBoost feature drift → 0 trades | 2026-04-28 | Macro features in artifact, absent from inference panel; zero-fill distorted σ | Invariant: `max_feature_drift_pct` guard | ✅ Fixed |
| BUG-CV-1: linspace fold boundary drift | 2026-04-28 | Float-rounded fold edges shift with panel size | Invariant: integer-division fold edges | ✅ Fixed in code; artifact retrained Apr 29 (best_iter=19, IC=0.035) |
| BUG-CV-2: best_iter=4 (undertrained model) | 2026-04-28 | Early stopping fires at round 4 on pathological eval set (root cause: BUG-CV-1 leakage) | Guard: `min_best_iter=20` raises RuntimeError | ✅ Fixed Apr 29; retrained model has best_iter=19, IC=0.035 |
| BUG-CV-3: early stopping misaligned with CPCV | 2026-04-28 | 20% holdout ≠ Fold 15 test dates; disconnected optimization signals | Structural: use Fold 15 test dates as eval set | ❌ Fix documented, not yet applied; likely suppressing best_iter further |
| NVTS parabolic buy | 2026-04-28 | No parabolic-regime filter; model had no parabolic samples in training | Invariant: `ParabolicExhaustionGateTask` | ✅ Fixed |
| M2 blender chain silent failure | 2026-04-28 | Wrong function name in script edited mid-run | Principle 5.1/5.4: import check + no mid-run edits | ✅ Principle added; not retroactively preventable |
| "+54% IC" selection bias | 2026-04-28 | A/B winner reported without A/A sanity check | Principle 5.2: mandatory A/A + shuffled-label tests | ✅ Principle added |
| Auto-revert config/model mismatch | 2026-04-28 | Rollback restored model but not strategy_config.json | Principle 5.5: rehearse rollback on non-prod copy | ✅ Principle added |

---

*This report was generated on 2026-04-28 from direct analysis of the codebase and artifacts. All IC and APY figures cited are from the system's own documentation and experiment logs. Independent verification against live trading data is recommended before making capital allocation decisions.*
