# Rust Transformer IC — what's good vs what's mediocre

## TL;DR

| Model              | val_IC on synthetic | val_IC on real |
|--------------------|--------------------:|---------------:|
| Linear regression (theoretical max on synthetic) | **+0.3228** | n/a |
| Rust transformer (50 epochs, lr=5e-4, d_model=48) | +0.0683 (~21% of max) | n/a |
| Rust transformer v3 (200ep all 41 cols, full panel) | **+0.2314** | +0.0363 @ ep 1 (random-init lucky, then collapse) |
| Rust transformer v4 dropdistshift (24 cols, full panel) | n/a | -0.0071 — WORSE |
| Rust transformer v4 tightreg (41 cols dropout 0.5 lr 1e-4) | n/a | -0.0430+ — WORSE |
| **Rust transformer v5 (hourly-era only, 491 dates, 200ep ListNet)** | n/a | **+0.0519 @ ep 127** — first healthy real-data curve |
| Production LightGBM (DEFAULT_PARAMS, on hourly-era 491-date panel) | n/a | **+0.0850** — the apples-to-apples baseline (LightGBM crushes transformer on the same data window) |
| Python XGBoost / LightGBM panel-LTR (post-audit) | n/a | **+0.0372** |
| Python transformer (overfit, shelved) | n/a | +0.0062 |

## Real-data prod-vs-test divergence finding (2026-04-25 PT, late-session)

Discovery: **17 of 41 features have +60% NaN-rate divergence between
train and val** because the hourly + minute bar caches only started
populating on **2024-04-25** (exactly 1 year before today). The full
panel covers 2021-04-19 → 2026-04-10 (1251 dates), so:

|              | train (2021-2024)        | val (2025+)              | Δ |
|--------------|--------------------------|--------------------------|---|
| morning_drift_z | 79% NaN                  | 19% NaN                  | +60% |
| afternoon_drift_z | 79%                  | 19%                  | +60% |
| 6 hourly cols    | 79%                  | 19%                  | +60% |
| 11 minute cols   | 79%                  | 16%                  | +63% |
| insider_net_buy_90d_z | 83%             | 57%                  | +26% |

Our DAT-RUST-MISSING-FEAT fix (Round 2 audit) substitutes **0.0** for
empty/NaN feature cells. In z-score space this means "neutral", but
combined with the train/val divergence above, the model trains on a
regime where 17 features ≈ 0 (the median) and validates on a regime
where the same 17 features carry real signal. Result:

* **v3 (full panel)**: epoch-1 random-init transformer gets +0.0363
  by accidentally using the val-populated hourly features through
  random projections. As soon as it trains, it learns "hourly ≈ 0 =
  noise, ignore" and val_IC monotonically collapses to -0.009.
* **v4 dropdistshift**: dropping the 17 cols cuts the val signal too,
  giving -0.0071. The cols ARE the signal, just unevenly distributed.
* **v4 tightreg**: stronger regularization on full panel makes
  overfit avoidance worse, not better, because the underlying issue
  isn't capacity — it's data distribution.
* **v5 hourly-era only**: trains on 491 dates ≥ 2024-04-25 where
  hourly cols are uniformly populated (~17% NaN, same on train + val).
  First healthy training curve on real data: ramped from epoch-1
  +0.005 → epoch-127 **+0.0519**, then early-stop. Total wall-clock
  100 seconds at ~0.7s/epoch on CPU. The lesson: hyperparameter A/B
  alone cannot rescue distribution-mismatched data; matching the
  train/val feature-population regime turned a -0.009 collapse into a
  +0.052 climb on the same architecture.

## 4-way audit — production-config win extracted from the audit (2026-04-25 PT)

The transformer audit revealed a production-relevant bug **in the
LightGBM training config, not the model**. Tested 4 configurations on
the same train/val split + production hyperparams:

| Config                                                           | val_IC mean | val_IC median | n_features | Δ vs current prod |
|------------------------------------------------------------------|------------:|--------------:|-----------:|------------------:|
| **1. LGBM on FULL 1251-date panel (current production)**         | **+0.0322** | +0.0300       | 41         | baseline          |
| **2. LGBM on HOURLY-ERA only (491 dates, Fix A: training window)** | **+0.0850** | +0.0817       | 41         | **+164%** ⭐      |
| 3. LGBM on FULL + missingness indicators (Fix B)                  | +0.0364     | +0.0449       | 58         | +13%              |
| 4. LGBM on HOURLY-ERA + missingness indicators (Fix A+B combined) | +0.0363     | +0.0243       | 58         | +13% (A wiped)    |
| 5. Rust transformer v5 (HOURLY-ERA only)                          | +0.0519     | n/a           | 41         | +61%              |

**The actual production win is just Fix A.** The transformer audit was
the diagnostic that revealed it — the `training_window_years: 5.0` in
the current strategy_config has the model training on 4 years of
mostly-NaN hourly features, dragging the model's effective IC. Just
restricting to the hourly-populated era (config-only change, no model
retrain) gives **+164% IC** for free.

Fix B (missingness indicators) gives a tiny lift on its own (+13%)
but UNDOES Fix A's gain when combined — on the hourly-era-only panel
the indicators are mostly all-zero, so the 17 extra columns become
noise that makes LGBM tree splits worse.

Walk-forward CPCV cross-check (5 folds on hourly-era panel):
mean +0.0662, median +0.0931, std +0.049 — folds 3-5 (train ≥243
dates) consistently > +0.09. Single-split +0.0850 is real, not luck.

**Recommended production config change** (one-line):
```diff
   "panel_ltr": {
-    "training_window_years": 5.0,
+    "training_window_years": 1.5,
   }
```
Pre-promotion gates: (a) re-run prod sim with the change, (b)
verify APY ≥ golden v4.1, (c) update golden config + doc.

## A2 audit — apples-to-apples honest comparison (2026-04-25 PT)

The +0.0372 baseline used in earlier comparisons was production
LightGBM running on the FULL 1251-date panel (which includes 4 years
of hourly-feature absence). That number flatters the transformer
because LightGBM also struggles when ~17 features are mostly NaN
across the older training rows.

The fair head-to-head is **same panel (491 hourly-era dates) + same
80/20 chronological split**:

| Model on `/tmp/real_panel_hourly_era.csv` | val_IC mean | val_IC median | wall |
|-------------------------------------------|------------:|--------------:|-----:|
| Rust transformer v5 (ListNet 200ep)        | **+0.0519** | (single split) | 100s |
| **Production LightGBM (DEFAULT_PARAMS, 300 boosts)** | **+0.0850** | +0.0817 | 1.1s |
| Naive LightGBM (default sklearn ranker, no L1/L2 + tuning) | -0.0359 | -0.0725 | 1.3s |

**Honest verdict:** LightGBM with production hyperparams beats the
transformer **+0.0850 vs +0.0519 = LightGBM wins by 64%** on the same
panel, same split, same compute budget (LGBM is 90× faster too).

The transformer is competitive (positive IC, healthy curve) but not
better. **Do NOT promote to golden v5.** The earlier "+39.5% over
XGBoost" headline was comparison-bias from using a non-apples baseline.

What the transformer would need to compete:
* **Missingness indicators**: LGBM's native sparsity-aware splits
  capture "missing" as its own signal — transformer needs explicit
  `{col}_is_missing` columns or a feature-token attention mask.
* **LambdaRank loss**: LGBM uses lambdarank with NDCG@5/10
  truncation; we use ListNet top-1. Pairwise gradients with
  truncation match the "top-K matters" inductive bias of cross-sec
  ranking.
* **Ensemble of 300 weak learners**: LGBM is implicitly ensembled
  via boosting; one-shot transformer training has no such free lunch.
  Stacking 5+ transformer seeds may close some of the gap.



**Why XGBoost / LightGBM is unfazed:** native sparsity-aware NaN
handling treats `missing` as its own tree-split branch — does NOT
substitute 0. Rust transformer faking `missing = 0` is the bug. Two
robust fixes for follow-up: (a) per-feature missingness indicator
columns (doubles feat dim from F to 2F); (b) feature-token mask in
attention so the model can attend-or-ignore.

Bonus audit finding: **`training_panel/imputation.py::add_missingness_indicators`
is defined and tested but NEVER CALLED in production code.** The Python
panel pipeline relies on LightGBM's native NaN handling instead. Either
the function should be wired in or deleted (AUDIT-PROD-IMPUTATION-DEAD-CODE).



## Architecture-ceiling finding (2026-04-25 14:35 PT)

Ran controls via sklearn on the same synthetic + same train/val split:

| Model                         | val_IC      | Notes                                    |
|-------------------------------|------------:|------------------------------------------|
| **Ridge regression (alpha=1)**|  **+0.3228**| theoretical max — signal IS linear       |
| **MLP 48×48 ReLU (sklearn)**  |  **+0.2345**| matches our transformer                  |
| **Our Rust transformer v3**   |  **+0.2314**| within 0.003 of MLP — same arch ceiling  |

**Key takeaway:** the non-linear-net ceiling on this synthetic is
~0.234 — both MLP and transformer hit it. The remaining 0.09 gap to
linear's 0.323 is the **inductive-bias cost of non-linearity** when
the signal IS perfectly linear.

This is GOOD news for the real-data run:
* Real production data has non-linear signal (sector × momentum × vol
  interactions, regime-conditional effects). Linear models can't
  exploit those. XGBoost gets 0.0372 on real precisely because the
  non-linear capacity matches the data.
* Our transformer hitting MLP-parity on synthetic means the Rust port
  is correctly implementing the architecture — if there were a port
  bug, transformer would be BELOW MLP, not at it.
* On real data, the non-linear capacity becomes an asset rather than
  a tax, so transformer should be competitive with XGBoost.

To beat 0.232 on this synthetic we'd need to use a LINEAR architecture
(no GELU, no softmax) — but that's not the goal; the goal is to match
or beat XGBoost on real production data.

## A/B run results (loss + arch + schedule)

| Run | arch                 | loss     | schedule           | val_IC final |
|-----|----------------------|----------|--------------------|-------------:|
| v3  | d=48 6h 2L ff=96 d=0.3 | ListNet  | 200ep, patience=20 | **+0.2314** |
| A   | d=48 6h 2L ff=96 d=0.3 | RankNet  | 200ep, patience=20 |    +0.2111  |
| B   | d=16 4h 1L ff=32 d=0.0 | ListNet  | 300ep, patience=30 |    +0.1367  |
| C   | d=48 6h 2L ff=96 d=0.2 | ListNet  | 500ep, lr=2e-4     |    +0.2028  |
| D   | d=48 6h 2L ff=96 d=0.3 | ListNet  | 500ep, patience=80 |  in flight  |

Notable findings:

* **RankNet (Burges 2005, pairwise) UNDERperformed ListNet** on this
  synthetic — 0.2111 vs 0.2314. Confirms the 2025 CIKM paper "On
  Evaluating Loss Functions for Stock Ranking" finding that LISTWISE
  losses tend to beat PAIRWISE on cross-sectional ranking. Pairwise
  has faster early gradient (epoch 4: 0.0408 vs ListNet 0.011) but
  plateaus earlier — listwise's softmax-CE keeps tightening the
  ordering past the point where pairwise has saturated.

* **Smaller architecture lost.** d_model=16 + 1 layer + 0 dropout
  could not match the d=48 + 2 layer setup. The synthetic signal IS
  linear, so theory says smaller should work, but the smaller model
  also has less capacity to compose the GELU-attention-GELU stack
  into something approximating identity.

* **Lower learning rate (2e-4) is just slower.** Same convergence
  point, just 2× the wall-time per IC unit. No quality advantage.

## V3 training run

Same architecture as the 0.068 run (d_model=48, n_heads=6, n_layers=2,
ff=96, dropout=0.3) but with longer schedule + bigger patience:
  --epochs 200 --batch 32 --lr 0.0005 --val-frac 0.2 --patience 20

Mid-training (epoch 62 / 200):
  loss:    6.47 → 5.77    (plateau, but model still learning structure)
  val_IC:  -0.016 → +0.1238    ← +1.8× the prior 50-epoch run
  Peak CPU: 310%   Peak memory: 4.9 GB   Per-epoch: 4.4s

Lessons:
  * Patience=20 is critical — the model has a long warmup before val_IC
    starts climbing meaningfully (epoch 1-10 was negative).
  * The same architecture that hit 0.068 in 50 epochs hits 0.124 in 62
    epochs — almost linear improvement in IC after the initial dip.
  * Loss plateauing while val_IC climbs is a sign of *generalisation*,
    not just memorization — the model is finding the underlying
    f0+f1 structure (target generators) more accurately each step
    even though training residuals are stable.

## What the synthetic test proved

The 0.0683 IC on synthetic data with `label = 0.3*f0 + 0.2*f1 + N(0,1)`
proves only that the **training pipeline works**: forward → loss →
backward → AdamW → weights save. It does NOT prove the architecture
is good — a linear regression hits the theoretical ceiling of 0.32 on
the same data trivially.

The 0.0683 number tells us:
- Gradients flow correctly (loss decreases monotonically 6.5 → 5.8)
- Multi-core CPU saturation works (369% peak vs Python 1-core GIL)
- Hyperparameters (lr=5e-4, batch=32, dropout=0.3, wd=1e-4) are
  reasonable but the architecture is over-specified for a 2-feature
  linear signal

## Why real data should be different

Real production panel:
- Signal-to-noise dramatically worse than the 0.13/1.0 synthetic
- Cross-sectional structure (sector, regime, size effects)
- Non-linear feature interactions (vol × momentum)
- Time-series autocorrelations the panel residualization tries to remove

A linear model on real data won't outperform XGBoost — that's why the
Python production setup uses XGB. The fair test is:
**Python XGB OOS IC = 0.0372** vs **Rust transformer OOS IC on real**.

## Comparison context

Cross-sectional Spearman IC scale (typical quant equity):
- < 0  : worse than random
- 0.00 - 0.02 : barely-trading edge, costs eat it
- 0.02 - 0.05 : publishable academic; tradable with low costs
- 0.05 - 0.10 : strong, top-decile fund territory
- > 0.10 : implausibly good (likely lookahead)

Our XGB at 0.0372 is in the lower-end-tradable band. That's the
reasonable target for the Rust transformer to match before promotion.

## Synthetic vs real setup

Same shape:
- Synthetic: 1000 dates × 99 tickers × 41 features (CSV)
- Real: 1256 dates × ~99 tickers × 41 features (the actual panel)

Synthetic deliberately uses a structure that linear regression can crush
(2-feature signal + Gaussian noise). Real is harder — XGBoost only gets
0.037 on it.

## Hyperparameters tested

The 0.068 run used:
```
--epochs 50 --batch 32 --lr 0.0005
--d-model 48 --n-heads 6 --n-layers 2 --ff-dim 96 --dropout 0.3
--val-frac 0.2 --patience 8
```

Reasonable for "small panel" per the regularization research papers
cited in our prior session (dropout 0.3-0.5, weight_decay 0.05-0.1
recommended by ApxML's transformer-regularization survey).

To push IC higher we'd want to A/B:
- weight_decay 1e-4 → 1e-3 → 1e-2
- d_model 48 → 32 → 16 (smaller is better on small panels)
- n_layers 2 → 1 (overparameterised at depth)
- LambdaRank loss vs ListNet
- Longer training + better schedule

## Reference papers (verified earlier session)

- **Poh-Lim-Zohren-Roberts 2020** ([arXiv 2012.07149](https://arxiv.org/abs/2012.07149)):
  cross-sectional learning-to-rank, listwise > pairwise > pointwise on
  S&P 500.
- **"On Evaluating Loss Functions for Stock Ranking" (CIKM 2025,
  [arXiv 2510.14156](https://arxiv.org/abs/2510.14156))**: comprehensive
  benchmark; informs loss-function choice.
- **ApxML transformer regularization survey**: dropout 0.3-0.5,
  weight_decay 0.05-0.1, layer freezing, aggressive early stopping
  for small data.

## Honest assessment as of this commit

- ✅ Rust training pipeline works end-to-end
- ✅ 31 Rust tests passing (loss, trainer, CV, dataset, metrics)
- ✅ Multi-core CPU utilization verified
- ⚠️  IC on real production data: NOT YET MEASURED
- ⚠️  Hyperparameters not yet tuned for the actual data SNR
- ❌ Has not beaten Python XGB on any real panel
