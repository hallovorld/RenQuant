# Papers & Algorithms Implemented Across the Codebase

Catalogue of academic papers, algorithms, and engineering patterns wired into production code.

**Scope (2026-05-20 update)**: originally focused on the 2026-04-25 Rust transformer
sprint at `rust/transformer_scorer/` (legacy). Expanded to cover the broader stack
per CLAUDE.md §5.12 "default to canonical references":
- HF Trainer-based PatchTST shadow (`scripts/patchtst_hf.py`, shipped 2026-05-19)
- FiLM regime conditioning (Perez 2017)
- Margin Ranking loss (CIKM 2025 arXiv 2510.14156)
- Student-t multi-task head via `torch.distributions.StudentT`
- cvxportfolio / Almgren-Chriss / Ledoit-Wolf for portfolio QP
- HXZ 2020 "Replicating Anomalies" (DDV disable rationale)
- Kelly-Gu-Xiu 2020 RFS firm characteristics
- Bailey-López de Prado 2014 DSR + 2015 PBO/CSCV
- Box-Behnken 1960 / Plackett-Burman 1946 DOE methodology (§5.14)

**Honest framing**: implementing a method ≠ confirming empirical benefit. Each entry
notes status (PRODUCTION / SHADOW / REJECTED / EXPLORATORY).

## Architecture

### 1. Vaswani et al., 2017 — *Attention Is All You Need* ([arXiv 1706.03762](https://arxiv.org/abs/1706.03762))

The base transformer encoder block. Multi-head self-attention + position-wise
feed-forward + residual + layer norm.

* **Implemented in:** `rust/transformer_scorer/src/transformer_block.rs`
* **What we ported:** TransformerEncoderLayer with configurable `d_model`,
  `n_heads`, `feedforward_dim`, dropout, and GELU activation. Attention mask
  uses `where_cond` (not additive masking) to avoid `0 * -inf = NaN`
  propagation — verified with 8 unit tests.
* **What we did NOT port:** positional encoding. Cross-sectional ranking is
  permutation-equivariant — order of tickers within a date-group should not
  matter, so absolute position embeddings would inject noise.
* **Empirical:** matches the candle reference implementation tensor-for-tensor
  on identity inputs (poc_parity test).

## Loss functions

### 2. Cao et al., 2007 — *Learning to Rank: From Pairwise Approach to Listwise* (ICML 2007)

The ListNet top-1 listwise loss. Cross-entropy between softmax-normalized
predicted scores and softmax-normalized labels within each list.

* **Implemented in:** `rust/transformer_scorer/src/loss.rs`
* **What we ported:** Top-1 ListNet over date-groups. Each date is one "list";
  labels are residualized forward returns. Per-row valid-mask handles NaN labels.
* **Bug fixed (DIVERGENCE-PY):** original implementation took mean over all rows
  including masked ones, which biased the gradient. Fixed to take mean only over
  valid rows where ≥2 tickers have finite labels (matches Python panel-LTR
  semantics).
* **Empirical (synthetic):** ListNet val_IC = +0.2314, beats RankNet +0.2111
  on the same 99-ticker × 1000-date synthetic — confirms the 2025 CIKM finding
  (paper 4 below) that listwise > pairwise on cross-sectional ranking.

### 3. Burges et al., 2005 — *Learning to Rank Using Gradient Descent* (RankNet, ICML 2005)

The pairwise BCE loss with σ scaling. For every (i, j) pair with `y_i > y_j`,
penalize the model when `s_i < s_j` via softplus(-σ(s_i - s_j)).

* **Implemented in:** `rust/transformer_scorer/src/loss_pairwise.rs`
* **What we ported:** RankNet pairwise loss with σ=1.0, configurable via
  `LossKind::RankNet`. 5 unit tests cover identity (zero loss when scores
  match labels), monotonicity, and gradient direction.
* **Empirical (synthetic):** val_IC = +0.2111 on synthetic, slightly UNDERPERFORMING
  ListNet (+0.2314). Pairwise has faster early gradient (epoch 4: 0.041 vs ListNet
  0.011) but plateaus earlier — listwise's softmax-CE keeps tightening the
  ordering past where pairwise saturates.
* **Empirical (real):** not run head-to-head on real (we used ListNet for v3-v5).

### 4. *On Evaluating Loss Functions for Stock Ranking* (CIKM 2025, [arXiv 2510.14156](https://arxiv.org/abs/2510.14156))

Comprehensive benchmark of pointwise / pairwise / listwise losses on equity
panel ranking. Headline finding: **listwise losses tend to beat pairwise on
cross-sectional ranking**.

* **What we ported:** the experimental design — head-to-head ListNet vs RankNet
  on the same panel + same hyperparameters.
* **Empirical:** confirmed on synthetic (ListNet +0.2314 > RankNet +0.2111).

### 5. Poh, Lim, Zohren & Roberts, 2020 — *Building Cross-Sectional Systematic Strategies By Learning to Rank* ([arXiv 2012.07149](https://arxiv.org/abs/2012.07149))

Listwise > pairwise > pointwise on S&P 500 cross-section. Same conclusion as
paper 4 from a different angle.

* **What we ported:** the cross-sectional learning-to-rank formulation —
  groups = dates, items = tickers, scores per (date, ticker), Spearman IC as
  evaluation metric.
* **Implemented in:** `rust/transformer_scorer/src/metrics.rs::pooled_ic_owned`
  computes per-date Spearman IC then averages, matching the paper's protocol.

## Cross-validation

### 6. Marcos López de Prado, 2018 — *Advances in Financial Machine Learning*, ch. 7 (Combinatorial Purged K-Fold CV)

CPCV: split N dates into K groups, take all (K choose K-2) train/val combinations,
purge boundary samples that overlap the val window's lookahead.

* **Implemented in:** `rust/transformer_scorer/src/cv.rs`
* **What we ported:** group-level splitter that takes the date-axis and the
  lookahead horizon, returns the n_splits × (train_idx, val_idx) splits.
* **Bug fixed (CV-LINSPACE-OFF):** original integer split used `linspace`-style
  arithmetic that produced off-by-one boundary errors. Fixed to use unambiguous
  integer truncation: `(k as u128) * (n_dates as u128) / (self.n_splits as u128)`.
* **Empirical:** verified by `tests/test_round3_audit_fixes_2026_04_25.py::test_cv_linspace_fix`
  — CPCV with 15 splits on 1000 dates produces correct fold sizes (66/67) and
  no off-by-one boundary samples.

## IC metrics

### 7. Spearman, 1904 — *The Proof and Measurement of Association Between Two Things*

Rank correlation. Standard cross-sectional metric for quant panel models.

* **Implemented in:** `rust/transformer_scorer/src/metrics.rs::spearman_ic`
* **What we ported:** average-rank tie-breaking (matches scipy default).
  Per-date computation aggregated to pooled IC via simple mean.

### 8. Pearson, 1895 — *Note on Regression and Inheritance*

Linear correlation, used as the underlying primitive for Spearman after rank
transform.

* **Bug fixed (PEARSON-UNDERFLOW):** the naive denominator `sqrt(var_x * var_y)`
  underflows to zero when both stds < 1e-19, producing inf/NaN IC values that
  poison the early-stop logic. Fixed to clamp via `.max(f32::MIN_POSITIVE)`
  before sqrt.

## Regularization

### 9. Srivastava et al., 2014 — *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (JMLR)

Standard dropout in attention output and FFN.

* **Implemented in:** `rust/transformer_scorer/src/transformer_block.rs`
* **What we ported:** training-time-only dropout via candle's `dropout()`. At
  inference dropout is identity (eval mode).

### 10. ApxML transformer regularization survey (referenced in [doc/experiments/rust-transformer-ic.md](rust_transformer_ic_baseline.md))

Recommendations for small-data transformer training: dropout 0.3-0.5,
weight_decay 0.05-0.1, layer freezing, aggressive early stopping.

* **What we ported:** dropout default 0.3 (mid-range), weight_decay default
  1e-4 (we tested 1e-3 too — see v4 tightreg below), early stopping with
  configurable patience (default 5, used 20 for v3+).
* **Empirical:** weight_decay=1e-3 + dropout=0.5 made things WORSE on real
  data (v4_tightreg: -0.0430 val_IC). Lesson: regularization can't fix
  data-distribution mismatch, only capacity overfit.

### 11. Loshchilov & Hutter, 2019 — *Decoupled Weight Decay Regularization* (AdamW, ICLR)

The decoupled-weight-decay variant of Adam.

* **Implemented in:** `rust/transformer_scorer/src/trainer.rs`
* **What we ported:** AdamW via candle's `optim::AdamW`, with `weight_decay`
  configurable on the CLI.

## Numerical safety

### 12. Howard Hinnant — *date algorithms* (`days_from_civil`, [howardhinnant.github.io/date_algorithms.html](http://howardhinnant.github.io/date_algorithms.html))

Branchless conversion between calendar dates and days-since-1970, used to
encode panel dates as monotonic integers for the CPCV splitter.

* **Implemented in:** `rust/transformer_scorer/src/dataset.rs::days_from_civil`
  + `civil_from_days` (inverse, used by `train_panel.rs::chrono_today` to
  format the artifact's `trained_on` date without pulling in chrono crate).
* **Bug fixed (CHRONO-LEAP):** correctly handles 2024 (leap year) — verified
  by `tests/test_round3_audit_fixes_2026_04_25.py`.

### 13. Sparsity-aware NaN handling — *XGBoost* (Chen & Guestrin, 2016, KDD) / *LightGBM* (Ke et al., 2017, NeurIPS)

Tree models treat `NaN` as its own split branch — the model learns the
optimal default direction per split rather than imputing.

* **Did NOT port to transformer:** transformers can't do this natively;
  they need a fixed numeric value at every position.
* **What we did instead (DAT-RUST-MISSING-FEAT):** substitute 0.0 for empty/NaN
  feature cells (z-neutral median). Combined with the train/val NaN-rate
  divergence finding, this turned out to be the headline real-data bug — the
  0-fill makes the model train a "feature ≈ 0 = noise, ignore" detector.
* **What this revealed (Fix A finding, +164% IC for production LGBM):** the
  LIGHTGBM production config trains on a 5-year panel where 4 years have the
  hourly + minute features mostly NaN. Even with native NaN handling,
  LightGBM can't learn to use those features because 80% of train rows have
  them missing. Restricting `training_window_years: 5.0 → 1.5` (one-line
  config change) gives **+164% IC** on the same model.

## Engineering patterns

### 14. Atomic file write (`.tmp + rename`) — POSIX safe-write idiom

Prevents corrupted artifacts when training is interrupted mid-write.

* **Implemented in:** `rust/transformer_scorer/src/trainer.rs::save_safetensors`
* **What we ported:** write to `<path>.tmp` first, then `fs::rename` to the
  final path. POSIX guarantees rename is atomic for same-filesystem moves.

### 15. Frisch-Waugh-Lovell theorem — *The Partial Time Regressions As Compared with the Individual Trends*, Frisch & Waugh (1933) + Lovell (1963)

Residualize labels by regressing out market beta + sector before computing
forward returns, so the LTR target is "alpha vs cross-section" not "raw
return".

* **Implemented in:** Python panel-LTR (`backtesting/renquant_104/training_panel/labels.py`)
  — we did NOT re-port this to Rust because the panel CSV exporter applies
  it before the Rust trainer sees the data.
* **What we use:** the residualized labels from `/tmp/real_panel*.csv` are
  already FWL-orthogonalized + cross-sectionally Gaussianized.

### 16. Multi-core CPU saturation via candle BLAS + rayon

Parallel matmul (candle's mkl/openblas backend) + parallel epoch shuffle
(rayon's `par_iter`) saturate all cores during training without GIL.

* **Implemented across:** `trainer.rs::train_epoch`, `transformer_block.rs`
  (relies on candle's parallel matmul).
* **Empirical:** observed 310-417% CPU peak (on 14-core machine) during
  v3 training — vs Python's 100% (single-core, GIL-bound) for the same
  workload.

## Negative results (still papers, still implemented)

### 17. Missingness indicators — `add_missingness_indicators` (Python `training_panel/imputation.py`)

Append `{col}_is_missing ∈ {0,1}` per column so the model knows missing-ness
itself is informative.

* **Status in production:** function exists with tests but is NEVER CALLED
  (AUDIT-PROD-IMPUTATION-DEAD-CODE finding from this session).
* **What we tested in the 4-way audit:** adding missingness indicators
  on the full panel gives +13% IC. Adding them on the hourly-era-only panel
  WIPES OUT Fix A's +164% gain (drops back to +13%). Lesson: indicators
  are useful when there's distribution shift to flag, but on a
  population-uniform panel they're just 17 noise columns that confuse tree
  splits. **The dead code being unused turns out to be CORRECT** for the
  production-LGBM path.

### 18. Self-supervised pre-training — *BERT* (Devlin et al., 2019), *MAE* (He et al., 2021)

Pre-train transformer on masked-feature reconstruction before LTR fine-tuning.

* **Did NOT implement.** Honest reason: not enough panel data to make this
  pay off (491 hourly-era dates is tiny by self-supervised standards). On
  the to-do list if we get a 5-year hourly-feature panel.

### 19. Bayesian dropout / MC-dropout — Gal & Ghahramani, 2016, ICML

Use dropout at inference to produce a Bayesian posterior over predictions.

* **Did NOT implement.** Would give the transformer a free σ estimate (like
  the NGBoost head used by the production stack) but adds inference cost.
  Considered for the ensemble path; deferred.

### 20. LambdaRank — Burges, 2010 (LambdaMART, MSR-TR-2010-82)

Pairwise gradient weighted by NDCG@K change. Used by production LightGBM
(it's the `lambdarank` objective with `lambdarank_truncation_level: 10`).

* **Did NOT implement in the transformer.** Identified as a likely
  improvement path in `doc/ops/transformer-promotion.md` — when revisited,
  this is the loss to test before any architectural changes (transformer
  uses ListNet currently).

## Test coverage

* **43 lib tests** (`cargo test --release -p transformer_scorer --lib`):
  18 dataset tests, 5 loss tests, 5 loss_pairwise tests, 6 cv tests, 8
  transformer_block tests, 1 trainer test.
* **9 bridge tests** (`tests/bridge_quality.rs`, `tests/poc_parity.rs`).
* **Python regression tests** — `tests/test_round3_audit_fixes_2026_04_25.py`
  (262 passing) covers all 50+ Round 2/3 audit fixes including the Rust-side
  ones via PyO3.

## Provenance

This sprint touched ~20 papers/algorithms. Empirical winners that shipped:

* ✅ ListNet over RankNet (paper 4 confirmed by paper 2 vs paper 3).
* ✅ AdamW (paper 11) — standard, no surprises.
* ✅ CPCV (paper 6) — verified Fix A finding.
* ✅ Sparsity-aware NaN handling (paper 13) — but the lesson was the OPPOSITE
  of what we expected: in production, the way to make LightGBM use it
  effectively is to restrict the training window so missingness becomes
  uniform rather than divergent.

Empirical losers (kept, not promoted):

* ❌ Stronger regularization (paper 9, 10) — paper recommendations of
  dropout 0.5 + weight_decay 1e-3 made real-data IC WORSE.
* ❌ Missingness indicators (paper 17) on hourly-era — wiped the +164%
  Fix A gain.
* ❌ Pairwise loss (paper 3) — confirmed worse than listwise on synthetic;
  not retested on real.

The transformer itself: validated on synthetic (val_IC +0.2314 within 0.003
of the MLP ceiling — port is correct) but lost to production LightGBM on
real data (+0.0519 vs +0.0850). Rust transformer stays as research artifact,
not promoted.

---

## Papers added 2026-05-19 / 2026-05-20 (HF Trainer refactor + Pillar B)

### 21. Nie et al., 2023 ICLR — *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers* ([arXiv 2211.14730](https://arxiv.org/abs/2211.14730))

PatchTST — patches the lookback into non-overlapping subseries, embeds each
patch as a token, channel-independent Transformer encoder.

* **Implemented in:** `scripts/patchtst_hf.py` (HF `transformers.PatchTSTModel` backbone)
* **What we ported:** vanilla PatchTST backbone via Hugging Face. Per `CLAUDE.md §5.12`
  canonical-lib mandate, the 2026-05-19 refactor swapped a hand-rolled custom
  PatchTST/iTransformer/LSTM/MLP implementation (784 LOC `scripts/transformer_v4.py`,
  shelved) for the HF version + custom multi-task head.
* **Status:** SHADOW since 2026-05-19 (commits `cf6311c`, `4e156e2`).
  Channel-independence is the documented #1 failure mode for cross-sectional
  finance (arXiv 2502.09683, 2505.12761). Cross-stock attention planned T2.1.

### 22. CIKM 2025 — *On Evaluating Loss Functions for Stock Ranking* ([arXiv 2510.14156](https://arxiv.org/abs/2510.14156))

Loss function benchmark on PortfolioMASTER × S&P 500 top-110. Margin Ranking
+ ListNet beat pairwise BCE + MSE on portfolio Sharpe + annual return.

* **Implemented in:** `scripts/patchtst_hf.py::margin_ranking_loss` (uses canonical
  `torch.nn.functional.margin_ranking_loss`)
* **Reported gain:** Margin Ranking 0.75 Sharpe / 16.23% AR vs MSE 0.58 / 12.0%
  on PortfolioMASTER. NDCG (ranking accuracy) does NOT correlate with portfolio
  Sharpe — pick loss for downstream metric.
* **Status:** Default loss for HF PatchTST shadow.

### 23. Perez et al., 2017 — *FiLM: Visual Reasoning with a General Conditioning Layer* ([arXiv 1709.07871](https://arxiv.org/abs/1709.07871))

Feature-wise Linear Modulation: `γ, β = MLP(context); h' = γ ⊙ h + β`.
Lightweight conditional learning — shared backbone, context-aware modulation.

* **Implemented in:** `scripts/patchtst_hf.py::FiLMLayer` (~25 LOC)
* **Status:** Pillar B foundation shipped 2026-05-19 commit `78e59d3`.
  `--film-regime-cond` opt-in flag. Zero-init last layer → identity at init
  → strict superset of FiLM-OFF baseline.
* **Empirical:** 5-cut × 5-seed A/B against baseline pending (driver:
  `scripts/eval_hf_film_5cut_5seed.py`).

### 24. Student-t NLL via `torch.distributions.StudentT`

Multi-task distributional head emits (df, loc, scale) — Student-t negative
log-likelihood as auxiliary loss for σ-calibrated predictions.

* **Implemented in:** `scripts/patchtst_hf.py::student_t_nll` + `dist_head`
* **Why:** NGB σ wire failed in 3 prior A/B (2026-05-17). End-to-end Student-t
  in HF Trainer shares representation with ranking head — no train/serve skew.
  Inference outputs (μ, σ) for downstream Kelly/QP.
* **Status:** Shipped 2026-05-19. σ-calibration verifier at
  `scripts/verify_sigma_calibration.py`. Target: σ-calibration coefficient
  ≥ 0.20 (NGB Duan 2020 §4 baseline).

### 25. Zeng et al., 2023 AAAI — *Are Transformers Effective for Time Series Forecasting?* ([arXiv 2205.13504](https://arxiv.org/abs/2205.13504))

DLinear / NLinear — single-matmul trend+seasonal decompose. Beats 8 transformer
variants on ETT/Electricity benchmarks; mandatory baseline per §5.12.

* **Implemented in:** `scripts/dlinear_baseline.py` (adapted for cross-sectional
  ranking — per-channel collapse T→1, then head over channels → score)
* **Status:** Shipped 2026-05-19 commit `97f6f35`. 5-cut × 5-seed eval driver
  at `scripts/eval_dlinear_5cut_5seed.py`.
* **§5.12 gate:** if PatchTST cannot beat DLinear by ≥+0.005 min-regime IC,
  architecture is NOT the bottleneck (labels/features/sample size is).

### 26. Bailey-López de Prado 2014 — *The Deflated Sharpe Ratio* ([SSRN 2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551))

DSR accounts for multiple-comparison selection bias. Mandatory for promotion
gating per CLAUDE.md §5.13.4a Tier 3 (`DSR > 0.5 OR PBO < 0.5 OR n ≥ 30 + t > 3.0`).

* **Implemented in:** `kernel/model_acceptance.py`, `sim/runner.py::run_backtest_multi_seed`
* **Status:** PRODUCTION — every multi-trial batch must quote DSR alongside raw Sharpe.

### 27. Bailey-Borwein-López de Prado-Zhu 2015 — *Probability of Backtest Overfitting via CSCV* ([J. Comp. Finance 14(1)](https://www.davidhbailey.com/dhbpapers/backtest-overfitting.pdf))

PBO via combinatorially-symmetric cross-validation. Complement to DSR.

* **Implemented in:** `kernel/cscv_pbo.py`, returned by `run_backtest_multi_seed` when N ≥ 2
* **Status:** PRODUCTION (used in DOE summary stats)

### 28. Hou-Xue-Zhang 2020 RFS — *Replicating Anomalies* ([RFS 33(5)](https://academic.oup.com/rfs/article/33/5/2019/5236625))

65% factor replication failure rate. Justified disabling `deep_drawdown_veto`
(DDV) globally on 2026-05-17 — distress/loser anomaly fails to replicate
post-Fama-French 2008.

* **Status:** Drives §5.14.5 "no decoration-citing" + DDV disable.

### 29. Kelly-Gu-Xiu 2020 RFS — *Empirical Asset Pricing via Machine Learning* / firm characteristics list

Canonical 102 firm characteristics. CSRankNorm per-day is their standard
preprocessing for ranking-based panel models.

* **Implemented in:** `scripts/patchtst_hf.py::csrank_norm_per_day`
* **Status:** PRODUCTION (applied to all 172 features at train + inference)

### 30. cvxportfolio (Boyd, Stanford) + Almgren-Chriss 2000 + Ledoit-Wolf 2004

Portfolio QP backend.

* **Implemented in:** `kernel/portfolio_qp/qp_solver.py` (cvxpy + CLARABEL),
  `kernel/portfolio_qp/cvxportfolio_backend.py` (full cvxportfolio SinglePeriodOpt)
* **Status:** PRODUCTION — cvxpy CLARABEL is primary path; cvxportfolio backend
  shipped 2026-05-07 as alternate.

### 31. Berkin-Jeffrey 1990 — HIFO tax-lot accounting

Highest-In-First-Out for tax-optimal lot disposal.

* **Implemented in:** `kernel/portfolio_qp/tasks.py::qp_tax_lot_method=hifo`
* **Status:** PRODUCTION default 2026-05-17 commit `bc18795` (was FIFO).
  Pure accounting change — `feedback_no_tax_driven_logic`-safe.

### 32. IRC §1091 cost-aware wash sale + min_reentry_days anti-churn

US tax code §1091 wash-sale rule applied as cost-aware QP constraint (gain =
no cost; loss = NPV deferred-tax cost). Compounded with `min_reentry_days=5`
business days anti-churn (2026-05-18 MCD incident).

* **Implemented in:** `kernel/portfolio_qp/`, `kernel/wash_sale.py`
* **Status:** PRODUCTION
