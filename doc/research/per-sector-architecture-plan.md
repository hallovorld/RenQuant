# Sector-aware ranking architecture — design v2 (post-audit)

**Branch:** `exp/wl500-and-sector-arch`
**Status:** Design v2 (post literature audit). v1 (per-sector hard split + percentile aggregation) **rejected** by Witter 2025 + qlib OSS pattern.
**Goal:** Unblock watchlist expansion past 178 tickers. Production retention ≥ +0.0418 OOS Spearman IC, ideally substantially better.

---

## v1 design rejected

The original v1 plan was per-sector hard splits with percentile aggregation. The literature audit decisively contradicted it:

- **Witter 2025 (JMS, "Predicting Stock Returns: Global Versus Sector Models")** ran exactly this A/B (global pooled NN vs. per-GICS-sector NN, 1963–2022 US equities) and found **the global model beats per-sector NNs**. Documented mechanism: per-sector splits trade pooling-driven sample efficiency for false specialization. *This is our exact experiment, with the opposite-of-our-hypothesis conclusion.*
- **Gu-Kelly-Xiu 2020 (RFS, "Empirical Asset Pricing via Machine Learning")** uses **one pooled estimator** with industry as a feature, not per-industry submodels. Standard ML-finance reference.
- **Poh-Lim-Zohren-Roberts 2020 (arXiv:2012.07149)** — the rank:pairwise paper we already cite — runs ONE LambdaMART over 88 futures contracts spanning commodities/FX/rates/equity. Far more heterogeneous than our tech-vs-financials problem. They handle it via feature normalization, not splits. *Directly contradicts the "rank-pairwise needs homogeneity" claim.*
- **OSS code:** Microsoft `qlib`, `alpha-mind`, `Temporal_Relational_Stock_Ranking`, WorldQuant Brain — none train per-sector models. All pooled with sector-as-feature or sector-as-graph-edge. Sector neutralization happens in the *risk model* downstream, not in the predictor.
- **Empirical-Bayes math:** order-statistic noise on N≈20–30 ticker pools makes per-sector percentiles a coin flip (σ/√N ≈ 0.22σ). Robbins 1955–56, Wheeler practitioner note.

Hard split = no published support, no OSS reference, predicted-bad-by-EB-math, predicted-bad-by-Witter. Drop it.

---

## v2 architecture: pooled with multi-layer sector awareness

A single pooled ranker with five layers of sector-conditioned structure. Each layer is independently published / OSS-validated. They compose; we activate them via flags so each can be A/B-tested in isolation per CLAUDE.md §5.2 (sanity check every claim).

### Layer 1 — cross-sectional rank-normalize features within `(date, sector)`

Pre-pipeline transform: for each numeric feature column `f`, replace its raw value with the per-`(date, sector)` percentile rank. Mathematically: `f_norm = pandas.groupby(['date','gics_sector'])[f].rank(pct=True)`.

Why: this is the actual fix Witter identifies. The reason wl178 fitting collapsed is that a tech ticker's 20-day momentum lives on a different scale than an energy ticker's. Once both are at "75th percentile within their own sector on this date," the pairwise comparison is between **comparable rankings**, not between raw scales.

Reference: `qlib.data.dataset.processor.CSRankNorm` (Microsoft `qlib`, [github.com/microsoft/qlib/blob/main/qlib/data/dataset/processor.py](https://github.com/microsoft/qlib/blob/main/qlib/data/dataset/processor.py)).

Implementation: feature transform task in `kernel/panel_pipeline/`, runs after raw feature computation, before model input. Flag `panel_ltr.cross_sectional_norm.enabled` (default false → exact wl103 behavior).

### Layer 2 — sector as model conditioning

Two paths depending on backend:

- **Tree path (XGBoost / current production):** add `gics_sector` as a categorical feature with `enable_categorical=True`. XGBoost 1.7+ handles GICS-cardinality categoricals natively; one config line.
- **NN path (transformer, future):** learnable 16D sector embedding concatenated to per-ticker feature row.

Reference: Gu-Kelly-Xiu 2020 §3.4 ("industry indicators as features"). Implementation cost: 5 LOC for the tree path; 50 LOC for NN.

### Layer 3 — temporal graph convolution over sector relations

For each date, build a graph where edges connect tickers in the same sector. A graph attention layer aggregates neighbor information so a candidate's prediction sees its sector cohort, not just its own row.

Reference: **Feng-Chen-He-Yang-Cao 2019 ("Temporal Relational Ranking for Stock Prediction", arXiv:1809.09441, TOIS 2019)** — TGC outperforms independent-stock baselines by ~98% return-ratio on NYSE. OSS implementation at [`fulifeng/Temporal_Relational_Stock_Ranking`](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking).

This layer is **flagged behind `panel_ltr.graph.enabled` (default false)** because (a) it's NN-only (XGBoost can't do attention) and (b) it's the heaviest compute lift. Ship Layer 1 + 2 first; add Layer 3 if Layer 1+2 don't recover IC to ≥ +0.040 on the expanded universe.

### Layer 4 — soft Mixture-of-Experts with sector-group aggregation

`K` ranker experts (e.g. K=8) all see the full feature row. A learned gating network produces a sparse top-2 routing weight per ticker conditioned on (sector embedding, market regime). Each expert's contribution is weighted; final score is the gated sum.

Reference: **MIGA 2024 ("Mixture-of-Experts with Group Aggregation for Stock Market Prediction", arXiv:2410.02241, NeurIPS-W 2024)** — +24% excess annual return on CSI300, +8 pts over prior SOTA. Group aggregation = constraining routing to ensure same-sector tickers see the same expert pool, which forces sector-cohesive specialization without hard splits.

Implementation: NN-only; replaces the single XGBoost ranker with a MIGA-style PyTorch module on the graph-attention output. Flagged behind `panel_ltr.moe.enabled` (default false). Comes online only after Layer 3.

### Layer 5 — empirical-Bayes shrinkage of per-sector percentile

Even with pooled training, downstream consumers (selection, gate B) may want a percentile-style score. Per-sector percentile alone is noisy at small N. EB shrinkage:

```
p_shrunk = (n / (n + k)) * p_raw_within_sector + (k / (n + k)) * p_global
```

where `n` is the sector's ticker count on this bar and `k` is a hyperparameter (start with `k = 10`). Robbins 1955; Wheeler 2018 practitioner note ([andrewpwheeler.com/2018/07/23/sorting-rates-using-empirical-bayes/](https://andrewpwheeler.com/2018/07/23/sorting-rates-using-empirical-bayes/)).

Implementation: post-scoring task in `kernel/panel_pipeline/`. ~30 LOC + tests.

---

## Mandatory diagnostic before any layer ships (per CLAUDE.md §2b)

The audit said: *"Before architectural change, run a per-feature distribution audit on Tech vs. Financials in our 178-ticker panel. The Witter mechanism predicts feature-distribution mismatch (volatility/liquidity scales) drives the negative IC — confirm it's that, not a label or a leakage bug."*

We do **not** start writing Layer 1+ code until the diagnostic confirms sector heterogeneity is the actual cause. The wl178 negative-IC could be:
- (a) Real sector heterogeneity → architecture change indicated.
- (b) Implementation bug (label leakage, train/eval set overlap, calibration unit mismatch, NaN propagation).
- (c) Stale/contaminated data on specific new tickers.

If (b) or (c), the architectural change is wasted compute and *masks* the real bug.

### Diagnostic script: `scripts/audit_wl178_failure.py`

Three independent checks. All must produce results consistent with hypothesis (a) before we proceed.

**Check 1 — A/A test on the 178-ticker universe.** Randomly split the 178 tickers into two homogeneous halves of 89 each. Train rank:pairwise on each half independently with the same code path that produced the wl178 negative-IC artifact. If half-A and half-B both produce IC ≈ +0.04 (similar to wl103), heterogeneity is the cause. If both produce negative IC, it's a code bug regardless of universe composition.

**Check 2 — Per-sector feature distribution KS test.** For each numeric feature, compute the Kolmogorov-Smirnov statistic comparing its distribution within each pair of GICS sectors on the 178-ticker panel. Sectors with KS ≥ 0.3 (large effect) on a majority of features are heterogeneous; sectors with KS < 0.1 are not. Witter mechanism predicts large KS between Tech and Financials, small KS within Tech.

**Check 3 — Label-leakage and unit-consistency audit.** Compute corr between (a) raw feature values on the wl178 panel and (b) raw feature values on the wl103 panel for each shared ticker on the same dates. Mismatches indicate the wl178 build path corrupted features (e.g. different normalization parameters, stale intraday cache, sector-relative rebasing applied to one but not the other).

Verdict matrix:

| Check 1 result | Check 2 result | Check 3 result | Interpretation | Action |
|---|---|---|---|---|
| Both halves +0.04 | Large KS | Clean | True heterogeneity (Witter) | **Proceed with Layer 1** |
| Both halves negative | — | — | Code bug | **Halt, find bug** |
| One half negative | — | — | Possibly contaminated subset | **Investigate the bad half** |
| Halves +0.04 | Small KS | Clean | Heterogeneity not measurable; success unclear | Alternative diagnosis |
| — | — | Mismatched features | Build pipeline corruption | **Fix build, retest wl178** |

---

## Implementation phases (after diagnostic clears)

Each phase ships as **one promotable artifact** with B2 hold-out validation per CLAUDE.md §B3.

### Phase A — Layer 1 + Layer 2 (XGBoost-compatible, fast)

- Add `kernel/panel_pipeline/feature_norm/cross_sectional.py` — `CrossSectionalRankNormalizer` Task implementing `pandas.groupby(['date','gics_sector']).rank(pct=True)`.
- Wire into `PanelFeatureJob` after raw feature build, before label assembly.
- Add `gics_sector` as XGBoost categorical input.
- Tests: A/A on shuffled within-sector ticker IDs; same-day cross-sector rank invariance; null-sector handling (default to "general").

Cost: ~150 LOC + tests. Compatible with existing XGBoost stack — no NN dependencies yet.

### Phase B — Layer 5 EB shrinkage (independent, ships any time)

- `kernel/panel_pipeline/eb_shrinkage.py` — Task that runs after `PanelScoringJob`.
- Tests: large-N → near-no-shrinkage; small-N → strong shrinkage; null handling.

Cost: ~80 LOC + tests.

### Phase C — Layer 3 graph attention (NN backend gated)

Substantial. New backend; leave behind `panel_ltr.graph.enabled` flag. Reference impl: [`fulifeng/Temporal_Relational_Stock_Ranking/RankLSTM`](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking).

Cost: ~500 LOC + tests + GPU/MPS validation. 1-2 weeks of work.

### Phase D — Layer 4 MIGA MoE (NN, replaces single ranker on graph-attention path)

The crown of the architecture. Replaces XGBoost rank when `panel_ltr.moe.enabled=true`. Reference: MIGA paper has GitHub repo (link in audit).

Cost: ~700 LOC + tests + careful gradient sanity check (MoE training is notoriously brittle). 2-3 weeks.

### Phase E — Promote (PR back to main)

Only after each phase's B2 hold-out beats wl103 baseline by margin appropriate to its complexity:
- Phase A+B: ≥ +1 APY pt (cheap; lower bar)
- Phase C: ≥ +2 APY pts (NN backend; demands evidence)
- Phase D: ≥ +4 APY pts (full architecture; demands strong evidence)

If a phase doesn't clear its bar, **document as failed experiment** in `failed-experiments-log.md` and revert. Phases A and B can ship even if C and D never land — they're independent improvements.

---

## Quality controls (per user 2026-05-01: "最先进最强的架构和代码质量")

Per CLAUDE.md §5 engineering principles:

1. **§5.1 — tests-before-push.** Every Task has unit tests for its core math + invariant. Run tests for the touched file + its dependents on every commit.
2. **§5.2 — every metric ships with a sanity check.** New IC numbers require at minimum: A/A test, shuffled-label test, time-shift placebo. No exceptions.
3. **§5.3 — name the invariant.** Each new Task has an "Invariant:" line in the docstring naming the structural property the design preserves.
4. **§5.4 — no edits to running scripts.** Anything wired into launchd path requires stop+restart; no live in-place edits.
5. **§5.5 — rehearse rollback.** Each promotable artifact ships with a `--revert` runbook (also tested).
6. **§5.6 — definition of fixed = full audit clean.** A phase isn't "done" until: patch + regression tests + 24h re-audit of every touched file + every upstream/downstream consumer.
7. **§5.7 — failed experiments documented same day.** If Layer 3 or 4 underperforms, write the entry before moving on.

---

## Reproduction recipe (post-merge)

```bash
# 1. Universe
python scripts/build_universe.py

# 2. OHLCV backfill for new tickers
python scripts/fetch_ohlcv.py --from-universe scripts/watchlist_universe.json

# 3. Diagnostic — proves heterogeneity is the actual issue
python scripts/audit_wl178_failure.py
#    Look for the verdict matrix in the report — proceed only if "Proceed with Layer 1"

# 4. Train Phase A architecture on expanded universe
python scripts/train_104.py --strategy-config-name strategy_config.sector_aware.json

# 5. Validate via B2 hold-out
python scripts/holdout_backtest.py \
    --strategy-config-name strategy_config.sector_aware.json \
    --train-end 2024-12-31 --sim-start 2025-01-02 --sim-end 2026-04-30

# 6. Compare to baseline
python scripts/holdout_backtest.py \
    --strategy-config-name strategy_config.golden.json \
    --train-end 2024-12-31 --sim-start 2025-01-02 --sim-end 2026-04-30

# 7. Promotion criterion: apy_holdout(sector_aware) ≥ apy_holdout(baseline) + 1.0 (Phase A bar)
```
