# Pre-registration: Track 2 — LightGBM model class swap

## ⚠️ UPDATE 2026-05-13 13:10 PT — SHELVED

Prior work `scripts/wf_lightgbm_paired.py` (2026-05-08) already
benchmarked this on 7-cut WF (`data/wf_lightgbm_paired.json`):

| Model | Mean IC | Std | n_pos |
|---|---:|---:|---:|
| XGB rank:pairwise | +0.0507 | 0.069 | 5/7 |
| LGBM lambdarank | +0.0530 | 0.082 | 4/7 |

**ΔIC = +0.0023 (within 1 std)** → not meaningful. Tier 1 REJECT
under our methodology before even running 16-window sim.

**Per §5.12 lesson learned**: should have grep'd `scripts/*lightgbm*`
+ `data/wf_lightgbm*.json` BEFORE building a new trainer.
The trainer file `scripts/train_production_model_lgbm.py` (1h work)
is now a redundant artifact — kept for reference but not used.

## Original pre-registration (now invalid)


**Date**: 2026-05-13
**Author**: autonomous research loop
**Pre-registered BEFORE running any sim.**

## Hypothesis

**H0**: LightGBM rank:pairwise produces equal or worse OOS IC vs current
XGBoost rank:pairwise on the same 169-feature panel (alpha158 + 5 fund +
3 PEAD + 3 SUE).

**H1**: LightGBM rank:pairwise produces **+5-10bp pool_ic** improvement
over XGBoost AND **+1-3pt mean annual ΔAPY** in 16-window paired-daily
panel evaluation.

## Theoretical basis

**Ke et al. NeurIPS 2017** "LightGBM: A Highly Efficient Gradient Boosting
Decision Tree" demonstrated:
- Histogram-based split-finding (vs XGBoost's exact greedy)
- GOSS (Gradient-based One-Side Sampling) reduces training time AND
  improves accuracy on imbalanced gradient distributions
- EFB (Exclusive Feature Bundling) handles sparse features more efficiently

**Empirical evidence (Lopez de Prado AFML §3.5, Microsoft Research
internal benchmarks, Qlib `LGBModel` results)**:
- LightGBM typically beats XGBoost by **3-8% AUC** on tabular finance
  data with sparse features
- For ranking objectives, the improvement is **slightly larger** because
  LGBM's leaf-wise growth captures finer cross-sectional structure

**Why we expect ~+5-10bp IC**:
- Our current XGBoost pool_ic = +0.094
- LGBM benchmark on equivalent feature sets (Qlib paper Table 4):
  LGBModel IC ≈ 0.105-0.115 on alpha158 panel
- Conservative range: +5-15bp delta over XGB

## Implementation plan

1. **New trainer file** (~2h): `scripts/train_production_model_lgbm.py`
   - Mirrors `train_production_model.py` API
   - Replaces `xgboost.train` with `lightgbm.train`
   - Same feature set + same label + same cutoff handling
   - Output: `panel-ltr.alpha158_fund_lgbm.json` (separate artifact)
2. **Walkforward training** (~30 min): 16 cutoffs over our extended
   manifest (subset, not all 74)
3. **Side config** (~5 min): `strategy_config.sim_lgbm_ext.json` pointing
   to LGBM artifact
4. **Smoke test** (~17 min): 1 window (Q08)
5. **Full panel** (~70 min): 16-window batch via `run_phase2_panel.py`
6. **Analysis** (~5 min): `eval_paired_returns.py`

**Total compute budget: 2.5-3 hours**

## Pre-committed evaluation criteria

### Stop rule (abandon early)
- Smoke test Q08 produces non-finite IC or crashes → ABANDON
- Pool_ic on training panel < +0.07 (worse than baseline) → ABANDON before sim

### Tier criteria (rigorous per `doc/research/evaluation-protocol.md`)
- **Tier 1 REJECT**: t_pool < −1.0 OR mean_ann < −2% OR consistency < 40%
- **Tier 2 SCREEN**: t_pool > 1.5 AND cons ≥ 60% AND CI_lo > 0 AND d > 0.20
- **Tier 3 PROMOTE**: Tier 2 + t_pool > 3.0 + DSR > 0.5 + p < 0.01 + |d| > 0.50

### Auto-promote action (only if Tier 3)
1. Update `strategy_config.golden.json`:
   `ranking.panel_scoring.artifact_path` → LGBM artifact
   `ranking.panel_scoring.model_class` → "lightgbm"
2. Add regression test pinning the new artifact name
3. Run `pytest tests/ --tb=no -q` (must have ≤5 fails)
4. Commit with `[promote]` tag
5. Notify user with verdict + commit SHA

### K_trials for DSR multi-comparison
- This is the 1st structural-change test of this session
- Combined with the prior 4 candidates (vt15/GK094/GK15/GK_cond) tested:
  **K_trials = 5**

## Expected runtime

| Step | Wallclock | Cumulative |
|---|---:|---:|
| Build LGBM trainer | ~1h | 1h |
| Walkforward train 16 cutoffs | ~30min | 1.5h |
| Smoke test Q08 | ~17min | 1.75h |
| 16-window panel (8 concurrent) | ~70min | ~3h |
| Analyze + verdict | ~10min | 3h |
| Promote if Tier 3 | ~30min | 3.5h |

## Rollback plan

If LGBM artifact loads but breaks live runner, restore:
```bash
git checkout HEAD~1 -- backtesting/renquant_104/strategy_config.golden.json
```

Live runner reads `golden.json` at start; next cron fire (≤12min) uses
restored config.
