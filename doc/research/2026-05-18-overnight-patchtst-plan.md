# Overnight PatchTST Optimization Plan — Resume Guide

**Created**: 2026-05-18 21:45 PT
**Authority**: User mandate "不管结果如何都做推进优化... 直接 promote 进 product"
**Cron**: `00975833` every 30min (session-only — dies if Claude Code window closes)

## State machine

The cron job recovers state by reading this doc. Each phase has an entry-condition check (so a re-fire after a crash picks up where it left off).

### Phase 0 — Current DOE finishing

- **Process**: `PID 7998` running `scripts/patchtst_doe_sweep.py`
- **Log**: `logs/patchtst_doe/doe_20260518-182702.log`
- **Output dir**: `artifacts/patchtst_doe/`
- **Progress at doc write**: 22/27 trials, ETA ~22:35 PT
- **Caffeinate**: `PID 19900` binds to 7998 — laptop awake until 7998 dies
- **Exit condition**: `ls -d artifacts/patchtst_doe/pt_*_seed_* | wc -l` == 27 AND `summary.md` exists

### Phase 1 — Preprocessing + HMM helper (tasks #83 + #87)

**Entry**: Phase 0 done, no `kernel/hmm_regime_labels.py` exists.

**Work**:
1. Edit `scripts/patchtst_hf.py` `load_panel_with_split` function:
   ```python
   # CSRankNorm per-day (Kelly-Gu-Xiu 2020)
   panel[feat_cols] = panel.groupby("date")[feat_cols].rank(pct=True) - 0.5
   # Label Winsorize ±3σ
   from scipy.stats.mstats import winsorize
   panel[label_col] = winsorize(panel[label_col], limits=[0.005, 0.005])
   ```
2. Add tests `tests/test_patchtst_hf.py::TestPreprocessing`:
   - CSRankNorm result range ∈ [-0.5, +0.5]
   - Winsorize caps extreme values
3. Create `kernel/hmm_regime_labels.py` reusing `kernel/regime.py` HMM logic:
   - `compute_hmm_regime_labels(panel_dates)` → DataFrame[date, regime ∈ {BULL_CALM, BULL_VOLATILE, BULL_STRONG, BEAR, CHOPPY}]
   - `per_hmm_regime_ic(preds_df, hmm_labels)` → dict[regime → mean_IC]
   - `bull_regime_ic(per_regime)` → mean of {BULL_CALM, BULL_VOLATILE, BULL_STRONG} ICs
4. Tests `tests/test_hmm_regime_labels.py` (≥3 BULL_* regimes appear in 2018-2024 SPY data)
5. Commit + push.

### Phase 2 — HF DOE script + launch (task #84)

**Entry**: Phase 1 done, no `scripts/patchtst_doe_hf.py` exists.

**Work**:
- Mirror `scripts/patchtst_doe_sweep.py` structure but call `scripts/patchtst_hf.py` (not transformer_v4.py)
- pyDOE2 FrFact 2^(4-1) + 1 center = 9 design points
- Per design point: 5 walk-forward cuts (from `kernel/walk_forward_splits.build_default_cuts()`) × 5 seeds = 25 trainings per point
- Predict-average across 5 seeds before per-regime IC computation
- Tightened knob ranges from Phase 0 main effects:
  - `lr ∈ [1e-5, 1e-4]` (Phase 0 showed lr=1e-3 catastrophic)
  - `seq_len ∈ [8, 24]` (Phase 0 showed seq=60 catastrophic)
  - `weight_decay ∈ [1e-5, 1e-3]` (Phase 0 wd=1e-1 hurt)
  - `warmup_epochs ∈ [2, 4]`
- Objective: `bull_regime_ic` pooled across 5 cuts (NOT pooled mean IC — PRIME DIRECTIVE)
- Run with caffeinate:
  ```bash
  nohup caffeinate -i .venv/bin/python scripts/patchtst_doe_hf.py \
    --n-cuts 5 --n-seeds 5 --epochs 4 --device mps \
    > logs/patchtst_doe_hf/doe_$(date +%Y%m%d-%H%M%S).log 2>&1 &
  ```
- Expected wallclock: 9 × 5 × 5 × ~5min = ~19h (overnight + most of next day)

### Phase 3 — SWA confirmatory (task #85)

**Entry**: Phase 2 done, HF DOE picked best point, no `--swa` flag in `patchtst_hf.py`.

**Work**:
- Add `--swa` + `--swa-start-epoch` flags to `scripts/patchtst_hf.py`
- Use `torch.optim.swa_utils.AveragedModel + SWALR` (canonical 3rd-party lib pattern, NOT custom)
- Train confirmatory at HF DOE best point, 5 seeds, 15 epochs, --swa
- Per CLAUDE.md §5.13.4 confirmatory needs 5 seeds + DSR/PBO

### Phase 4 — Bidirectional promote (task #88)

**Entry**: Phase 3 done, SWA confirmatory artifact exists.

**Logic**:
1. Compute `bull_regime_ic(HF_confirmatory)` and `bull_regime_ic(XGB_current_prod)`
2. **IF HF >= XGB**:
   - Write `kernel/panel_pipeline/hf_patchtst_scorer.py` (HF inference wrapper)
   - Register `kind="hf_patchtst"` in `kernel/panel_pipeline/model_registry.py`
   - Edit `golden.json`:
     - `ranking.panel_scoring.kind` → `hf_patchtst`
     - `ranking.panel_scoring.artifact_path` → HF artifact path
     - `ranking.panel_scoring.shadow_models` → append `{name: "xgb_legacy", kind: "xgb", artifact_path: "<current XGB path>"}`
3. **ELSE** (HF < XGB on bull regime):
   - Keep XGB primary (no golden edits to `kind`)
   - `golden.json`: `ranking.panel_scoring.shadow_models` → append `{name: "hf_patchtst_v1", kind: "hf_patchtst", artifact_path: "<HF artifact path>"}`
4. e2e Alpaca LIVE validation:
   ```bash
   nohup bash -c 'set -a; source .env; set +a; .venv/bin/python -m live.runner --strategy renquant_104 --broker alpaca --once' > logs/live_e2e/e2e_$(date +%Y%m%d-%H%M%S).log 2>&1 &
   ```
5. Verify `mlruns/renquant_104_shadow/` populated
6. Commit + push
7. `CronDelete 00975833` to stop the autonomous loop

## Crash recovery checklists

### If laptop reboots / Claude crashes mid-DOE
- Cron `00975833` is in-memory; will be GONE on Claude restart
- DOE PID 7998 will be GONE
- DOE artifacts in `artifacts/patchtst_doe/pt_*_seed_*` PERSIST on disk
- User must re-open Claude → say "继续" → I resume from disk state per this doc

### If laptop sleeps but doesn't reboot
- DOE process likely paused (macOS may suspend on sleep)
- Cron may miss fires while suspended
- On wake: cron resumes fires; DOE process resumes
- Caffeinate `-i` SHOULD prevent idle sleep but lid-close + power-disconnect can override
- Wife closing the lid with power connected: stays awake (caffeinate `-i`)
- Wife unplugging then closing lid: sleeps anyway

### If 27/27 DOE complete but no summary.md
- Likely DOE crashed during fit_main_effects step
- Re-run only the summary writer:
  ```python
  from scripts.patchtst_doe_sweep import (fit_main_effects_and_interactions,
                                            write_summary)
  import pandas as pd
  runs = pd.read_csv('artifacts/patchtst_doe/runs.csv')  # if exists, else build from pt_* dirs
  main_df, inter_df, point_df, intercept = fit_main_effects_and_interactions(runs)
  write_summary(Path('artifacts/patchtst_doe'), runs, main_df, inter_df, point_df, intercept)
  ```

## Pass / fail decision authority

User mandate is **bidirectional**:
- HF wins → swap primary (architecture change AUTHORIZED for this single decision)
- HF loses → register as shadow (zero risk, infrastructure validation)

There is NO scenario where I "do nothing" — every outcome ships something.

The ONE thing that should STOP the autonomous loop:
- Regression tests fail
- NaN/Inf in HF predictions
- All-negative per-regime IC on confirmatory (= broken pipeline)
- e2e Alpaca returns error
- Any other "this looks wrong" signal

In those cases: PushNotification user immediately with details, do NOT promote, end turn.
