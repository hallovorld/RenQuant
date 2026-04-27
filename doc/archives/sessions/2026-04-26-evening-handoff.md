# Session — 2026-04-26 evening + autonomous handoff

**Context**: Continuation of 2026-04-26 round-7 day. Late-evening session + 8-hour autonomous handoff window starting ~T+5h. User direction across the session:

1. Build model-selection systematization (Phases 1-4)
2. Audit + fix all findings
3. Doc reorg deep pass (README + 104-first + sync to current code)
4. Roadmap items prioritized + parallel execution per "并行进行所有任务"
5. 8-hour autonomous handoff to make progress on S/A items

This doc captures the work shipped from ~22:00 PT 2026-04-26 onward.

---

## What shipped (chronological)

### Model-selection systematization (Phase 1 → 4)

| Phase | Commit | Delivers |
|---|---|---|
| Phase 1 | `570348b` | G4 default tightened 30%→5%; G7 hardened soft→hard; config-driven gate factory; `acceptance` block in strategy_config + golden |
| Phase 2 | `aee943f` | G9 sim_apy / G10 sim_sharpe / G11 turnover gates + `kernel/sim_smoke.py` (compute_metrics_from_equity_curve, add_smoke_metrics_to_artifact, run_smoke_test stub) |
| Phase 3 | `34e2af2` | `scripts/select_best_model.py` — backend tournament with composite ranking + `--promote` |
| Phase 4a | `b96260d` | `kernel/challenger.py` (ChallengerConfig, ChallengerEvaluator, log_decision, compare_window) + `challenger_decisions` DB schema |
| Phase 4b-1 | `0f40b08` | `scripts/finalize_challenger.py` (window verdict report + ntfy) + `scripts/check_challenger_window.sh` |
| Phase 4b-2 | `ecc8288` | `scripts/model_dashboard.py` operator UX |
| SOP doc | `1e96295` | `doc/components/model-selection.md` — canonical 4-tier model-selection SOP |

**Test impact**: +88 model-selection tests (test_model_acceptance.py, test_acceptance_phase2.py, test_select_best_model.py, test_acceptance_phase4.py, test_finalize_challenger.py).

### Bug #25 train-inference macro symmetry (commits e436fd9, 080d21c)

The macro retrain auto-promoted but inference path couldn't reproduce the 33 macro features → silent feature drop at scoring. Full architectural fix:
- `prepare_inference_panel_frames` returns 3-tuple `(ff, fac, macro)` (was 2-tuple)
- All 3 adapters (sim/lean/runner) updated to unpack + attach `_panel_macro_frame`
- `feature_matrix.py::build_inference_matrix` accepts macro_frame param + broadcasts per-date
- `tests/test_train_inference_symmetry.py` — symmetry guard test (enumerates Load*Tasks in PanelDataJob.tasks vs prepare_inference_panel_frames source)

### 12-finding audit fix sweep

Independent agent did deep audit of model-iteration code path; surfaced 3 HIGH + 9 MED findings. All fixed:

| Severity | Commit | Findings |
|---|---|---|
| HIGH | `702ce6c` | #2 promote validates JSON before swap; #9 always-clean .pre-train.json (try/finally); #12 atomic os.replace via .incoming.json |
| MED | `b4bfbe9` | #3 G4 with non-positive prior requires new > 0 strictly; #4+#11 hard gates emit log.warning when skip-passing on missing metadata |
| MED | `d72df61` | #5 compare_window now computes score_rank_corr; #6 _load_artifact_metadata warns on metadata/top-level conflict; #7 DEFAULT_GATES annotated as fallback-only; #8 pp_inference.py warns when challenger.enabled=true but live wiring missing |

#1 + #10 (macro per-ticker context) investigated and confirmed as **false positives** — BuildPanelTask reads `ctx.macro_factor_frame` from PARENT PanelTrainingContext, not per-ticker.

### Documentation reorg deep pass

Per user direction "至少readme都是陈旧内容，以104为主的内容应该是主要内容":

| Commit | Doc | Change |
|---|---|---|
| `aceee5f` | README.md, doc/README.md | Full rewrite — 104-first, accurate paths, themed index |
| `2d6401c` | usage.md, strategy-104.md, overview.md, golden-config.md | Sync to round-7 surface |
| `343a983` | tech-stack.md | Added LGBM/NGBoost/transformer/cvxpy/Rust/etc to backend mix |
| `aea5020` | databases.md, training-pipeline.md, models.md | 11-table inventory + Round-7 acceptance flow + panel-LTR layer cross-refs |
| `4ec3261` | strategy-104.md | S1 — resolved 41→28 as deliberate Tier 1 cleanup, not regression |
| `9cd98b3` | asset-embeddings-design.md, boyd-rotation-design.md | NEW design docs for A2 (Tier 2) and A1 (Tier 2) |
| `2ed9c52` | indicators.md | Added 104 kernel row + Tier 1 drop_cols note |

### Metadata DB + cloud backup plan (deferred)

- `8fc0b96` — `doc/components/metadata-db-and-backup-plan.md` — 7-column training_runs extension + 2 new tables (training_run_gates, tournament_rankings) + Backblaze B2 3-tier backup design (~$1/year)
- `0aea613` — Added to roadmap as DEFERRED per user "下次再处理"

---

## Experiments run + results

| Backend | Macro | Trained | OOS IC | Status |
|---|---|---|---|---|
| **XGBoost (PROD)** | OFF | 2026-04-26 | **+0.0482** | ✅ ACTIVE |
| XGBoost | ON (61 feat) | 2026-04-26 | +0.0393 (-18%) | preserved at `panel-ltr.macro-enabled.bak.json` |
| LightGBM | ON | 2026-04-26 23:25 | +0.0224 (-53%) | preserved at `panel-ltr.lgbm-macro.bak.json` |
| LightGBM | OFF (S2) | RUNNING (handoff) | tbd | retrain in progress (PID 62842) |
| XGBoost rank:ndcg (A3) | OFF | queued | tbd | will run after S2 |

**Conclusion (so far)**: macro features hurt IC on both XGBoost (-18%) and LightGBM (-53%). #158 transformer + macro skipped — evidence is sufficient.

---

## Production safety status

**Current production artifact** (verified via `model_dashboard.py`):
- `panel-ltr.json`: XGBoost, 28 features, OOS IC 0.0482, trained 2026-04-26
- `ngboost-head.json`: 526k XGBoost-compatible (restored after LGBM-macro contamination)
- `panel-rank-calibration.json`: 1.3k XGBoost-compatible (untouched by LGBM-macro experiment)
- Rollback target at `panel-ltr.previous.json`

**Damage control history**: LGBM-macro retrain (--skip-acceptance) overwrote panel-ltr.json + ngboost-head.json. Both restored from .xgboost.bak. New audit finding to add: `--skip-acceptance` should write to sidecar paths instead of stomping production. Roadmap entry pending.

---

## Roadmap items prioritized (per user "按预期收益排序")

### Tier S — Highest ROI

1. **S1 (closed 2026-04-27)**: 41→28 feature regression — investigated, false alarm. The 28-feature prod is a deliberate Tier 1 cleanup that IMPROVED OOS IC by +23% vs PRE-MINUTE.
2. **S2 (in progress at handoff start)**: LGBM no-macro re-test of T2-1 +128% claim. Result tbd.

### Tier A — High confidence + impact

3. **A3**: listwise loss (rank:ndcg) XGBoost retrain — config built, queued after S2.
4. **A2**: asset embeddings (Dolphin 2024) — design doc written. Implementation deferred for user review.
5. **A1**: Boyd convex rotation (cvxpy) — design doc written. G-P 2013 +20% Sharpe claim. Implementation deferred for user review.

---

## Open decisions for user (when you return)

1. **#158 transformer + macro?** Recommend SKIP — both prior backends show macro is net-negative.
2. **Restore macro-enabled XGBoost to prod?** Recommend NO — −18% OOS IC vs current.
3. **Implement A2 asset embeddings?** Design doc ready (`doc/components/asset-embeddings-design.md`); ~2.5 days work, low risk.
4. **Implement A1 Boyd convex rotation?** Design doc ready (`doc/components/boyd-rotation-design.md`); ~4.5 days work, medium risk; G-P claims +20% Sharpe.
5. **Implement metadata DB + cloud backup?** Design doc ready; ~6h work; needs your provider/retention/encryption decisions.
6. **Phase 4b live wiring of ChallengerEvaluator?** Platform ready (Phase 4a); operator UX ready (Phase 4b-1+2); needs design call on per-bar challenger eval risk.

---

## Test totals

- **Pre-session**: ~2330 tests
- **Post-session**: 2418 tests (+88 model-selection tests, all passing)

---

## Commits since handoff start (in push order)

```
570348b feat(model-selection Phase 1): tighten G4 + harden G7
aee943f feat(model-selection Phase 2): G9/G10/G11 + sim_smoke
34e2af2 feat(model-selection Phase 3): select_best_model.py
b96260d feat(model-selection Phase 4a): challenger / shadow infra
1e96295 docs(model-selection): canonical 4-tier SOP
0f40b08 feat(4b-1): finalize_challenger.py + check_window.sh
8fc0b96 docs+config: metadata-db plan + LGBM-macro retrain config
ecc8288 feat(4b-2): scripts/model_dashboard.py
702ce6c fix(audit): HIGH #2 + #9 + #12
0aea613 roadmap: defer metadata DB + backup
b4bfbe9 fix(audit): MED #3 + #4 + #11
d72df61 fix(audit): MED #5 + #6 + #7 + #8
aceee5f docs(reorg): rewrite root README + doc/README
2d6401c docs(reorg): sync usage/strategy-104/overview/golden-config
343a983 docs(reorg): refresh tech-stack.md
aea5020 docs(reorg): databases + training-pipeline + models
4ec3261 docs(strategy-104): S1 resolution
9cd98b3 docs(roadmap A1+A2): design plans
2ed9c52 docs(indicators): add 104 kernel impl + Tier 1 note
```

(plus future commits during the rest of the handoff window — see `git log --oneline e436fd9..HEAD` for the full ordered list at session end)
