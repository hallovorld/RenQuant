# Umbrella `kernel/` inventory — Track C1

**Last updated:** 2026-05-30 by Claude Opus 4.7 (Track C1).

The umbrella's `backtesting/renquant_104/kernel/` has **170+ files** that predate
the multi-repo split. This doc classifies each module/subpackage by its proper
target repo per placement-by-owner. Use as the discovery / decision input for
Track C2-C5 (the actual moves).

**Rule of thumb:**
- **Runtime** (used by the live runner for daily decisions) → `renquant-pipeline`
- **Sim / forensics / eval** (only run offline) → `renquant-backtesting`
- **Shared primitives / contracts** → `renquant-common`
- **Pure data IO** (fundamentals / macro / external feeds) → `renquant-base-data`

---

## A. Runtime (→ renquant-pipeline)

These are imported by `live.runner`, `kernel.pipeline.pp_inference`,
`kernel.preflight`, or any code that runs in production daily. Track C2.

### A1. Decision pipeline core (60 files)

| Path | Notes |
|---|---|
| `pipeline/pp_inference.py` | The InferencePipeline entrypoint |
| `pipeline/pp_execution.py` | Order emission pipeline |
| `pipeline/pp_training.py`, `pp_training_full.py` | Training pipelines (also used by orchestrator) |
| `pipeline/pp_research_acceptance.py` | Could be sim-side; needs grep |
| `pipeline/job_*.py` (12 files) | Per-phase Jobs |
| `pipeline/task_*.py` (28 files) | Atomic Tasks |
| `pipeline/context.py`, `exit_params.py`, `order_attribution.py`, `order_dedupe.py`, `pipeline.py`, `soft_exit_guards.py` | Pipeline glue |

### A2. Panel scoring (16 files)

| Path | Notes |
|---|---|
| `panel_pipeline/panel_scorer.py` | Loads + scores panel-LTR artifacts at runtime |
| `panel_pipeline/feature_transform.py` | Runtime feature normalisation |
| `panel_pipeline/job_panel_scoring.py` | Pipeline Job that loads scorer + scores candidates |
| `panel_pipeline/alpha158_features.py`, `feature_matrix.py`, `tasks_feature_matrix.py` | Feature assembly |
| `panel_pipeline/hf_patchtst_scorer.py`, `patchtst_scorer.py` | Two PatchTST scorer paths (DUPLICATE — D7-style audit needed) |
| `panel_pipeline/transformer_scorer.py`, `ensemble_scorer.py`, `regime_router_scorer.py`, `regime_router.py`, `shadow_scoring.py` | Scorer variants |
| `panel_pipeline/model_registry.py`, `task_quality_floor.py` | Registry + floor |

### A3. Portfolio QP (7 files)

| Path | Notes |
|---|---|
| `portfolio_qp/qp_solver.py`, `signal_combiner.py`, `cvxportfolio_backend.py` | QP core |
| `portfolio_qp/tasks.py`, `task_joint_qp.py`, `job_qp.py` | Pipeline integration |

### A4. Risk / sizing / market gates (top-level)

| Path | Notes |
|---|---|
| `preflight.py` | Live preflight gates (P-WF-GATE etc.) |
| `kelly.py`, `vol_target.py`, `sizing.py` | Position sizing |
| `market_gates.py`, `exits.py`, `exit_types.py` | Exit rules |
| `regime.py`, `regime_resolver.py`, `regime_hmm.py` | Regime detection (currently duplicates renquant-common's hmm_regime_labels) |
| `scoring.py`, `selection.py`, `rotation.py`, `rotation_convex.py` | Decision logic |
| `net_safety.py`, `intraday.py`, `intraday_wash.py` | Intraday + wash-sale |
| `portfolio.py`, `realized_pnl.py`, `trade_events.py` | Portfolio state |
| `data.py`, `data_cache.py`, `data_coverage.py` | Data fetch + cache |
| `indicators.py` | Technical indicators |
| `typed_past/` (4 files) | Typed-past read-only views |

### A5. Execution (8 files)

| Path | Notes |
|---|---|
| `execution/backend.py`, `backend_sim.py`, `backend_lean.py` | Backend abstractions — backend_sim might be sim-only |
| `execution/fees.py`, `slippage.py`, `t2_settlement.py`, `types.py` | Execution mechanics |

---

## B. Sim / forensics (→ renquant-backtesting)

### B1. Acceptance + forensics

| Path | Notes |
|---|---|
| `model_acceptance.py`, `model_acceptance_short.py` | Promotion acceptance gates |
| `acceptance_entry_ic.py` | Entry-IC acceptance |
| `challenger.py` | Challenger model comparison |
| `trade_score_diagnostics.py` | Trade-level diagnostics |
| `sim_smoke.py` | Sim smoke test |
| `artifact_snapshot.py` | Snapshot pre-promotion artifacts |

### B2. Meta-label / triple-barrier (9 files)

| Path | Notes |
|---|---|
| `meta_label/labeler.py`, `predictor.py`, `triple_barrier.py`, `purged_kfold.py` | Triple-barrier method (López de Prado) |
| `meta_label/snapshot.py`, `task_snapshot.py`, `task_meta_label_veto.py`, `job_meta_label_log.py` | Pipeline integration — A4 if veto fires in runtime; B otherwise |
| `triple_barrier.py` (top-level) | Possibly duplicate of meta_label/triple_barrier.py |

### B3. Walk-forward eval (7 files)

| Path | Notes |
|---|---|
| `walk_forward/loader.py` | Manifest loader — also used at runtime (P-WF-GATE reads stamps) so might split |
| `walk_forward/manifest.py` | Manifest assembly |
| `walk_forward/leakage_guard.py`, `correlation_guard.py`, `gmm_guard.py`, `lean_guard.py` | Eval guards |

### B4. Metrics (6 files)

| Path | Notes |
|---|---|
| `metrics/deflated_sharpe.py`, `pbo.py`, `block_bootstrap.py`, `hac_se.py`, `perf_summary.py` | Statistical eval — purely offline |

### B5. Reconciliation

| Path | Notes |
|---|---|
| `reconciliation/live_sim_reconcile.py` | live-vs-sim parity audit |

### B6. Other

| Path | Notes |
|---|---|
| `decision_trace.py` | Writes pipeline_runs / candidate_scores DB — used BOTH by sim and live. **Shared** — possibly stays in renquant-common or splits with a thin shim |

---

## C. Shared primitives (→ renquant-common)

| Path | Notes |
|---|---|
| `config_consistency.py` | ✅ already canonical in renquant-common (D8 deleted pipeline duplicate; this umbrella copy is the remaining shadow) |
| `artifact_contract.py` | Artifact-shape validators |
| `persistence.py` | DB writer (training_runs, pipeline_runs) — already imported by orchestrator/train_gbdt from renquant-pipeline; the umbrella version might be stale duplicate |
| `state_paths.py` | Path conventions |
| `calibrator_quality.py` | Calibrator diagnostics |
| `row_coverage.py` | Row-coverage gate |
| `registry/mlflow_registry.py` | Optional MLflow integration |

---

## D. External data (→ renquant-base-data)

| Path | Notes |
|---|---|
| `fundamentals.py` | SEC fund data fetch |
| `macro.py`, `macro_per_ticker.py`, `fred_macro.py` | Macroeconomic data (FRED) |
| `insider_trades.py` | Form-4 insider trades |
| `earnings_surprise.py` | Earnings surprise feeds |

---

## E. Misc / unclassified

| Path | Notes |
|---|---|
| `models.py` | Top-level model module (likely runtime) |
| `risk_metrics.py` | Risk computation — runtime or sim depending on caller |
| `pipeline/task_monitor.py`, `task_topup.py`, `task_trim.py` | Monitoring + sleeve mgmt — runtime |

---

## Audit findings (immediate, before any move)

1. **Two PatchTST scorer files** — `panel_pipeline/hf_patchtst_scorer.py` AND
   `panel_pipeline/patchtst_scorer.py`. Either redundant or different versions.
   **Action**: D7-style grep to find canonical + delete other.

2. **Triple-barrier duplicated** — top-level `triple_barrier.py` and
   `meta_label/triple_barrier.py`. Likely one is the lift target.

3. **`regime_hmm.py` vs `renquant_common.hmm_regime_labels`** — almost certainly
   redundant; another C8-style cleanup candidate.

4. **`decision_trace.py`** is the only top-level module that's actively used by
   BOTH live and sim. Cannot move to one; either split or keep in
   renquant-common as the writer contract.

5. **`persistence.py`** umbrella vs `renquant_pipeline.kernel.persistence` —
   verify identical (D8-style); resolve to one canonical.

---

## How to use this inventory

When doing Track C2-C5:

1. Pick **one bounded chunk** (e.g. `metrics/` — pure offline, 6 files, zero
   runtime consumers → easy lift).
2. Grep umbrella for all importers of that chunk.
3. Mirror into the target repo (`renquant-backtesting` / `-pipeline` / `-common` /
   `-base-data`) byte-identical (Phase 1).
4. Add Task/Job/Pipeline wrappers in the target repo (Phase 2).
5. Re-write the umbrella copy as a thin shim `from renquant_backtesting.metrics import *`
   so umbrella callers keep working (Phase 5).

Do **not** attempt all 170 files at once — pick one subpackage per session and
land the full Phase 1-5 cycle before starting the next.

## Suggested next-session sequence

| Order | Chunk | Reason | Size |
|---|---|---|---|
| 1 | `metrics/` → renquant-backtesting | Pure offline, smallest, zero runtime risk | 6 files |
| 2 | `walk_forward/` → renquant-backtesting | Unblocks B8-B10 lifts | 7 files |
| 3 | `meta_label/` → renquant-backtesting | Mostly sim; veto Task stays in pipeline | 9 files |
| 4 | `reconciliation/` → renquant-backtesting | Trivial | 1 file |
| 5 | `execution/backend_sim.py` only → renquant-backtesting | Cleanly identifiable as sim-side | 1 file |
| 6 | `panel_pipeline/` core → renquant-pipeline | Largest runtime piece | 16 files |
| 7 | `portfolio_qp/` → renquant-pipeline | Self-contained QP module | 7 files |
| 8 | `pipeline/` core → renquant-pipeline | THE biggest; do last, after all sim pieces are out | 60 files |
| 9 | Top-level runtime files → renquant-pipeline | preflight, kelly, sizing, etc. | ~30 files |
| 10 | Top-level shared → renquant-common | persistence, registry, etc. | ~10 files |

Estimated total: ~10 sessions of careful, Phase-1-through-5 lifts. Cannot be
parallelised easily because many depend on shared imports (a runtime module
moving might need a shim until the dependent module moves too).
