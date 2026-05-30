# Duplicates audit — multi-repo

**Last updated:** 2026-05-30 by Claude Opus 4.7 (Track D7).

This catalogues every place where two or more files do the same job in the
multi-repo. For each row: which is **canonical**, what to do with the others,
and the migration status.

The audit covers files I've created/edited this week and the ecosystem they
touch. Cross-repo duplicates that pre-date the multi-repo split (e.g. parts of
umbrella `kernel/`) are listed but not resolved here — those belong to Track C
(kernel.* split).

---

## A. Training entry points

### A1. PatchTST single-run training

| File | Where | Purpose | Status |
|---|---|---|---|
| `renquant_model_patchtst.hf_trainer.main()` | renquant-model | Standalone CLI: `python -m renquant_model_patchtst.hf_trainer --cut ... --epochs ...`. Parses args, calls `train_one()` (the 9-line delegate that runs the 4-Job Pipeline) | **Canonical** — research harness calls `train_one` in-process; `build_patchtst_wf_manifest` subprocesses this entry |
| `renquant_orchestrator.train_patchtst` | renquant-orchestrator | Driver shell — was supposed to layer above hf_trainer for the multi-repo daily, but research / WF builder both bypass it and call hf_trainer directly | **Redundant** — no current caller. Decision: either delete OR make it the single entry that all callers use (mirror `train_gbdt`'s driver/engine split). Recommended: delete after confirming no plist/cron references |

**Action**: grep cron/plists/orchestrator for `train_patchtst`. If zero callers, delete in D7-followup commit.

### A2. GBDT panel-LTR training

| File | Where | Purpose | Status |
|---|---|---|---|
| `renquant_model_gbdt.ModelTrainingJob` (Pipeline) | renquant-model | Engine — the actual training Pipeline (Task/Job-shaped) | **Canonical engine** |
| `renquant_orchestrator.train_gbdt` | renquant-orchestrator | Driver — CLI wrapper, side-label & walkforward path guards, training_runs DB writer, README auto-refresh | **Canonical driver** — calls the engine Pipeline |

**Status**: clean layering, NOT a duplicate. Document as the reference pattern other families should follow.

### A3. Walk-forward manifest builders

| File | Where | Purpose | Recipe |
|---|---|---|---|
| `RenQuant/scripts/train_walkforward_panel.py` | umbrella | The OLD WF panel-LTR builder (predates multi-repo split) | recipe `sha256:509e5d7e…` |
| `renquant_orchestrator.build_wf_manifest` | renquant-orchestrator | NEW GBDT WF manifest builder (calls `train_gbdt` per cutoff) | recipe `sha256:c3cd6b47…` |
| `renquant_orchestrator.build_patchtst_wf_manifest` | renquant-orchestrator | PatchTST analog (calls `hf_trainer` per cutoff) | per-config |

**Status**: NOT duplicates — different recipes (`feature_norm_kind` + `feature_source_contract_keys` differ between trainers). Both must exist to evaluate candidates trained by either path. The 2026-05-30 manifest-mismatch incident proved this.

**Action**: the umbrella `train_walkforward_panel.py` becomes a thin shim that delegates to `renquant_orchestrator.build_wf_manifest` once Track C unblocks. Until then it stays.

---

## B. CLI / daily entry points

### B1. Daily orchestration (intentional rollback split)

| File | Where | Purpose | Used when |
|---|---|---|---|
| `RenQuant/scripts/daily_104.sh` | umbrella | Bash wrapper invoking the multi-repo orchestrator (or umbrella runner) | `RQ_DAILY_RUNNER=multirepo` (default) → calls `daily_multirepo.py`; `RQ_DAILY_RUNNER=umbrella` → calls `python -m live.runner` directly |
| `renquant_orchestrator.daily_multirepo` | renquant-orchestrator | Python orchestrator: runs the daily flow against pinned subrepos | Default path |

**Status**: NOT a duplicate — explicit rollback design (`feature_no_pr_verbal_merge` memory documents this). Keep both. The bash wrapper is the env-driven router.

### B2. WF gate runner (Phase 1-5 migration)

| File | Where | Status |
|---|---|---|
| `RenQuant/scripts/run_wf_gate.py` | umbrella | **Authoritative + live** until Phase 5 |
| `renquant_backtesting.wf_gate.runner` | renquant-backtesting | Phase 1 byte-identical copy; Stages 1, 2, 6 already lifted into `pipelines.py` (10 of 14 Tasks runner-free) |

**Status**: intentional Phase 1-5 transition. NOT a stale duplicate.

### B3. Sim driver + ledger

| File | Where | Status |
|---|---|---|
| `RenQuant/scripts/run_sim_104.py` | umbrella | **Authoritative + live** |
| `renquant_backtesting.wf_gate.sim_driver` | renquant-backtesting | Phase 1 copy |
| `RenQuant/scripts/sim_trade_ledger.py` | umbrella | **Authoritative + live** |
| `renquant_backtesting.wf_gate.sim_ledger` | renquant-backtesting | Phase 1 copy |

**Status**: same as B2 — Phase 1 staging.

---

## C. Shared utilities (cross-repo dupes)

### C1. `config_consistency`

| File | Where | Status |
|---|---|---|
| `renquant_common.config_consistency` | renquant-common | **Canonical** (multi-repo code imports here) |
| `RenQuant/backtesting/renquant_104/kernel/config_consistency.py` | umbrella | Live (umbrella internals import via `kernel.config_consistency`) — Track C blocker |
| ~~`renquant_pipeline.kernel.config_consistency`~~ | renquant-pipeline | **DELETED 2026-05-30 (D8, commit d3dbed8)** — was unused dup |

**Action**: D8 done. Umbrella copy stays until Track C.

### C2. Subrepo registry rendering

| File | Where | Purpose |
|---|---|---|
| `RenQuant/RENQUANT_REPOS.md` (the umbrella one) | umbrella | **Authoritative** (canonical multi-repo SOP) |
| Each subrepo's `RENQUANT_REPOS.md` | each subrepo | **Auto-generated** from `subrepos.lock.json` via `scripts/sync_subrepo_docs.py` — `subrepo_doctor.py` enforces byte-equivalence |

**Status**: NOT a duplicate — auto-generated single-source pattern. Don't hand-edit subrepo copies.

### C3. `walk_forward_splits`, `hmm_regime_labels`, `purged_cv`

| File | Where | Status |
|---|---|---|
| `renquant_common.walk_forward_splits` | renquant-common | **Canonical** (used by renquant-model trainers) |
| `renquant_common.hmm_regime_labels` | renquant-common | **Canonical** |
| `renquant_common.purged_cv` | renquant-common | **Canonical** |
| Umbrella copies of these | umbrella | Live (umbrella internals still import) — Track C blocker |

**Action**: Track C work. Out of D7's scope.

---

## D. Test placement

| Location | What's there | Status |
|---|---|---|
| `RenQuant/tests/` (umbrella) | The original test suite — covers umbrella `kernel/`, `live/`, `training_panel/`, `scripts/` | Live, authoritative |
| Each subrepo's `tests/` | Tests for that subrepo's package (renquant-common, renquant-model, etc.) | Live, authoritative for the package they cover |

**Status**: NOT duplicates — by-repo-ownership. After Track C, umbrella tests covering moved modules also move.

---

## E. README "Latest models" auto-refresh

| File | Where | Purpose |
|---|---|---|
| `renquant-model/README.md` | renquant-model | Auto-generated `<!-- LATEST_MODELS:START/END -->` block |
| `renquant-model/scripts/refresh_readme_latest_models.py` | renquant-model | The generator (D3-tested) |
| `train_gbdt._record_and_refresh` | renquant-orchestrator | Calls the refresh script post-train (E1) |
| `sequence_training.RecordTrainingRunTask` | renquant-model | Same hook for PatchTST training (E2) |

**Status**: NOT duplicates — generator + two driver-side hooks. The DB (`training_runs`) is the single source.

---

## How to use this audit

1. **Before adding a new training script** — check section A. If it overlaps an existing one, extend the existing OR layer above it (driver/engine pattern).
2. **Before lifting code from umbrella** — check section C. If it's already in `renquant-common`, just import from there. If it's in two subrepos, fix it (the D8 pattern).
3. **Before creating a new "umbrella-script alias in subrepo"** — check section B. Phase 1 staging is fine; permanent dual-life is not.

## Open items

- **A1 dead-code check**: confirm `renquant_orchestrator.train_patchtst` has zero live callers; delete if so.
- **A3 / B2-B3 / C1 umbrella copies**: blocked on Track C (kernel.* split).
- **D6** (rename `train_one` → `train_single_run`): cascade-risk; defer until after Track A unblocks daily.
