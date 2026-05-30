# Multi-repo task plan — durable cross-session tracker

This is the master task list for the renquant multi-repo work. Each row is
discrete, bounded, and survives across sessions. Update status here (not in
ephemeral chat task IDs) so future Claude sessions can pick up cleanly.

**Last updated:** 2026-05-30 by Claude Opus 4.7

---

## Track A — Unblock daily (highest priority)

Goal: stop `BUY-BLOCKED` on the live runner.

| ID | Task | Status | Where | Notes |
|---|---|---|---|---|
| A1 | `build_patchtst_wf_manifest` BG run completes | 🟡 running | `b8kynsdcl` | 10 cutoffs × ~15min on MPS, cwd fix in 93f9f38 |
| A2 | Pin new manifest in `strategy_config.shadow.json::walkforward.manifest_path` | ⏳ | umbrella | Required so the gate picks it up |
| A3 | Re-run `run_wf_gate.py --artifact A.pt --strategy-config strategy_config.shadow.json` | ⏳ | umbrella | A2 prerequisite |
| A4 | Verify A's `wf_gate_metadata` stamps real Sharpe + sanity numbers | ⏳ | A artifact | If passes → A4a; if fails → diagnose |
| A4a | If sharpe positive 3/3 cuts AND beats SPY 1/3+ AND sanity placebo ≈0 → flip prod `strategy_config.panel_scoring.artifact_path` to A | ⏳ | umbrella | Per §5.13.4a Tier 3 gate |
| A5 | Run `daily_104.sh` e2e on LIVE alpaca → confirm BUY-BLOCKED is gone | ⏳ | umbrella | Verification of unblock |
| A6 | Resume Phase 2 multi-seed confirm (was killed for A's manifest) | ⏳ | renquant-model | `b3y8vo7yr`-style relaunch |

---

## Track B — wf_gate move (renquant-backtesting)

Goal: move the WF gate + sim ledger ecosystem from umbrella to its proper repo.

| ID | Task | Status | Where | Notes |
|---|---|---|---|---|
| B1 | Phase 1 — copy runner/sim_driver/sim_ledger byte-identical | ✅ done | `2b6985c` | |
| B2 | Phase 1 sweep — copy 18× wf_/walkforward + 18× analyze + 1 sim + 2 lean | ✅ done | `7decced` | 42 files total, byte-identical |
| B3 | Phase 2 — Task/Job/Pipeline scaffold (6 Jobs / 14 Tasks) | ✅ done | `d52f102` | 7 tests pin shape |
| B4 | Phase 3a — lift `LoadArtifactTask` body via `artifact_loader.py` | ✅ done | `4b1dfbb` | 7 tests pin behaviour; runner copy keeps inline for Phase 4 oracle |
| B5 | Phase 3b.3 — `DeriveConfigTask` + `matching_manifest_for_recipe` lifted (DI strategy_dir) | ✅ done | next commit | 5 new tests pin preferred-required policy + auto-discovery + tiebreak |
| B6 | Phase 3c — `CheckConfigParityTask` wires `evaluate_wf_config_parity` with DI strategy_dir | ✅ done | next commit | 3 tests pin skip-on-missing-config / skip flag / no strategy_dir |
| B7a | Phase 3d.1 — lift recipe_match helpers (semantic_params, recipe_projection, recipe_fingerprint) | ✅ | `wf_gate/recipe_match.py` | runner copy keeps inline for byte-equivalence smoke |
| B7b | Phase 3d.2 — lift `manifest_recipe_usage` via DI (`strategy_dir` injected through ctx) | ✅ done | next commit | 7 tests pin behaviour incl. relative URI resolution + per-sample diff |
| B8 | Phase 3e — lift `RunWfSimTask` (calls run_sim_cut × 3, ThreadPool when --jobs > 1) | ⏳ | same | Most complex, kernel.* deps |
| B9 | Phase 3f — lift `RunTradeContractTask` + `RunTradeMonotonicityTask` | ⏳ | same | runner.run_trade_*_gate stays as helper |
| B10 | Phase 3g — lift `RunSanityBatteryTask` (wraps run_sanity_battery + _score_manifest_sanity) | ⏳ | same | Already updated for hf_patchtst dispatch in 82f928b |
| B11 | Phase 3h — Stamp/Verdict tasks wired (write_artifact_payload + ctx-driven verdict) | ✅ done | next commit | 7 tests pin behaviour incl. binary preservation + skip-on-None |
| B12 | Phase 4 — smoke vs umbrella: byte-equivalent metadata on a known fixture | ⏳ | `tests/wf_gate/` | Critical gate before any flip |
| B13 | Phase 5 — flip callers: weekly_wf_promote.sh, daily_104.sh, orchestrator → backtesting package | ⏳ | umbrella shells + renquant-orchestrator | Umbrella scripts become thin shims |

---

## Track C — kernel.* split (the biggest open architectural question)

Goal: `kernel/*` lives in umbrella; should split between renquant-pipeline (runtime) and renquant-backtesting (sim). This is the blocker for Phase 3 lifts that touch `kernel.panel_pipeline.*`, `kernel.walk_forward.*`, `kernel.preflight.*`.

| ID | Task | Status | Notes |
|---|---|---|---|
| C1 | Inventory umbrella `kernel/` modules and classify (runtime / sim / shared) | ✅ done | `doc/arch/kernel-inventory.md` — 170+ files classified into 5 buckets (pipeline/backtesting/common/base-data/misc) + 10-step suggested sequence + 5 immediate dup-audit findings |
| C2 | Move `kernel.pipeline.*`, `kernel.walk_forward.loader`, `kernel.panel_pipeline.panel_scorer`, `kernel.preflight` → renquant-pipeline | ⏳ | These are runtime (already partially in pipeline repo) |
| C3 | Move `kernel.sim.*`, `kernel.exits.*`, `kernel.regime.*` → renquant-backtesting | ⏳ | Sim-specific |
| C4 | Keep umbrella `kernel/` as shim that re-exports for back-compat (Phase 5 friendly) | ⏳ | Avoids breaking everything at once |
| C5 | After C2: kernel.* imports in `wf_gate/runner.py` resolve from package, not umbrella | ⏳ | Unblocks B8-B10 lifts |
| **C2.1** | **kernel/metrics/ → renquant-backtesting** (6 files) | ✅ first chunk done | next commit | byte-identical copy; umbrella callers unchanged; 7 import-lift tests pass |
| **C2.2** | **kernel/walk_forward/ → renquant-backtesting** (7 files) | ✅ done | next commit | unblocks B8-B10 once Phase 5 flips; 8 import-lift tests pass |
| **C2.3** | **kernel/reconciliation/ → renquant-backtesting** (1 file) | ✅ done | next commit | 3 import-lift tests pass; smallest chunk done |

---

## Track D — self-audited debt (5.13.17)

Debt I personally introduced this session that needs fixing.

| ID | Task | Status | Where | Severity |
|---|---|---|---|---|
| D1 | `build_wf_manifest.py` (GBDT) should be Task/Job/Pipeline per §1c | ⏳ | renquant-orchestrator | minor — only ~100 lines |
| D2 | `build_patchtst_wf_manifest.py` same | ⏳ | renquant-orchestrator | minor |
| D3 | Tests for `refresh_readme_latest_models.py` | ✅ done | next commit | 5 tests: write block, replace block, no-rows, --limit, missing-db |
| D4 | Tests for both `build_wf_manifest` drivers (GBDT + PatchTST) | ✅ done | next commit | 4 GBDT + 5 PatchTST tests, monkeypatch subprocess.run |
| D5 | `PatchTstStatefulScorer.bootstrap_from_history(history_df)` — warms buffer to seq_len-1 | ✅ done | next commit | 5 new tests; caller-driven (load stays clean) |
| D6 | `train_one` rename → `train_single_run` | ⏳ | renquant-model + research.py + tests | cosmetic but cascades |
| D7 | Duplicates-audit doc with canonical/redundant classification | ✅ done | next commit | `doc/arch/duplicates-audit.md` — A1 confirmed: orchestrator/train_patchtst is dead code (delete in followup); A3/B2-B3/C1 umbrella copies blocked on Track C |
| D8 | Delete unused renquant-pipeline/.../kernel/config_consistency.py (canonical = renquant-common; umbrella copy is Track C blocker) | ✅ done | next commit | byte-identical to common, zero importers — safe delete |

---

## Track E — observability / DB tracking (user requested 2026-05-30)

| ID | Task | Status | Where | Notes |
|---|---|---|---|---|
| E1 | `training_runs` DB writer wired into train_gbdt | ✅ done | `48e9838` | |
| E2 | Same for train_one (PatchTST) | ✅ done | `8997151` (RecordTrainingRunJob) | |
| E3 | README "Latest models" auto-refresh | ✅ done | `8c760f5` + `48e9838` | |
| E4 | `docs/training_pipelines.md` — every hyperparameter | ✅ done | `00b3da9` | |
| E5 | `pipeline_runs` writer audit — already populated by live runner (38,360 rows in runs.alpaca.db, 1089 in sim_runs.db) | ✅ resolved | next commit | gap was misstated; existing infra works. New schema fields (buy_blocked / skip_buys / bear_only / counters_json) already in DDL |
| E6 | `--training-window-years` CLI added to train_gbdt + hf_trainer, threaded into record_training_run | ✅ done | next commit | schema field finally populated; diagnostic only (no training behaviour change) |

---

## Track F — promotion + governance gates (existing CLAUDE.md)

| ID | Task | Status | Notes |
|---|---|---|---|
| F1 | Prod GBDT carries `promotion_status: gated_buys` | ✅ done | `be0a087` |
| F2 | 5/27 audit memory correction (verdict was NaN not +0.729) | ✅ done | memory file edited |
| F3 | A's sidecar `promotion_status: candidate` | ✅ done | A turn |
| F4 | If A passes A4: stamp `promotion_status: active` on A; demote prod GBDT to shadow | ⏳ | umbrella | The actual promote moment |

---

## Track G — research backlog (lower priority)

| ID | Task | Status | Notes |
|---|---|---|---|
| G1 | Phase 2 multi-seed confirm (B_tuned + C_xstock × 5 seeds × 8ep) | 🟡 paused | Killed at 25/50 for A's manifest; resumable |
| G2 | Phase 1 (Box-Behnken HP optimization) on the winner | ⏳ | Only after Track A passes |
| G3 | PatchTST features: try alpha158_no_sentiment vs current 169 | ⏳ | E_drop_senti already shown +0.011 win for XGB |
| G4 | Held-out TEST split scoring for A (not just val) | ⏳ | Removes selection optimism |

---

## How to pick the next task

1. **If daily is gated**: do Track A in order. Track A → unblock buys.
2. **If daily is running**: do Track B + Track D in parallel — refactor + debt cleanup.
3. **Never**: skip Track A items if daily is gated. Refactor is not a substitute for unblock.

## Cross-session continuity rules

- Update status here BEFORE ending a session.
- Use 🟡 for tasks running in background (manifest builds, training runs).
- Use ✅ with the commit hash when a task lands.
- Never lose the "currently running BG" state — that's the only way to resume cleanly.

_**2026-05-30 Stage 1 + Stage 2 + Stage 6 完全 lifted**: 10 of 14 Tasks runner-independent (Config: Load/Derive/CheckConfigParity; Recipe: Resolve/ValidateRecipe; Stamp: Assemble/Stamp/EmitVerdict + 2 placeholders for inner-config). Only Stages 3-5 (WF sim + trade gates + sanity battery) still need lifts; those are Track C blocked on `kernel.*` split._

---

## Process lessons learned (2026-05-30 session)

1. **Always `pytest` BEFORE `git commit`.** 5 fix-forward commits this session shared one root cause: the commit-then-test sequence. Lesson logged.
2. **Phase 1 lifts: byte-equivalence is the invariant**, not full importability. Modules whose `__init__.py` uses absolute `from kernel.x import` re-exports fail to import in the package context — that's expected pre-Phase-5. Tests must soft-skip on `ModuleNotFoundError('kernel')`, not assert.
3. **Heredoc + cwd reset**: when using `.venv/bin/python - <<'PY'` inside a multi-cd bash, the cwd resets between calls. Always use absolute paths inside heredocs.
4. **Grep filters for string literals matter.** A grep for `import X` misses occurrences of `X` in lists / docstrings. When doing a `git rm`-style delete, also grep for the bare module name.
5. **Commit + push + advance pin** is 3 ops, not 1. Each must be verified before moving on (commit can succeed while push fails on network reset).
