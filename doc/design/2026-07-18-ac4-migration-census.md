# AC4 migration census — every reader/writer of the flat 104 serving pair

Date: 2026-07-18
Status: census document — the RFC §3 **P2 entry criterion** for
`doc/design/2026-07-17-artifact-bundle-transactionality.md` (GOAL-5 AC4).
Scope: read-only sweep of ALL sibling checkouts (umbrella RenQuant,
renquant-{orchestrator,pipeline,backtesting,model,strategy-104,common,
base-data,execution,artifacts} plus the deployed `-run` variants).

## 1. Scope, ground truth, method

**The flat pair** (umbrella working tree, git-TRACKED — see W4-g):

- `backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json`
- `backtesting/renquant_104/artifacts/prod/panel-rank-calibration.json`

**Side-file conventions observed live in `artifacts/prod/`** (all census'd
as part of the pair surface): `*.weekly_<runid>.staging.json`,
`panel-rank-calibration.json.staging-<runid>.json`,
`*.weekly_rollback_<date>.json`, `panel-rank-calibration.monthly_rollback_<date>.json`,
`panel-ltr.alpha158_fund.previous.json`, `*.incoming.json` (transient),
`panel-rank-calibration.json.accepted_receipt-<runid>.json`,
`*.bak*` / `.bak_restamp` / `.pre-train.json`, `_rejected_calibrators/`
(quarantine), `_acceptance_log/` (reject/rollback archive), and orphaned
ad-hoc copies `*.pre-v1-restamp-*.json`, `*.pre-binding-fix-*.json`
(see W4-e).

**Config indirection** — every runtime resolution goes through these keys
(pinned `renquant-strategy-104/configs/strategy_config.json`; umbrella
working copy `backtesting/renquant_104/strategy_config.json` mirrors them):

| key | value |
|---|---|
| `ranking.panel_scoring.artifact_path` | `artifacts/prod/panel-ltr.alpha158_fund.json` |
| `ranking.panel_scoring.global_calibration.artifact_path` | `artifacts/prod/panel-rank-calibration.json` |
| `panel_ltr.artifact_path` (training-side alias) | `artifacts/prod/panel-ltr.alpha158_fund.json` |
| alias accepted by resolvers: `panel_ltr.calibrator_artifact_path` | (calibrator) |
| `configs/xgb_prod_artifact_manifest.json:29,34` | binds both, roles `production_primary` / `production_primary_calibrator` |

Shadow configs (`strategy_config.shadow*.json`) point their **scorer** keys
at the PROD panel but override the calibrator to `artifacts/shadow/...` —
the shadow arm is a reader of the prod panel only.

Method: filename/path-fragment/config-key grep across all checkouts
(read-only, no git commands in any live tree), then manual read of every
writer call site. Hits inside `artifacts/` data dirs (WF corpora, sim
bundles, snapshots) and doc/evidence files are data, not code, and are
excluded. Classifications: **{migrates to bundle API | keeps flat VIEW |
test-only n/a}**; per RFC §3, views are read-only and any writer must go
through the publication API.

## 2. WRITERS (the headline table)

### 2.1 W1 — weekly WF promote chain (RFC §1 writer 1) — **migrates to bundle API**

Execution context: launchd `com.renquant.weekly-wf-promote.plist`
(Sat 04:00 PT); also reached via `com.renquant.retrain-panel104.plist`
(Sunday compat wrapper `retrain_panel.sh` → delegates) and
`com.renquant.conditional-retrain104.plist` (`conditional_retrain_104.sh`
SPY/VIX anomaly trigger → delegates). All three converge on the same
writer chain:

| file:line (repo) | write |
|---|---|
| `RenQuant/scripts/weekly_wf_promote.sh:225,229` | `cp` active pair → `*.weekly_rollback_<date>.json` |
| `RenQuant/scripts/weekly_wf_promote.sh:246-250` | staging pair via `daily_retrain_alpha158_fund.sh --xgb-artifact-out/--calibrator-out` |
| `renquant-orchestrator/src/renquant_orchestrator/retrain_alpha158_fund.py:1447-1452` (defaults) `:850-852` (atomic publish) | the PRIMARY retrain runner (`daily_retrain_alpha158_fund.sh` multirepo mode) — **defaults are the flat prod pair**; the weekly chain redirects to staging via `--xgb-artifact-out/--calibrator-out` |
| `RenQuant/backtesting/renquant_104/training_panel/daily_retrain_alpha158_fund.py:71-72,228-237,254-261` | the umbrella ROLLBACK twin (`RQ_RETRAIN_RUNNER=umbrella`): absolute prod-path constants; subprocess `scripts/train_production_model.py`, `scripts/fit_calibrator_alpha158_fund.py` |
| `RenQuant/scripts/run_wf_gate.py:3315` (via `:685-692`) | **in-place** `wf_gate_metadata` stamp on `--artifact` (the staging candidate by convention; no prod default but WILL rewrite prod if pointed there) |
| `renquant-backtesting/src/renquant_backtesting/wf_gate/stamp_walkforward_fingerprints.py:142,176` (+ umbrella twin `scripts/stamp_walkforward_fingerprints.py`) | **in-place** manifest-driven `config_fingerprint` stamp on scorer artifacts and `scorer_model_content_fingerprint` **binding stamp on calibrators** (weekly Step 3.5) |
| `RenQuant/backtesting/renquant_104/kernel/model_acceptance.py:749 promote()` / multirepo primary `renquant-backtesting/.../forensics/model_acceptance.py:850` | the active swap: `shutil.copy2` staging→temp, `os.replace` active→`.previous.json`, `os.replace` temp→active |
| `RenQuant/scripts/weekly_wf_promote.sh:358-362` | calibrator swap: copy staging→`.incoming.json`, `promote()` panel, `os.replace` incoming→active calibrator |

This is exactly the RFC's §2.3 writer-protocol customer: staging, verdict,
promote, per-file rollback — all of it becomes one bundle publication.

### 2.2 W2 — monthly calibrator refresh (RFC §1 writer 2) — **migrates to bundle API**

Execution context: launchd `com.renquant.monthly-calibrator-refresh.plist`
(1st of month 03:00 PT).

| file:line | write |
|---|---|
| `RenQuant/scripts/monthly_calibrator_refresh.sh:194` | archival `cp`+`mv` active calibrator → `panel-rank-calibration.monthly_rollback_<date>.json` |
| `renquant-model/src/renquant_model_gbdt/fit_calibrator_alpha158_fund.py` (invoked `:216-219` with `--out $STAGING_CAL`) | writes staging `panel-rank-calibration.json.staging-<runid>.json` (serializer primitive: `renquant-model/src/renquant_model_common/global_calibrator.py:127 save()`) |
| `RenQuant/scripts/monthly_calibrator_atomic_swap.py:195` | `os.replace` staging → **prod calibrator** (single-file atomic publish, post 3/3b gates) |
| `RenQuant/scripts/monthly_calibrator_atomic_swap.py:159` | acceptance receipt `*.accepted_receipt-<runid>.json` |
| `RenQuant/scripts/monthly_calibrator_atomic_swap.py:219-225` | quarantine → `_rejected_calibrators/<ts>_REJECTED_*` + reason file |

Path resolution is fully config-driven (`ranking.panel_scoring.artifact_path`
+ `...global_calibration.artifact_path`, `monthly_calibrator_refresh.sh:153-176`).
Note the RFC §1 diagnosis verbatim: this is an atomic **per-FILE** swap of
one half of the pair.

### 2.3 W3 — restamp tools (RFC §1 writer 3) — **migrate to bundle API** (re-worked, see blockers B3)

| file:line | write | context |
|---|---|---|
| `RenQuant/scripts/restamp_prod_fingerprint.py:100-110` | **in-place** re-stamp of the ACTIVE panel (path from `ranking.panel_scoring.artifact_path`) + `.bak_restamp` copy | manual repair (sector-map fingerprint drift) |
| `RenQuant/scripts/stamp_panel_contract_missing_fields.py:196,224` | **in-place** contract-field stamp; `--artifact` **defaults to the prod panel** | manual repair |
| `renquant-backtesting/.../wf_gate/stamp_walkforward_fingerprints.py:142` | calibrator **binding** stamp (also listed in W1; manifest-driven, hits the pair only if a manifest names it) | weekly Step 3.5 / manual |
| `renquant-orchestrator/src/renquant_orchestrator/m6_restamp.py:227` (+ `scripts/prestamp_legacy_fingerprints.py:587`) | metadata **sidecar** writes for the M6 fingerprint migration — targets model metadata sidecars, not the pair files themselves; census'd here because it is the tooling family that performed the 07-15 v1 re-stamp campaign | orchestrator CLI, manual |
| `RenQuant/scripts/stamp_patchtst_fingerprint.py:84` | PatchTST `.metadata.json` sidecars (shadow family) — **not the pair**; n/a for this census | shadow repair |

### 2.4 W4 — manual ops / recovery (RFC §1 writer 4) — **break-glass tool required**

| id | surface | evidence | classification |
|---|---|---|---|
| W4-a | `RenQuant/scripts/manual_promote.sh:33,73` | emergency promote (`RQ_ALLOW_NO_WF=1`) of `artifacts/prod/panel-ltr.staging.json` → active panel via `kernel.model_acceptance.promote` | becomes `bundle_breakglass` |
| W4-b | `RenQuant/backtesting/renquant_104/kernel/model_acceptance.py:868 rollback()` | operator rollback: `.previous.json` → active, archive to `_acceptance_log/` | becomes `bundle_breakglass --rollback-to` |
| W4-c | bare `RenQuant/scripts/daily_retrain_alpha158_fund.sh` (no args) | BOTH runners default to the prod pair: multirepo primary `renquant-orchestrator/src/renquant_orchestrator/retrain_alpha158_fund.py:1447-1452` and umbrella rollback twin `training_panel/daily_retrain_alpha158_fund.py:71-72`; a bare manual invocation retrains STRAIGHT ONTO the live pair with no staging, no gate | blocker B4 |
| W4-d | `RenQuant/scripts/train_104.py:434` (defaults `:311-316`, snapshot `:350`) | `promote()` to the config-resolved prod path, gated by `RQ_ALLOW_NO_WF=1`; legacy-trainer path additionally gated by `RQ_ALLOW_LEGACY_PANEL_TRAINER=1` | becomes `bundle_breakglass` (gate flags map to authorization.source) |
| W4-e | **ad-hoc session edits (no tool)** | orphan side files `panel-ltr.alpha158_fund.pre-v1-restamp-20260715T212457Z.json`, `panel-rank-calibration.pre-v1-binding-restamp-20260715T213443Z.json`, `panel-rank-calibration.pre-binding-fix-20260716T203819Z.json`, `panel-rank-calibration.json.bak-20260714-143011` exist in prod with **no committing code in ANY checkout** (literal-suffix sweep) — these are the 07-14→16 incident's hand-edits | blocker B2 |
| W4-f | **documented recovery runbooks** | `renquant-orchestrator-run/doc/ops/2026-07-03-p1-remediation-landing.md:13-15` — `git checkout origin/main -- <prod panel>` restore + calibrator refit/install onto the prod path; its verifier script (`verify_prod_scorer_restore_20260703.py`) exists in the umbrella but the runbook's own copy was transient/uncommitted | becomes `bundle_breakglass --rollback-to` |
| W4-g | **git working-tree operations** | the pair is git-tracked in the umbrella repo (`git ls-files` confirms); `git checkout/reset/pull` on the live tree rewrites both files with zero authorization/audit — the 2026-07-08 incident class; `scripts/backup_to_github.sh` explicitly relies on "already committed" for artifact backup | blocker B1 |
| W4-h | **side-file GC** | `renquant-orchestrator/src/renquant_orchestrator/retention_policy.py:71-88,209` — `prune-artifacts --execute` DELETES `*.weekly_*.staging.json` / `*_rollback_*.json` side files (manual CLI, dry-run default) | migrates to bundle-store GC (RFC §2.6 — reference-rooted, lock-serialized) |

### 2.5 Defunct-legacy writer cohort — **retire/redirect** (targets no longer exist)

The legacy flat dir `backtesting/renquant_104/artifacts/{panel-ltr.json,
panel-rank-calibration.json}` is empty today, so these are dead paths that
would silently **recreate flat orphans** if run; they must be retired or
redirected at P2, not migrated:

`RenQuant/scripts/select_best_model.py:267,272` (`--promote` targets flat
`artifacts/panel-ltr.json`); `sunday_panel_sweep.py:59-63,93,107`;
`auto_revert_b1_regression.sh:142-150` (2026-04 ablation revert, legacy
dir); `compare_panel_backends.py:142,216,249`; `train_panel_model.py:75-77,171`;
`fit_panel_calibrator.py:160-167,388` (legacy default + `--force`
overwrite-guard from the 2026-05-05 triple-overwrite incident);
`model_dashboard.py:61-62,97-98` (reader); umbrella
`kernel/panel_pipeline/model_registry.py:106` and pipeline
`kernel/panel_pipeline/model_registry.py:112` (`train_cmd` builds
`--output <flat artifacts/>panel-ltr.alpha158_fund.json`);
`dagster_renquant/_paths.py:26-27` + `assets/training.py:60-76` (Dagster
asset graph validates the LEGACY flat dir — stale surface, see B5).

### 2.6 Writers per repo (summary)

| repo | writers of the pair | count |
|---|---|---|
| RenQuant umbrella | weekly chain (sh + retrain module + run_wf_gate + model_acceptance), monthly chain (sh + atomic_swap), restamp x2, manual x4 (incl. bare-retrain, train_104), git surface, ad-hoc | ~13 surfaces |
| renquant-backtesting | `forensics/model_acceptance.py promote()/rollback()/reject()` (paths caller-injected), `wf_gate/stamp_walkforward_fingerprints.py` (in-place stamps) | 2 |
| renquant-model | serializer primitives only (`global_calibrator.py:127 save`, `panel_data.py:321 WriteArtifactTask`) — path always caller-supplied, no prod default | 0 direct |
| renquant-pipeline | **zero** writes (retrain `train_cmd` builds a legacy-path subprocess command; never on daily/live) | 0 |
| renquant-orchestrator | `retrain_alpha158_fund.py` (primary trainer; prod defaults when un-redirected) + `retention_policy.py` (side-file deleter); m6_restamp/prestamp write metadata sidecars only | 2 |
| renquant-{strategy-104,common,base-data,execution,artifacts} | none (model repo exposes serializer primitives; artifacts repo has no write path today) | 0 |

**No fifth automated writer exists.** Every automated mutation path
converges on `model_acceptance.promote()` or
`monthly_calibrator_atomic_swap.py` (with `retrain_alpha158_fund` /
`fit_calibrator_alpha158_fund` producing the bytes); everything else is
manual-class, in-place stamping, or side-file GC.

## 3. READERS

### 3.1 renquant-pipeline (runtime) — **migrates to bundle API** (the RFC §5 reader)

| file:line | reads | context |
|---|---|---|
| `kernel/panel_pipeline/job_panel_scoring.py` `LoadScorerTask` (`:866`, `_resolve_artifact_path :870-881`, loads `:932,938,1004,1010`) | panel | daily live + sim + shadow |
| `kernel/panel_pipeline/job_panel_scoring.py` `LoadGlobalCalibrationTask` (`:2521`, `:2534-2591`; refuses to default `:2566-2574`, fail-closed) | calibrator | daily live + sim + shadow |
| `kernel/panel_pipeline/job_panel_scoring.py::_assert_calibrator_matches_scorer` (`:2240`; called `:2558,2579,2647,2677`) | pair binding | **the runtime matcher the RFC exports as `bundle_contract.validate_pair`** |
| `kernel/preflight.py:344,389,454,581,707,788,910,1145,1294` (panel) `:1751,1902` (calibrator) + `kernel/preflight_pipeline/tasks/{artifact:42,90,154, calibrator:39,65, config_fingerprint:41, feature_coverage:37, gate:50,198, run_id:33, watchlist:35}` | pair (prod-literal defaults) | preflight — keeps flat VIEW initially, API at P3 |
| `kernel/pipeline/pp_training_full.py:260,267` | config-shaping (injects prod default) | sim/prod isolation |
| `kernel/panel_pipeline/regime_ensemble_scorer.py:16-19` | hardcoded per-regime prod variants (sibling family) | regime arm |
| `persistence_backup_check.py`, `scripts/shadow_replay_bl1_recenter.py`, `kernel/walk_forward/{loader,lean_guard}.py`, `decision_trace.py`, `portfolio_qp/tasks.py` | downstream consumers / string refs | n/a-adjacent |

### 3.2 RenQuant umbrella kernel (legacy twin of the pipeline reader) — **keeps flat VIEW** until the P3 retirement of the umbrella kernel path

`kernel/preflight.py` (same task set, `:343-1712`), `kernel/preflight_pipeline/tasks/*`,
`kernel/panel_pipeline/{job_panel_scoring.py:205-285, calibration.py:265-376}`,
`adapters/runner.py:2126-2127` (live runner), `adapters/sim.py:640` (sim
replay; isolation is config-override, NOT a code copy — `adapters/sim_artifacts.py`
is read-only metadata and does not redirect prod), `kernel/artifact_snapshot.py:90-116`
(backup copytree — reader+backup-writer of copies, not the pair),
`dagster_renquant/assets/training.py:60-76` (stale flat-path existence stubs).

### 3.3 Diagnostics / experiment readers (umbrella scripts) — **keep flat VIEW**

- Config-resolved health readers on the daily/weekly/monthly chains:
  `smoke_test_model.py:67,183`, `build_dashboard.py:172-187`,
  `run_sanity_checks.py:86`, `preflight_analyzer.sh:43-44` (mtime staleness),
  `diagnose_funnel.sh` (legacy names, display).
- Literal-prod-path experiment readers (read `feature_cols`/fingerprints,
  write their OWN artifacts elsewhere): `train_ngboost_alpha158_fund.py:35`,
  `train_quantile_head_alpha158_fund.py:55`, `train_ngboost_proper.py:255`,
  `train_ngb_vol_adjusted_label.py:47`, `train_qhead_neural.py:155`,
  `train_qhead_catboost_multiquantile.py:36`, `train_quantile_head_rawlabel.py:44`,
  `qhead_{purged_baseline:67,neural_sanity:35,phaseA_experiments:104}.py`,
  `ngb_proper_placebo.py:54`, `train_production_model_lgbm.py:85`,
  `test_insider_features.py:108`, `long_short_prereq_gate.py:41`,
  `diagnose_calibrator_saturation.py:47,51`, `eval_prod_vs_shadow.py:230`,
  `run_a3_ngboost_retrain.sh:41`, `verify_prod_scorer_restore_20260703.py:59-63`
  (incident verifier), `kelly_param_validation.py:67`, `fetch_macro_factors.py:69`.
- WF/sim writers to NON-prod dirs (n/a for the pair):
  `train_walkforward_panel.py:184` (+ backtesting twin `wf_gate/train_walkforward_panel.py:87-96`),
  `retrain_prod_truly_oos.py:70`, `walk_forward_*.sh` (/tmp),
  `fit_walkforward_calibrators.py` (per-cut `calibrator_root/<cutoff>/panel-rank-calibration.json`),
  `train_{regime_ensemble,per_regime_panel,per_regime_walkforward,panel_alpha158_xgb,panel_linear,horizon_blender*}.py`,
  `run_{stage3_greedy,b2_on_stage3_winner,m1_chain_overnight,topdown_wl_max,wl_size_sweep,feature_ablation_4way,b1_2_b1_3_chain}.sh/.py`,
  `retrain_alpha158_linear.sh` (linear variant family).

### 3.4 renquant-orchestrator — readers (provenance/monitoring), **migrate provenance fields at P2, monitors keep flat VIEW**

Launchd wiring from `ops/launchd_manifest.json`; the deployed
`renquant-orchestrator-run` checkout's pair-referencing file set is
byte-identical to canonical (verified) — no hand-dropped scripts touch
the pair.

| file:line | reads | context |
|---|---|---|
| `src/renquant_orchestrator/weekly_promote_monitor.py:64-76` | globs `*.weekly_*.staging.json` in prod dir | scheduled `run-job weekly_promote_monitor` (read-only liveness) |
| `src/renquant_orchestrator/scorer_identity_monitor.py:146,257-289` | prod + staging pair identity | launchd `com.renquant.rq104-scorer-identity` |
| `src/renquant_orchestrator/model_freshness_monitor.py:258` / `model_freshness_enforcer.py:193-227,299` | panel age / rglob `panel-ltr*.json` | observe-only freshness governance |
| `src/renquant_orchestrator/{native_live_context.py:215, native_context_hydration.py:341, model_bundle.py:52, intraday_session_inputs.py:130-151}` | config-key resolution (`ranking.panel_scoring.artifact_path` + legacy `panel_ltr.*`) | daily/intraday inference context hydration — **this is the surface that records `{bundle_id, manifest_digest, …}` into run bundles at P2** |
| `src/renquant_orchestrator/artifact_resolver.py:89,185` | ref + sha256 verify | shared loader |
| `src/renquant_orchestrator/cli.py:190` | `--artifacts-prod-dir` default `<repo-root>/backtesting/renquant_104/artifacts/prod` | CLI surface |
| `scripts/d6_freeze_record.py:89-91` | read-only hash of both members for freeze evidence | manual evidence tool |
| `src/renquant_orchestrator/agent_workflows.py:84` | (policy) regex guard REFUSING agent workflows that write `artifacts/prod/` | agent-control enforcement — keep; it becomes redundant-in-depth once P3 removes the write path |
| `ops/run_surface_drift_check.py` | launchd surface only (plist/program_args sha256) — does NOT hash pair content | daily drift scan; NOTE: pair-content drift is invisible to it today (the bundle store's operation log closes this) |

`renquant-orchestrator/scripts/prestamp_legacy_fingerprints.py:129-140`
(`--apply`) writes fingerprints into data-lake / WF-fold artifacts
(`data/panel-ltr-prod-*.json`, `artifacts/sim/walkforward_calibrators/...`)
— NOT the prod pair; census'd n/a-adjacent. `retrain_alpha158_linear.py:161-165`
writes the LINEAR variant family (`artifacts/panel-ltr.alpha158_linear.json`),
distinct filenames, not the pair.

### 3.5 renquant-backtesting (sim/WF/reporting readers) — **keep flat VIEW**

`analysis/smoke_test_model.py`, `reporting/build_dashboard.py`,
`walk_forward/lean_guard.py`, `wf_gate/{runner.py:680, wf_config_builder.py,
wf_config_parity.py, wf_panel_args.py:115, artifact_loader.py:104}`,
`forensics/challenger.py` — read for WF gate, dashboards, forensics.

### 3.6 renquant-artifacts — contract library — **migrates to bundle API (it becomes the publisher)**

`src/renquant_artifacts/contracts.py`: `resolve_artifact_paths :124`
(resolves BOTH pair members from the config keys),
`validate_panel_artifact_contract :167`, `build_run_bundle :444`,
`_iter_artifact_refs :509`. All read/validate; no writes. The registry
(`registry/*.json`, `store/STORE-MANIFEST.json`) tracks only
shadow/experiment variants today — **the prod pair is not registered**,
confirming the RFC §5 migration gap this design closes.

### 3.7 renquant-{strategy-104, common, base-data, execution}

- strategy-104: config values only (§1 table) + `src/renquant_strategy_104/config.py`
  loader; no file I/O on the pair.
- common: key/field primitives only (`model_fingerprint.py:226,620,756`,
  `config_consistency.py`, `contracts/scorer.py:8`, `row_coverage.py`) — no pair I/O.
- base-data, execution: **zero hits** (confirmed; base-data has only the
  generic `panel_ltr.row_coverage` config-key reader).
- Variant checkouts (`renquant-pipeline-fractional`, `renquant-common-crypto`,
  `renquant-model-crypto`, `*-run` trees): pair-referencing surfaces are
  mirrors of their canonical repos — no divergent reader/writer found
  (verified by file-set comparison for the orchestrator run tree and
  src-level grep for the rest).

### 3.8 Tests — **test-only n/a** (all use tmp fixtures; spot-checked)

~49 umbrella `tests/` + `tests/acceptance/**`, ~24 pipeline, ~19
backtesting, ~7 model, 6 common, 4 artifacts, 1 strategy-104 test files
reference the pair names. Spot-checks (`tests/test_monthly_calibrator_atomic_swap.py`,
`tests/_weekly_promote_fixture.py`, `tests/test_restamp_prod_fingerprint_snapshot_backstop.py`,
`tests/test_calibrator_overwrite_guard.py`, pipeline
`test_shadow_artifact_resolution.py`, backtesting
`test_promotion_integrity_guard.py`) confirm tmp_path/monkeypatch-only;
they exercise the REAL writer modules (atomic_swap, weekly promote,
restamp) — valuable as the P2 regression suite. One anomaly:
`renquant-strategy-104/tests/test_strategy_configs.py:55` hardcodes an
absolute umbrella path in a string assertion (no file access).

### 3.9 Totals

| repo | reader surfaces | writer surfaces |
|---|---|---|
| RenQuant umbrella | ~40 code readers (kernel/preflight/sim/scripts diagnostics + experiment cohort) | ~13 (§2.6) |
| renquant-pipeline (+ fractional mirror) | ~20 (runtime loader, matcher, preflight x2 impls, regime arm) | 0 |
| renquant-orchestrator (+ run mirror) | ~10 (monitors, hydration/provenance, CLI, freeze tool) | 2 |
| renquant-backtesting | ~8 (WF gate, reporting, forensics) | 2 |
| renquant-model (+ crypto mirror) | serializer/loader primitives | 0 direct (primitives) |
| renquant-artifacts | 4 contract functions | 0 |
| renquant-strategy-104 | config only | 0 |
| renquant-common / base-data / execution | key primitives / 1 config-key / zero | 0 |
| tests (all repos) | ~110 files | 0 (tmp-only) |

## 4. WRITERS summary (the classified list)

| # | writer | context | classification |
|---|---|---|---|
| 1 | weekly WF promote chain (`weekly_wf_promote.sh` + `model_acceptance.promote` + retrain/staging + `run_wf_gate` stamp + `stamp_walkforward_fingerprints` binding stamp) | launchd weekly + 2 delegating jobs | **migrates to bundle API** (the primary publication) |
| 2 | monthly calibrator refresh (`monthly_calibrator_refresh.sh` + `monthly_calibrator_atomic_swap.py`) | launchd monthly | **migrates to bundle API** |
| 3 | restamp tools (`restamp_prod_fingerprint.py`, `stamp_panel_contract_missing_fields.py`, binding stamper) | manual repair / weekly step | **migrates to bundle API** after B3 rework (no in-place mutation) |
| 4 | `manual_promote.sh`, `model_acceptance.rollback()`, `train_104.py --promote` (RQ_ALLOW_NO_WF), documented recovery runbooks (W4-f) | manual emergency | **migrates to `bundle_breakglass`** |
| 5 | bare `daily_retrain_alpha158_fund.sh` (both runners' prod defaults: orchestrator `retrain_alpha158_fund.py:1447-1452` + umbrella twin `:71-72`) | manual (unscheduled) | **blocker B4** — remove direct-to-prod default |
| 6 | ad-hoc session edits (orphan `pre-v1-restamp`/`pre-binding-fix` copies, no tool) | incident response | **blocker B2** — `bundle_breakglass` must exist first |
| 7 | git checkout/reset/pull on the tracked pair | any live-tree git op | **blocker B1** — resolved only at P3 (untrack + views) |
| 8 | defunct-legacy cohort (§2.5) | manual, dead paths | retire/redirect at P2 |
| 9 | `kernel/artifact_snapshot.py` backup copies | ops backup | keeps flat VIEW (writes copies, never the pair) |
| 10 | `renquant-orchestrator retention_policy.py` `prune-artifacts --execute` (deletes staging/rollback side files) | manual CLI | migrates to bundle-store GC (§2.6 of the RFC) |

## 5. Migration blockers — writers that CANNOT go through the publication API yet

Per RFC §3, "a legacy WRITER discovered post-census is a migration
blocker"; these are the writers discovered AT census time that have no
publication-API path today:

- **B1 — git itself.** The pair is git-tracked in the umbrella repo; any
  `checkout/reset/pull` is an unmediated pair writer with no
  authorization record (2026-07-08 incident class; `models/` trap
  analog). No API can intercept git. Resolution is structural and lands
  at P3: untrack the flat paths, serve read-only views, bundle store
  outside the git index. Until P3, any git operation that touches
  `artifacts/prod/` is a containment event under AC3.
- **B2 — ad-hoc manual edits.** Evidence (§2.4 W4-e): three orphan side
  files from 07-14→16 with no committing tool in any checkout. The
  practice exists BECAUSE no sanctioned tool does; `bundle_breakglass`
  must ship in P0 (before the P1 seal), or the incident-response habit
  will bypass the bundle store on day one.
- **B3 — in-place stampers.** `restamp_prod_fingerprint.py`,
  `stamp_panel_contract_missing_fields.py`, `run_wf_gate.py:3315`, and
  `stamp_walkforward_fingerprints.py:142,176` all mutate artifact files
  IN PLACE. Immutable bundles make in-place mutation impossible by
  construction; each must be reworked to "read active bundle → produce
  NEW bundle via the publisher" (or, for the WF-manifest stamps, be
  confirmed to target only non-pair corpus artifacts) before P2 can
  complete.
- **B4 — bare daily-retrain default.** BOTH retrain runners default to the
  prod pair (orchestrator `retrain_alpha158_fund.py:1447-1452` with atomic
  publish at `:850-852`; umbrella twin
  `training_panel/daily_retrain_alpha158_fund.py:71-72`); a no-arg manual
  run trains straight onto the live pair with no staging or gate. The
  defaults must become staging-only (publication via API) before the flat
  paths go read-only.
- **B5 — stale Dagster surface.** `dagster_renquant/_paths.py:26-27`
  validates the LEGACY flat dir (not `prod/`) — retire or repoint; today
  it silently validates nothing.

## 6. Proposed P0-P3 phase mapping (census attached to P2)

- **P0 — build, no live change.** renquant-artifacts publisher +
  operation log + `bundle_breakglass` (closes B2's tool gap);
  renquant-pipeline `bundle_contract.validate_pair` exported from the
  `_assert_calibrator_matches_scorer` logic (`job_panel_scoring.py:2240`);
  contract fixture in renquant-common. §4 kill-injection suite green in CI.
- **P1 — seal.** Publish the CURRENT pair as bundle generation 1
  (verbatim members incl. embedded `wf_gate_metadata`, §2.7 semantics);
  ACTIVE pointer + OPERATIONS.jsonl live; readers untouched. Rollback =
  point ACTIVE back; flat files remain the served copies.
- **P2 — writer migration (THIS CENSUS is the entry gate).** Convert in
  order of incident frequency: W1 weekly promote → publisher; W2 monthly
  refresh → publisher (pair-level, killing the per-FILE swap); W3 restamp
  → breakglass-authorized republication (B3 rework); W4 manual paths →
  `bundle_breakglass` only; retire the §2.5 defunct cohort and B5; fix
  B4. Orchestrator run-bundles start recording
  `{bundle_id, manifest_digest, member digests, generation}`. Flat paths
  become regenerated-on-flip views (still writable only by the
  publisher's view refresh).
- **P3 — flat-path retirement.** Views mode 0444; untrack the pair from
  the umbrella git index (closes B1); umbrella-local publication
  attempts have no API to call (§4 write-authority test); readers that
  still consume the flat location (§3.2/§3.3 cohort, preflight) read the
  0444 views or move to bundle resolution.

## 7. Cross-check against RFC §1's four incident writers

| RFC §1 writer | found in code? | census section |
|---|---|---|
| 1. weekly WF promote (stage → verdict → per-file rollback) | YES — exact per-file rollback copies + `.previous.json` machinery confirmed | §2.1 |
| 2. monthly calibrator refresh (atomic per-FILE swap) | YES — `os.replace` single-file publish confirmed | §2.2 |
| 3. restamp tooling (binding/fingerprint edits, per file) | YES — three in-place stampers + orchestrator sidecar tooling | §2.3 |
| 4. manual ops / incident response | YES — sanctioned (`manual_promote.sh`, `rollback()`) AND unsanctioned (ad-hoc edits W4-e, git W4-g) | §2.4 |

No automated writer outside these four classes was found; the surprises
are the two UNSANCTIONED manual sub-classes (W4-e ad-hoc edits, W4-g git)
— precisely the gap §2.4's authorization model + break-glass design must
close, now backed by census evidence.
