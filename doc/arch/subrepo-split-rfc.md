# RenQuant Physical Subrepo Split RFC

Date: 2026-05-25

Status: draft for discussion, no files moved yet.

## Goal

Split RenQuant into physical Git repositories so multiple agents can work in
parallel with clear ownership, smaller checkouts, independent CI, and fewer
ways for research artifacts to contaminate production code.

This is not a cosmetic directory cleanup. It is an architecture boundary reset:
source code, base data, model training, pipeline/runtime logic, execution,
backtesting, artifacts, and orchestration must stop sharing one mutable working
tree.

## Decisions Already Fixed

1. `model-gbdt` means the current production model line: panel-LTR / GBDT
   scorer, currently XGBoost rank:pairwise plus its calibrator and acceptance
   metrics.
2. The split target is physical repositories, not only folders in this repo.
   The current `RenQuant` repo should become an umbrella/integration repo that
   pins the subrepo revisions.
3. Data and artifacts must not be copied into normal Git history. They need
   DVC, Git LFS, object storage, or manifest-only tracking.

## Repository Set

### `renquant-common`

Purpose: small shared code package.

Owns:

- Typed contracts and schemas for models, scores, decisions, trades, metrics,
  and artifact manifests.
- Calendar/session helpers, config loading primitives, path resolution, tax
  primitives, indicators, split/regime-label utilities that are genuinely
  model-agnostic.
- Test fixtures that other repos can import.

Must not own:

- A strategy-specific decision tree.
- Live broker code.
- Model training loops.
- Large data or artifacts.

Dependency rule: no internal RenQuant repo dependency.

### `renquant-base-data`

Purpose: versioned data lake and dataset manifests.

Owns:

- OHLCV cache manifests.
- Fundamental, macro, news, sentiment, insider, IV, and LEAN data manifests.
- Dataset build provenance: input fingerprints, source timestamps, schema
  versions, and validation reports.
- Optional DVC/LFS pointers for large parquet/zip/db files.

Must not own:

- Training algorithms.
- Decision-tree logic.
- Live broker credentials or live state.

Dependency rule: may use `renquant-common` validation schemas, but data files
remain data artifacts, not Python import dependencies.

### `renquant-model-gbdt`

Purpose: current production model repo.

Owns:

- Panel-LTR / GBDT training code.
- XGBoost/LightGBM/CatBoost backend wrappers.
- Feature assembly code that is strictly part of the GBDT training contract.
- Global rank/expected-return calibrators for GBDT artifacts.
- Model acceptance outputs: OOS IC, regime IC, placebo/shuffle sanity,
  calibration health, feature/config fingerprints.
- A writer for `training_runs`-equivalent model ledger rows.

Must not own:

- Live order placement.
- QP portfolio construction.
- Broker adapters.
- Strategy scheduling.
- PatchTST/HF sequence-model training.

Runtime output contract:

```text
fit(dataset_manifest, model_config) -> model_artifact, calibration_artifact, metrics_record
score(feature_frame, model_artifact) -> per_ticker_scores
validate(model_artifact, validation_manifest) -> acceptance_report
```

### `renquant-model-patchtst`

Purpose: PatchTST/PatchTXT and sequence-model research/production candidate
repo.

Owns:

- HF PatchTST wrappers, trainer code, DLinear/iTransformer/FiLM/cross-stock
  sequence baselines if retained.
- PatchTST walk-forward manifest generation.
- Declared-label and raw-expected-return sanity reports.
- PatchTST shadow artifact registry and metrics writer.

Must not own:

- Production decision tree.
- Live broker code.
- GBDT primary model training.

Promotion rule: this repo can produce candidate artifacts only. Promotion is
decided by `renquant-pipeline`/`renquant-orchestrator` acceptance gates.

### `renquant-pipeline`

Purpose: production decision engine.

Owns:

- `InferencePipeline`, `SellOnlyPipeline`, `TrainingPipeline` orchestration
  interfaces.
- Task/Job implementations for regime, buy gates, sell logic, ranking,
  rotation, QP, preflight, acceptance gates, decision trace persistence.
- Strategy-independent portfolio construction primitives.
- Runtime model-loader interfaces that consume artifacts through contracts.

Must not own:

- Model training loops.
- Broker credentials or order submission.
- Data lake files.

Dependency rule: depends on `renquant-common`; may depend on model runtime
packages only through scorer interfaces. It must not import model training
modules.

### `renquant-execution`

Purpose: live/paper execution boundary.

Owns:

- Alpaca/IBKR/Paper/readonly broker adapters.
- Live runner CLI.
- Order submission/cancel/reconcile logic.
- ntfy/macOS notification integration.
- Live-state snapshots, but not large historical DBs.

Must not own:

- Model training.
- Sim/backtest-specific engines.
- Research artifacts.

Dependency rule: depends on `renquant-common`, `renquant-pipeline`, and a
pinned artifact manifest.

### `renquant-backtesting`

Purpose: simulation and LEAN validation.

Owns:

- LEAN algorithm wrappers and data export tools.
- Sim adapters and walk-forward portfolio simulation.
- Trade forensics and decision-tree replay tools.
- Backtest-specific fixtures and reports.

Must not own:

- Live broker credentials.
- Model training implementation.
- Production artifact promotion logic, except through acceptance reports.

Dependency rule: depends on `renquant-common`, `renquant-pipeline`, and data /
artifact manifests.

### `renquant-artifacts`

Purpose: model/data/run registry, not a dumping ground.

Owns:

- Accepted production artifact manifests.
- Shadow artifact manifests.
- Model registry metadata, fingerprints, and acceptance reports.
- Training/simulation/live DB snapshots when deliberately exported.
- DVC/LFS pointers or object-store URIs for large artifacts.

Must not own:

- Source code.
- Raw data lake contents in normal Git.
- Random experiment outputs without an owner, TTL, and verdict.

Hard rule: every artifact entry has owner, strategy, model family, data
fingerprint, config fingerprint, metric summary, promotion status, and expiry
or retention class.

### `renquant-orchestrator`

Purpose: umbrella/integration repo.

Owns:

- Submodule or pinned-revision manifest for all repos.
- Dagster/launchd/cron workflow definitions.
- Daily full / shadow full / weekly train / promotion workflows.
- Cross-repo CI/CD, integration tests, release notes, and operator docs.
- Strategy-level config assembly for active production.

Must not own:

- Model internals.
- Data lake blobs.
- Broker secrets.

This can be the renamed/current `RenQuant` repo to preserve operator muscle
memory.

### Optional Later Repos

- `renquant-strategy-104`: active strategy config and strategy-specific policy
  if we decide strategy ownership is too large for `renquant-orchestrator`.
- `renquant-research`: notebooks, failed experiments, exploratory scripts.
- `renquant-docs`: public/operator docs if docs churn becomes noisy.

## Dependency DAG

```text
renquant-common
    ^
    |
    +-- renquant-base-data        (schemas/validators only)
    +-- renquant-model-gbdt       (training + scoring artifacts)
    +-- renquant-model-patchtst   (training + scoring artifacts)
    +-- renquant-pipeline
            ^
            |
            +-- renquant-execution
            +-- renquant-backtesting

renquant-artifacts is consumed by execution/backtesting/orchestrator through
manifest files, not Python imports.

renquant-orchestrator pins every repo and runs cross-repo workflows.
```

Forbidden dependencies:

- `renquant-common` importing any other RenQuant repo.
- Any model repo importing `renquant-execution`.
- `renquant-pipeline` importing training modules from model repos.
- `renquant-execution` importing research notebooks or training scripts.
- `renquant-base-data` importing model or pipeline code.

## Migration Strategy

### Phase 0: Freeze and Classify

No movement yet.

1. Snapshot current dirty state.
2. Classify dirty/untracked files as source, data, artifact, live state, cache,
   log, or obsolete experiment.
3. Add or tighten `.gitignore` before creating any new repo.
4. Create split manifests and cross-repo contracts.

Reason: the current working tree has hundreds of dirty/untracked entries and
large generated files. Moving files now would mix production source with
experiment residue.

### Phase 1: Create Physical Repos from `HEAD`

Use clean clones or `git filter-repo`/`git subtree split` from committed
history, not the dirty working tree.

Recommended local paths:

```text
/Users/renhao/git/github/renquant-common
/Users/renhao/git/github/renquant-base-data
/Users/renhao/git/github/renquant-model-gbdt
/Users/renhao/git/github/renquant-model-patchtst
/Users/renhao/git/github/renquant-pipeline
/Users/renhao/git/github/renquant-execution
/Users/renhao/git/github/renquant-backtesting
/Users/renhao/git/github/renquant-artifacts
/Users/renhao/git/github/RenQuant          # umbrella/orchestrator
```

Do not push generated repos until each has:

- README with ownership and dependency rules.
- Minimal package/test scaffold.
- CI command.
- Import-boundary test.
- No secrets.
- No accidental data/artifact blobs beyond policy.

### Phase 2: Package Contracts First

1. Publish/install `renquant-common` locally as editable package.
2. Replace cross-directory imports with package imports.
3. Add boundary tests that fail if forbidden imports reappear.
4. Keep compatibility shims in the umbrella repo while callers migrate.

### Phase 3: Split Models

1. Move GBDT training/scoring into `renquant-model-gbdt`.
2. Move PatchTST/HF/DLinear sequence work into `renquant-model-patchtst`.
3. Both repos write the same model-ledger record format.
4. Both repos emit artifact manifests consumed by `renquant-artifacts`.
5. Neither repo can write live orders or mutate live state.

### Phase 4: Split Runtime

1. Move task/job/pipeline/QP/preflight into `renquant-pipeline`.
2. Move broker/live runner/notifications into `renquant-execution`.
3. Move sim/LEAN/WF evaluation into `renquant-backtesting`.
4. Umbrella pins revisions and runs integration tests.

### Phase 5: Data and Artifact Externalization

1. Convert `data/`, `backtesting/data/`, and large `artifacts/` to
   DVC/LFS/object-store manifests.
2. Export only selected historical DB snapshots to `renquant-artifacts`.
3. Keep live WAL/SHM files and local state out of all repos.

## CI Gates Per Repo

Every repo needs:

- Unit tests for its own code.
- Import-boundary tests.
- Schema compatibility tests against `renquant-common`.
- Contract fixture tests with one tiny synthetic dataset/artifact.
- No large-file or secret scan.

Umbrella integration CI adds:

- Build all editable packages from pinned revisions.
- Run inference smoke with tiny fixture.
- Run model-ledger query smoke.
- Run readonly daily-full smoke without live order submission.

## Immediate Open Questions

1. GitHub naming convention: `renquant-model-gbdt` vs `RenQuant-Model-GBDT`.
   I recommend lowercase hyphenated names.
2. Submodules vs pinned manifest plus clone script. I recommend pinned manifest
   first because submodules are easy to desync during rapid agent work.
3. Whether active strategy config stays in `renquant-orchestrator` or becomes
   `renquant-strategy-104`. I recommend keeping it in orchestrator until the
   pipeline contracts stabilize.

## Acceptance Criteria For The Split

The split is not done until:

1. A fresh machine can clone the umbrella repo, run one bootstrap command, and
   get all source repos at pinned commits.
2. `daily full` uses the split packages without changing decision-tree output
   versus the pre-split baseline on a frozen fixture.
3. Training a GBDT model writes a ledger row without importing execution code.
4. PatchTST WF sanity writes comparable ledger rows without importing GBDT
   internals.
5. Execution can place/skip/cancel orders without importing any training
   module.
6. Data/artifacts are referenced by fingerprinted manifest, not by accidental
   relative paths into a developer's local tree.
