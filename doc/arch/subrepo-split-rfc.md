# RenQuant Physical Subrepo Split RFC

Date: 2026-05-25 (initial draft); revised 2026-05-27 to reconcile shipped
bootstrap state with intended contracts.

Status: **Backfill Plan P0–P3 COMPLETE (2026-05-27).** The contract layer
is live and the model repos are merged. Remaining work is the functional
lift of umbrella code (decision-tree kernel, training_panel, live, sim,
scripts) into the subrepos — see §"Backfill Plan" status annotations and
the functional-lift tasks.

Completed this session:
- **P0** — `renquant-common` v0.2.0: Scorer Protocol + `load_scorer`
  registry, `RegimeLabel` enum + `validate_regime_params`, Pydantic
  schemas (ArtifactManifest/AcceptanceReport/DecisionTraceRow/…),
  `renquant_common.stats` (DSR/PBO/Wilcoxon/HAC/regime_stratified),
  canonical `PurgedKFold`. API-snapshot test.
- **P1** — XGBoost scorer relocated to the model repo; `renquant-pipeline`
  consumes only via `load_scorer`; `xgboost_scorer.py` leak deleted;
  `RegimeLabel` adopted in `context.py`.
- **P2** — AST import-boundary tests in pipeline + backtesting; raw-regime
  string grep test in common (caught + fixed 2 real drift instances in
  strategy-104 and artifacts); `validate_regime_params` wired into
  strategy-104; artifacts macro defaults keyed off `RegimeLabel`.
- **P3** — `renquant-model` created (v0.1.0), merging gbdt + patchtst with
  git-history preserved via filter-repo subtree merge; both old repos
  archived (`MIGRATED_TO_renquant-model.md`, `archived_subrepos` in lock);
  pipeline repointed; scorer entry point now owned by `renquant-model`.
  Namespace note: the two family packages stay top-level
  (`renquant_model_gbdt`, `renquant_model_patchtst`) co-located rather
  than deep-renamed to `renquant_model.{gbdt,patchtst}`, to preserve the
  working entry points / consumer wiring from P1. Consolidation goal met;
  nested namespace is a deferred non-breaking refactor.

Cross-repo refactors are now unblocked (contracts are stable).

## Goal

Split RenQuant into physical Git repositories so multiple agents can work in
parallel with clear ownership, smaller checkouts, independent CI, and fewer
ways for research artifacts to contaminate production code.

This is not a cosmetic directory cleanup. It is an architecture boundary reset:
source code, base data, model training, pipeline/runtime logic, execution,
backtesting, artifacts, and orchestration must stop sharing one mutable working
tree.

Equally important: the split is a **multi-agent collaboration substrate**.
Every cross-repo interface — Scorer Protocol, regime taxonomy, decision-trace
schema, artifact manifest, acceptance-report shape — must be defined as a
typed contract in `renquant-common` before consumers depend on it. Without
typed contracts, parallel agents desync through duck-typed dicts; the split
then makes integration *worse*, not better. This is what the bootstrap got
wrong and what this revision fixes.

## Decisions Already Fixed

1. The "GBDT line" means the current production model line: panel-LTR / GBDT
   scorer, currently XGBoost rank:pairwise plus its calibrator and acceptance
   metrics. PatchTST/PatchTXT and other sequence baselines are the "sequence
   line".
2. The split target is physical repositories, not only folders in this repo.
   The current `RenQuant` repo should become an umbrella/integration repo that
   pins the subrepo revisions. It must never be deleted, emptied, or rewritten
   as part of the split; it remains the local orchestrator, integration
   harness, and rollback source.
3. Data and artifacts must not be copied into normal Git history. They need
   DVC, Git LFS, object storage, or manifest-only tracking.
4. `renquant-strategy-104` is part of the first-wave split. It owns active
   strategy policy and config, not model training or broker execution.
5. `renquant-orchestrator` owns local assembly: pin repo revisions,
   fetch/validate data and artifacts, build the deterministic local bundle,
   export LEAN data, and place the assembled strategy into the LEAN runtime.
6. Every repo follows the pipeline pattern. `renquant-common` owns generic
   `Task` / `Job` / `Pipeline` / `run_parallel` primitives. Model training,
   inference, execution, and backtesting repos compose those primitives rather
   than inventing repo-local orchestration.
7. **(2026-05-27)** All model lines collapse into a single `renquant-model`
   repository with per-family subdirectories (`gbdt/`, `patchtst/`, shared
   `common/`). The two-repo bootstrap (`renquant-model-gbdt` +
   `renquant-model-patchtst`) is drift to be reconciled in §"Backfill Plan"
   P3. Rationale: shared `PurgedKFold`, shared feature-assembly utilities,
   shared training-ledger writer, single Scorer-registration entry point —
   none of which can exist without duplication while the two repos are
   separate.
8. **(2026-05-27)** Promotion-gating responsibilities are split by layer:
   pure statistical primitives (DSR, PBO, Wilcoxon, HAC, regime-stratified
   scorer) live in `renquant-common` as `renquant_common.stats`; experiment
   aggregation and acceptance-report production live in
   `renquant-backtesting`; the promotion-decision workflow trigger lives in
   `renquant-orchestrator`. This mirrors the Scorer-Protocol layering: pure
   contract in common, implementation in domain repo, workflow in
   orchestrator.
9. **(2026-05-27)** Contract-first discipline is non-negotiable going
   forward. Every cross-repo interface must be a typed contract in
   `renquant-common` *before* any consumer ships code that relies on it. The
   bootstrap shipped consumers (XGBoost-aware pipeline, raw-string regime
   labels, per-repo `PurgedKFold`) without their contracts; §"Backfill Plan"
   reverses that ordering.
10. **(2026-05-27)** **Branch model is part of the cross-repo contract.**
    In *every* subrepo — code, data, artifacts — the `main` branch
    represents the current promoted/production state. Non-`main` branches
    are R&D, candidate, shadow, archive, or storage. Promotion is *a merge
    to `main`*, not a metadata field on an artifact. This collapses
    "what's in production right now" from a database query to a Git
    operation (`git ls-tree main`). See §"Branch Model" for the full
    convention and per-repo applications.

## Bootstrap Drift Audit (as of 2026-05-27)

The Phase 1 bootstrap shipped 10 sibling repos but skipped the contract-first
discipline that the original RFC implied in §"Repository Set". A review on
2026-05-27 surfaced five concrete drift items. Each is annotated with the
backfill priority used in §"Backfill Plan".

1. **`XGBoostPanelScorer` lives in `renquant-pipeline`.**
   `renquant-pipeline/src/renquant_pipeline/xgboost_scorer.py` is a concrete
   class that imports `xgboost` lazily. Pipeline now *knows* a specific model
   backend. When `renquant-model` PatchTST asks for promotion, pipeline has
   to change. Direct violation of the RFC rule "renquant-pipeline must not
   import model training modules" — the spirit of which includes "must not
   know which backend produced an artifact". **Severity: P1.**

2. **Regime taxonomy is a raw string repeated in three repos.**
   The literals `"BULL_CALM" / "BULL_VOLATILE" / "BULL_STRONG" / "BEAR" /
   "CHOPPY"` are hardcoded in `renquant-pipeline/context.py`,
   `renquant-artifacts/contracts.py` (as policy defaults), and
   `renquant-strategy-104/config.py` (as required keys). Adding or renaming
   a regime requires synchronized edits across three repos with no schema to
   enforce the set is closed. This is the §5.13.5 anti-pattern ("one
   business decision = one function") shipped across repo boundaries.
   **Severity: P0.** PRIME DIRECTIVE makes detector quality P0; the
   taxonomy backing it is therefore also P0.

3. **`PurgedKFold` lives only in `renquant-model-gbdt`.**
   `renquant-model-gbdt/src/renquant_model_gbdt/purged_cv.py` is the only
   canonical copy. `renquant-model-patchtst` does not import it. When
   PatchTST training adds embargo logic, the most likely outcome is a second
   implementation with subtly different semantics — exactly the
   2026-05-20 walk-forward-leakage incident path that §5.13.16 was written
   to prevent. **Severity: P0.**

4. **`renquant-common` is anemic.**
   It exports only `Task / Job / Pipeline / run_parallel / PipelineResult /
   PipelineStepRecord / ParallelTimeoutError`. The original RFC stated common
   owns "typed contracts and schemas for models, scores, decisions, trades,
   metrics, and artifact manifests." None of those exist yet. Cross-repo
   data flows through untyped `dict[str, Any]`. **Severity: P0.**

5. **Two model repositories.**
   `renquant-model-gbdt` and `renquant-model-patchtst` were bootstrapped
   independently. Item 3 (duplicate `PurgedKFold`) is one symptom; another
   is that each repo has its own feature-assembly entry points with no
   shared scaffold. Decision (fixed item 7): merge into `renquant-model`
   with subdirectories. **Severity: P3** (reconciliation, blocked by P0/P1
   so the contract surface is stable before the merge).

Additional smaller items folded into the backfill plan rather than called
out separately:

- `renquant-artifacts/contracts.py` mixes artifact-completeness validation
  (in scope) with regime-default policy (`SENTIMENT_DEFAULT_REGIME_POLICY`,
  out of scope — belongs to common/schema or strategy-104/policy).
- `renquant-orchestrator` currently owns no enforced cross-repo
  pip-resolution test, so a breaking change to common would only surface at
  daily-run time.
- Scripts in the umbrella `scripts/` (~200+ entries) have no documented
  target subrepo; they have been treated as orchestrator-local utilities by
  default. §"Scripts Distribution" makes ownership explicit.

## Branch Model

The branch is the **promotion mechanism**. Every subrepo — code, data,
artifacts — follows the same convention:

| Branch | Role | Who writes here |
|---|---|---|
| `main` | Current promoted / production state | Merge from PR only |
| `feature/<topic>` | Active R&D | Agents working on candidates |
| `candidate/<run-id>` | Concrete experiment output awaiting acceptance | Training / sim drivers |
| `shadow` | Long-lived; runs alongside production but does not promote | Shadow training cron |
| `archive/<YYYY-Q#>` | Cold storage of retired versions | Quarterly archive job; immutable |
| `storage/<kind>` | Append-only bulk storage that is not "promoted" but worth keeping | Data backfills, audit dumps |
| `hotfix/<incident>` | Emergency carve-out for P0 production bugs | On-call human only |

**Implication: `main = production` is the single source of truth.** No
extra `promotion_status` field, no parallel registry table, no
"is_promoted" boolean. Asking "what's in production?" is
`git checkout main && ls`.

### Per-repo applications

| Repo | `main` contains | Branches contain |
|---|---|---|
| `renquant-common` | Shippable code, semver-tagged releases | Feature work; release candidates |
| `renquant-base-data` | Validated data manifests for the production pipeline | `wip/<source>` in-flight refreshes; `archive/<date>` quarterly snapshots |
| `renquant-model` | Code for every promoted model family | Feature branches per training-run candidate |
| `renquant-artifacts` | Production artifact manifests + their stored `AcceptanceReport` | `shadow/<run-id>` shadow manifests; `candidate/<run-id>` R&D; `archive/<date>` retired |
| `renquant-pipeline` | Production decision-tree code | Feature work |
| `renquant-execution` | Production broker / live runner code | Feature work; emergency hotfix |
| `renquant-backtesting` | Production sim + promotion-gating aggregator code | Sim experiments under feature branches |
| `renquant-strategy-104` | Current live `strategy_config.json` + golden config | Proposed config changes as PRs with attached `AcceptanceReport` |
| `renquant-orchestrator` | Production daily-full / cron / promotion workflow | Workflow refactors as feature branches |

### Promotion workflow

RenQuant is a solo-operator codebase. **Promotion uses verbal approval
plus a local `--no-ff` merge, not a GitHub PR.** The PR layer would add
ceremony without adding a second reviewer; the user IS the reviewer, and
the chat IS the review.

For any artifact, config, or code change, promotion is:

```text
1. R&D produces candidate on a feature or candidate branch (pushed to
   origin so it survives session/laptop loss).
   - model: training writes ArtifactManifest to candidate/<run-id>
   - config: agent edits strategy_config.golden.json on feature/...
   - code: standard feature branch

2. renquant-backtesting writes AcceptanceReport for the candidate
   (when applicable — config / artifact promotions require Tier 3 per
   §5.13.4a; pure code refactors require tests green).

3. Agent reports completion: branch pushed, AcceptanceReport summary,
   "ready to merge?".

4. User gives verbal approval ("merge it" / "go" / "ok").

5. Agent runs the local promotion sequence:
     git checkout main
     git merge --no-ff <feature-branch> -m "Merge ... promoted per
       verbal approval per feedback_no_pr_verbal_merge"
     git push origin main
     git tag -a vX.Y.Z -m "..."       # when bumping a semver release
     git push origin vX.Y.Z
     git branch -d <feature-branch>
     git push origin --delete <feature-branch>

6. Orchestrator's next daily-full run pins the new main HEAD
   automatically; or a coordinated local lock-advance commit on the
   umbrella advances subrepos.lock.json for atomic multi-repo
   promotions.
```

**Pre-merge internal review checklist** (the agent runs through this
before reporting "ready to merge?" — it replaces what a PR template
would have enforced):

- [ ] All tests green locally (`pytest -q`).
- [ ] Boundary tests green (`test_import_boundaries`, plus per-RFC
      §"Cross-Repo Contracts → Boundary test matrix" guards).
- [ ] If schema/contract changed: `tests/api_snapshot/` updated and
      classified additive vs breaking.
- [ ] If artifact/config promotion: `AcceptanceReport.tier ==
      LIVE_PROMOTABLE` evidence included in the report.
- [ ] If renquant-common version bumped: pyproject `version` field
      bumped per semver, and consumer subrepos' bounds checked.

**Hotfix carve-out** (§ below) is unchanged — emergency `hotfix/<incident>`
branches still merge directly per the same verbal-approval mechanism.

**Note on existing GitHub PRs:** PRs opened during the early bootstrap
(e.g. `renquant-common#1`, closed 2026-05-27) should be closed without
GitHub-merge and merged locally so the merge commit carries the right
authorship and timestamp. Going forward, no `gh pr create` calls during
promotion.

### Lock file semantics

`subrepos.lock.json` records *which commit on main* each subrepo is
deployed from. Constraints enforced by orchestrator CI:

- Every pinned commit MUST be an ancestor of that subrepo's `main`.
  Pinning a feature-branch tip in production lock is forbidden.
- The lock advancement workflow: bot opens a daily PR to umbrella that
  advances every pin to current `main` HEAD; integration tests must pass
  before merge.
- R&D / shadow runs use a *separate* lock file (e.g.,
  `experiment-lock.json` or per-experiment override) so the production
  lock cannot accidentally point at experimental code.

CI gate: `tools/check_lock_main_ancestry.py` in orchestrator fails if
any pin is off-main.

### Archive and storage policy

- `archive/<YYYY-Q#>` branches are created by a quarterly cron, frozen
  on creation. They preserve the state of `main` at quarter close.
  Never advanced, never deleted. Forensics on "what did production look
  like on 2026-Q1 close?" = `git checkout archive/2026-Q1`.
- `storage/<kind>` branches hold large append-only data that is too
  costly to materialize on every checkout. Examples: full historical
  decision-trace exports, daily preflight dumps. These do not flow
  into `main`; they exist to be queried.
- Branches never deleted (only archived) so audit trails survive
  rebases / squashes elsewhere.

### Hotfix carve-out

The only path that bypasses an `AcceptanceReport` requirement is a
`hotfix/<incident>` branch merged to `main` by a human on-call. Hotfix
PRs require:

- Incident link / one-paragraph description
- A follow-up issue tagged `post-mortem-required` opened in the same PR
- A test that would have caught the bug, landed in the same PR per
  §5.13.3

Hotfix carve-out is for production-down situations (broker reject loop,
live order corruption, P0 data pipeline crash) — not for "wanted to ship
a small improvement without waiting for sim".

### What this simplifies

- **`ArtifactManifest.promotion_status` is removed.** Derived from the
  branch the manifest lives on.
- **"Current production model" is unambiguous.** `git ls-tree main` in
  `renquant-artifacts` returns the answer; no parallel registry.
- **Rollback is a Git operation.** `git revert <merge-commit>` on
  `renquant-artifacts/main` reverts the promotion atomically; the
  orchestrator's next lock-advance picks it up.
- **Branch hygiene replaces metadata hygiene.** Stale candidate branches
  get pruned by a cron; stale `promotion_status="candidate"` rows
  cannot drift because the field does not exist.
- **R&D vs production lock files separate cleanly.** Production lock
  pins main; experiment lock pins anything.

### Open follow-ups

1. **Merge vs squash for promotions.** Recommend `--no-ff` merge so the
   promotion is a single visible commit with provenance to the
   candidate branch. Squash collapses provenance and is discouraged for
   promotion PRs (fine for code-only PRs to keep history clean).
2. **Candidate-branch retention.** Proposal: keep candidate branches
   for 90 days post-merge then archive (rename to
   `archive/candidate-<run-id>`). Long enough for forensics, short
   enough to keep `git branch -a` manageable.
3. **Cross-repo atomic promotions.** A config change in
   `renquant-strategy-104` that depends on a new artifact in
   `renquant-artifacts` needs both merges to land before the next
   orchestrator pin advance. Recommend the orchestrator lock PR be the
   atomic unit (it's the only repo that pins everything else).

## Repository Set

### `renquant-common`

Purpose: shared **typed-contract** package. Other repos talk to each other
through this package; if a type isn't here, the cross-repo interface doesn't
exist.

Owns (orchestration primitives — shipped):

- `Task`, `Job`, `Pipeline`, `PipelineResult`, `PipelineStepRecord`,
  `run_parallel`, `ParallelTimeoutError`. Domain-neutral, stdlib-only.

Owns (typed contracts — **to be backfilled**, see §"Cross-Repo Contracts"):

- `Scorer` Protocol plus a loader registry. Every model backend (XGBoost,
  PatchTST, future LightGBM/CatBoost, future neural) implements this
  Protocol and registers via Python entry-point. Pipeline / backtesting
  consume only the Protocol; they must not import any concrete backend.
- `RegimeLabel` enum. Closed set of regime identifiers. The single source
  of truth for `BULL_CALM / BULL_VOLATILE / BULL_STRONG / BEAR / CHOPPY`.
  Strategy configs, detector code, panel-artifact policies, and acceptance
  reports all reference the enum.
- `DecisionTraceRow` Pydantic schema. The forensics row written per ticker
  per bar — currently a sprawling dict. Pinning it as Pydantic enables
  live/sim parity tests at the schema level rather than field-by-field
  asserts.
- `ArtifactManifest` Pydantic schema. The cross-repo handshake for model
  artifacts: artifact kind, feature fingerprint, training-data
  fingerprint, config fingerprint, OOS evidence summary, calibrator
  reference, promotion status.
- `AcceptanceReport` Pydantic schema. Output of `renquant-backtesting`
  promotion gating; consumed by `renquant-orchestrator` to flip pins.
- `renquant_common.stats` module: DSR (Bailey-López de Prado 2014), PBO
  (Bailey-Borwein-López-Zhu 2015) via CSCV, Wilcoxon signed-rank median,
  HAC-corrected mean, regime-stratified mean/min. Pure functions, no I/O.
- `PurgedKFold` canonical implementation (relocated from
  `renquant-model-gbdt`). Embargo invariant pinned by a test that runs in
  common's CI.

Owns (light utilities):

- Calendar/session helpers, config loading primitives, path resolution,
  tax primitives, indicators that are genuinely model-agnostic.
- Test fixtures (fake `Scorer` implementation, fake `ArtifactManifest`,
  small synthetic panels) that other repos import for boundary tests.

Must not own:

- A strategy-specific decision tree.
- Live broker code.
- Model training loops (only the `Scorer` Protocol they target).
- Large data or artifacts.
- Concrete model-backend imports (`xgboost`, `torch`, `lightgbm`).

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

### `renquant-model`

Purpose: single repository housing every model family (GBDT, PatchTST, future
families) plus all model-shared scaffolding. Replaces the bootstrap's
`renquant-model-gbdt` + `renquant-model-patchtst` split (see §"Bootstrap
Drift Audit" item 5, reconciled in §"Backfill Plan" P3).

Layout:

```text
renquant-model/
  src/renquant_model/
    __init__.py                # registers Scorer entry points per family
    common/                    # cross-family scaffolding
      feature_assembly.py      # alpha158 + fund + PEAD + SUE builders
      training_ledger.py       # ledger-row writer used by every family
      calibrator.py            # global rank/expected-return calibrator
      acceptance.py            # OOS IC + regime IC + placebo sanity utils
    gbdt/                      # panel-LTR / GBDT family
      __init__.py              # entry point: Scorer registration
      trainer.py               # XGBoost rank:pairwise loop
      scorer.py                # XGBoostPanelScorer (impl of common.Scorer)
      pipelines.py             # Train/Score/Validate pipelines
    patchtst/                  # PatchTST / sequence family
      __init__.py              # entry point: Scorer registration
      trainer.py               # HF Trainer wrapper
      scorer.py                # PatchTSTPanelScorer (impl of common.Scorer)
      walkforward.py           # WF manifest generation
      baselines/               # DLinear / iTransformer / FiLM if retained
      pipelines.py
  pyproject.toml               # optional deps: [gbdt] xgboost; [patchtst] torch
  tests/                       # per-family tests + cross-family contract tests
```

Owns:

- Training code for every model family.
- Backend wrappers (`xgboost`, `torch`, etc.) **only** under their family
  subdirectory. Optional Python-extras keep installs slim (`pip install
  renquant-model[gbdt]` vs `[patchtst]`).
- Feature assembly that is genuinely part of a training contract (alpha158,
  fund, PEAD, SUE). Anything strategy-level lives in `renquant-strategy-104`.
- Global rank/expected-return calibrators.
- Model acceptance outputs: OOS IC, regime IC, placebo/shuffle sanity,
  calibration health, feature/config fingerprints.
- A writer for `training_runs`-equivalent model ledger rows.
- Walk-forward manifest generation (per family).
- Shadow artifact registry writers.
- All `Scorer` Protocol implementations (one per family subdirectory).

Must not own:

- Live order placement.
- QP portfolio construction.
- Broker adapters.
- Strategy scheduling.
- The decision-tree runtime (that's `renquant-pipeline`).
- Promotion gating (that's `renquant-backtesting` + `renquant-orchestrator`).

Per-family runtime output contract (identical across families):

```text
TrainingPipeline(Task/Job chain from renquant-common)
  -> ArtifactManifest, CalibratorArtifact, MetricsRecord

ScoringPipeline(Task/Job chain from renquant-common)
  -> per_ticker_scores via Scorer Protocol

ValidationPipeline(Task/Job chain from renquant-common)
  -> AcceptanceReport (consumed by renquant-backtesting promotion gating)
```

Scorer-registration mechanism: each family subpackage declares an entry point
in `pyproject.toml`:

```toml
[project.entry-points."renquant_common.scorers"]
panel_ltr_xgboost = "renquant_model.gbdt.scorer:load"
patchtst_panel    = "renquant_model.patchtst.scorer:load"
```

`renquant_common` exposes `load_scorer(artifact_manifest: ArtifactManifest)
-> Scorer` which dispatches on `artifact_manifest.kind`. Pipeline and
backtesting use only `load_scorer`; they never `import renquant_model.gbdt`
or `renquant_model.patchtst`.

Promotion rule: this repo produces candidate artifacts only. Promotion is
decided in `renquant-backtesting` (acceptance gating) and triggered in
`renquant-orchestrator` (pin flip). The model repo never mutates production
pins.

### `renquant-pipeline`

Purpose: production decision engine.

Owns:

- `InferencePipeline`, `SellOnlyPipeline`, runtime acceptance, and production
  decision orchestration built from `renquant-common` primitives.
- Task/Job implementations for regime detection, drawdown, buy gates, sell
  logic, ranking, rotation, QP, preflight, acceptance gates, decision-trace
  persistence.
- Strategy-independent portfolio construction primitives (QP solver,
  rotation, selection).
- The regime *detector* itself (`task_regime.py` and supporting code). The
  regime *taxonomy* (which labels exist) lives in `renquant-common`; the
  detector reads `RegimeLabel` from common and produces values in that
  closed set.
- `load_scorer(artifact_manifest) -> Scorer` consumption — never the
  `Scorer` implementation itself.

Must not own:

- Model training loops.
- Concrete model backends (`xgboost`, `torch`, `lightgbm` — these were
  smuggled in via `xgboost_scorer.py`; see §"Bootstrap Drift Audit" item 1
  and §"Backfill Plan" P1).
- Broker credentials or order submission.
- Data lake files.
- Promotion gating logic.

Dependency rule: depends on `renquant-common`; consumes model artifacts
**only** through `renquant_common.load_scorer` and the `Scorer` Protocol. It
must not `import renquant_model` (any subpackage) and must not import a
model-backend library directly. An import-boundary test in this repo's CI
enforces this.

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

Purpose: simulation, LEAN validation, and **acceptance-report production for
promotion gating**.

Owns:

- LEAN algorithm wrappers and data export tools.
- Sim adapters and walk-forward portfolio simulation.
- Trade forensics and decision-tree replay tools.
- Backtest-specific fixtures and reports.
- **Promotion-gating aggregator**: reads sim outputs (per-seed,
  per-window, per-regime), invokes the pure stats primitives from
  `renquant_common.stats` (DSR, PBO via CSCV, Wilcoxon, HAC,
  regime-stratified scorer), classifies each candidate as
  Tier 1 (REJECT) / Tier 2 (SCREEN) / Tier 3 (LIVE-PROMOTABLE) per
  `doc/research/promotion-methodology.md`, and writes an
  `AcceptanceReport` (schema in common) to `renquant-artifacts`.
- The previously umbrella-scoped scripts that implement this — including
  `analyze_experiments.py`, `dsr_pbo_truly_oos.py`,
  `eval_regime_stratified.py`, `eval_paired_returns.py`,
  `analyze_regime_stratified.py`, `compare_sims.py`. (See §"Scripts
  Distribution".)

Must not own:

- Live broker credentials.
- Model training implementation (consumes `Scorer` Protocol; never
  imports `renquant_model`).
- The promotion **decision** itself — that is `renquant-orchestrator`
  reading the `AcceptanceReport` and flipping pins. Backtesting writes the
  report; orchestrator acts on it.
- The statistical primitives themselves (those are pure functions in
  `renquant_common.stats`).

Dependency rule: depends on `renquant-common`, `renquant-pipeline`, and
data/artifact manifests. Does not import `renquant-model` subpackages.

### `renquant-artifacts`

Purpose: model/data/run **registry** plus artifact-level **integrity
validators**. Not a schema repo, not a policy repo.

Owns:

- Accepted production artifact manifests (instances of common's
  `ArtifactManifest`).
- Shadow artifact manifests.
- Model registry metadata, fingerprints, and stored acceptance reports
  (instances of common's `AcceptanceReport`).
- Training/simulation/live DB snapshots when deliberately exported.
- DVC/LFS pointers or object-store URIs for large artifacts.
- Artifact-level integrity validators: file hashes, fingerprint
  recomputation, lookahead-days consistency, OOS-evidence completeness.
  These are functions over `ArtifactManifest` *instances*; the schema
  itself lives in common.

Must not own:

- Source code beyond registry / validation helpers.
- Raw data lake contents in normal Git.
- Random experiment outputs without an owner, TTL, and verdict.
- **Schema definitions** — `ArtifactManifest`, `AcceptanceReport`,
  `RegimeLabel` all live in `renquant-common`. The current
  `renquant_artifacts.contracts` module has bled schema work into the
  artifacts repo (e.g., `SENTIMENT_DEFAULT_REGIME_POLICY` hardcoded
  regime keys); §"Backfill Plan" P0 moves these to common.
- **Policy defaults** — regime gating defaults, sentiment policy maps, etc.
  belong to `renquant-strategy-104` (strategy policy) or
  `renquant-common` (schema validation), not here.

Hard rule: every artifact entry has owner, strategy, model family, data
fingerprint, config fingerprint, metric summary, promotion status, and expiry
or retention class — all as fields of common's `ArtifactManifest`.

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

This is the current `RenQuant` repo. It stays in place permanently during and
after the split; subrepo extraction must never delete, empty, or rewrite the
working tree at `/Users/renhao/git/github/RenQuant`.

### `renquant-strategy-104`

Purpose: active strategy policy repo.

Owns:

- `strategy_config.json`, golden/shadow variants, and strategy-level policy
  declarations.
- Watchlist, universe policy, sector map, regime params, gates, thresholds,
  and accepted production/shadow artifact pins.
- Strategy 104 runbook and decision-tree policy docs.
- Tiny fixtures for config validation.

Must not own:

- GBDT or PatchTST training implementation.
- Broker execution code.
- QP solver implementation.
- Large data or model artifacts.

Dependency rule: depends on `renquant-common` schemas only. It is consumed by
`renquant-orchestrator`, `renquant-pipeline`, `renquant-execution`, and
`renquant-backtesting`.

### Optional Later Repos

- `renquant-research`: notebooks, failed experiments, exploratory scripts.
- `renquant-docs`: public/operator docs if docs churn becomes noisy.

## Cross-Repo Contracts

These types live in `renquant-common`. Every cross-repo data flow must be
typed through one of them. If a new flow doesn't fit an existing type, add
the type here first, ship it in common, **then** consume it. The Python
signatures below are illustrative; the canonical source is the actual
package code.

### Scorer Protocol

```python
# renquant_common/contracts/scorer.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class Scorer(Protocol):
    """Stateless inference adapter for a trained model artifact."""

    feature_cols: list[str]

    def feature_fingerprint(self) -> str:
        """Stable hash of (feature_cols, transform_version)."""

    def predict_rows(
        self, rows: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Map ticker -> raw score. Order of rows is not significant."""

    # Optional. Returns None when the backend does not produce variance.
    def predict_variance(
        self, rows: dict[str, dict[str, float]]
    ) -> dict[str, float] | None: ...
```

Loader contract:

```python
# renquant_common/contracts/scorer.py
def load_scorer(manifest: "ArtifactManifest") -> Scorer:
    """Dispatch on manifest.kind via the entry-point registry.

    Raises if the kind is unknown or the backend package is not installed.
    """
```

Registration is via `pyproject.toml` entry-points (see `renquant-model`
section). Pipeline and backtesting must use `load_scorer` only — never an
`import renquant_model.gbdt` or `import xgboost`.

Boundary-test invariant (enforced in pipeline and backtesting CI):
`importlib.metadata` reports zero references to `renquant_model.*` or
known backend libraries (`xgboost`, `lightgbm`, `catboost`, `torch`,
`transformers`) from those repos' source trees.

### RegimeLabel enum

```python
# renquant_common/contracts/regime.py
from enum import Enum

class RegimeLabel(str, Enum):
    BULL_CALM      = "BULL_CALM"
    BULL_VOLATILE  = "BULL_VOLATILE"
    BULL_STRONG    = "BULL_STRONG"
    BEAR           = "BEAR"
    CHOPPY         = "CHOPPY"

    @classmethod
    def all(cls) -> tuple["RegimeLabel", ...]:
        return tuple(cls)
```

Closed-set guarantees enforced by:

- `renquant-strategy-104` config validator: `regime_params` must have every
  `RegimeLabel` member as a key and contain no extras.
- `renquant-pipeline` regime detector: return type annotated `RegimeLabel`;
  CI lints any raw `str` return.
- `renquant-artifacts` registry: any per-regime field on `ArtifactManifest`
  is `dict[RegimeLabel, ...]`, not `dict[str, ...]`.
- Import-boundary test in common's CI: grep across all subrepo sources for
  raw `"BULL_*"` / `"BEAR"` / `"CHOPPY"` string literals outside the enum
  definition itself.

Adding or renaming a regime is a **breaking change** to common (semver
major). The detector, strategy config, and every consumer must coordinate
through a deprecation window — see §"Schema Versioning".

### DecisionTraceRow

The per-ticker per-bar forensics row. Currently a dict in
`renquant-pipeline/decision_trace.py`; pinning as Pydantic makes live/sim
parity testing a schema-level assertion.

```python
# renquant_common/contracts/decision_trace.py
from pydantic import BaseModel
from datetime import datetime
from .regime import RegimeLabel

class DecisionTraceRow(BaseModel):
    model_config = {"frozen": True}

    run_id: str
    bar_ts: datetime
    ticker: str
    regime: RegimeLabel
    raw_score: float | None
    calibrated_score: float | None
    decision: str  # buy | sell | hold | reject | skip — separate Enum
    gate_history: list[str]
    artifact_fingerprint: str
    # ... full field list ratified at backfill time
```

### ArtifactManifest

Cross-repo handshake for model artifacts. Every promote/load operation
talks Manifest-in, Manifest-out.

```python
class ArtifactManifest(BaseModel):
    model_config = {"frozen": True}

    kind: str                      # e.g. "panel_ltr_xgboost", "patchtst_panel"
    family: str                    # "gbdt" | "patchtst" | ...
    artifact_uri: str              # file://, s3://, dvc://
    feature_fingerprint: str
    config_fingerprint: str
    training_data_fingerprint: str
    trained_at: datetime
    lookahead_days: int
    oos_evidence: OOSEvidence      # mean_ic, std_ic, per_fold_ic, cv_method, embargo_days
    calibrator_uri: str | None
    owner_repo: str                # which model-family subdir produced this
```

**Note: no `promotion_status` field.** Promotion is derived from which
branch of `renquant-artifacts` the manifest lives on per §"Branch Model"
(main = production, `shadow` = shadow, `candidate/<run-id>` = R&D,
`archive/<date>` = retired). Storing the status as a manifest field would
recreate the drift this RFC is trying to remove.

### AcceptanceReport

Output of `renquant-backtesting` promotion gating. Consumed by
`renquant-orchestrator` to flip artifact pins.

```python
class AcceptanceReport(BaseModel):
    model_config = {"frozen": True}

    candidate: ArtifactManifest
    baseline: ArtifactManifest
    n_seeds: int
    n_windows: int
    per_regime: dict[RegimeLabel, RegimeMetric]
    overall: PooledMetric        # mean, std, t, HAC-corrected
    dsr: float                   # Deflated Sharpe (Bailey-LdP 2014)
    pbo: float                   # CSCV (Bailey-Borwein-LdP-Zhu 2015)
    wilcoxon_p: float
    tier: Tier                   # REJECT | SCREEN | LIVE_PROMOTABLE
    rationale: str
```

### Stats primitives (`renquant_common.stats`)

Pure functions, no I/O, no model assumptions:

- `deflated_sharpe(returns, n_trials) -> float`
- `pbo_cscv(per_split_metrics) -> float`
- `wilcoxon_signed_rank(paired) -> WilcoxonResult`
- `hac_mean(series, lag) -> HACResult`
- `regime_stratified(per_regime_metric, weight: str = "min") -> float`

Co-located with the schemas because the schemas reference their result
shapes.

### PurgedKFold (canonical)

Relocated from `renquant-model-gbdt` to `renquant-common`. Same API. CI
test in common pins the embargo invariant `max(train_date) +
label_lookahead_days < min(val_date)` (per §5.13.16). Every model family
imports this; no parallel implementations.

### Boundary test matrix

Every consumer repo's CI runs the boundary checks relevant to it:

| Boundary | Owner | Mechanism |
|---|---|---|
| No raw `"BULL_*"` etc. outside common | common, lint cron | grep across all repos |
| Pipeline must not import model backends | pipeline | AST scan of imports |
| Backtesting must not import model backends | backtesting | AST scan of imports |
| Common public API snapshot stable | common | golden-file diff vs prior tag |
| All scorers registered via entry-point | common | `importlib.metadata` test |
| `RegimeLabel` set closed | common, strategy-104 | schema validator |

## Dependency DAG

```text
renquant-common
    ^
    |
    +-- renquant-base-data        (schemas/validators only)
    +-- renquant-artifacts        (manifest instances + integrity validators)
    +-- renquant-model            (gbdt/, patchtst/, common/; registers Scorers)
    +-- renquant-strategy-104     (policy/config)
    +-- renquant-pipeline         (inference/runtime; load_scorer only)
            ^
            |
            +-- renquant-execution
            +-- renquant-backtesting       (sim + promotion-gating aggregator)

renquant-model is consumed *only* via the Scorer Protocol exposed through
renquant_common.load_scorer. Pipeline and backtesting never import the
renquant_model package directly.

renquant-artifacts is consumed by execution/backtesting/orchestrator through
manifest files, not Python imports (other than ArtifactManifest validators).

renquant-orchestrator pins every repo, runs cross-repo workflows, and is
the only repo allowed to act on AcceptanceReports (flip pins, schedule
promotions).
```

Forbidden dependencies:

- `renquant-common` importing any other RenQuant repo.
- `renquant-model` (any subdir) importing `renquant-execution`,
  `renquant-pipeline`, or `renquant-backtesting`.
- `renquant-pipeline` importing `renquant_model.*` or any model backend
  library (`xgboost`, `torch`, `lightgbm`, `catboost`, `transformers`).
- `renquant-backtesting` importing `renquant_model.*` or any model backend
  library.
- `renquant-execution` importing research notebooks, training scripts, or
  any model backend.
- `renquant-base-data` importing model or pipeline code.
- Any repo importing raw regime string literals — must go through
  `RegimeLabel`.

Each forbidden edge has a CI boundary test in the importing repo (see
§"Cross-Repo Contracts → Boundary test matrix").

## Inter-Repo Communication Mechanism

This is the wire-level "how" — what command runs, what import resolves, what
file the consumer reads. Three things this is *not*:

- **Not Git submodules.** Submodules embed a child SHA in each parent
  commit; in a multi-agent workflow this becomes a constant source of
  SHA-pin churn and merge conflicts. Already rejected during Phase 1.
- **Not bash scripts that `git pull` other repos at runtime.** Git only
  appears during sync/assemble, never inside a hot path.
- **Not a monorepo simulator with symlinks-only.** Symlinks appear only
  in the assembly bundle as a convenience for `PYTHONPATH` setup; they
  are not the dependency mechanism.

The actual mechanism has four layers.

### Layer 1: Disk layout — independent sibling repositories

Every subrepo is a standalone `git clone` at a sibling path. They do not
know about each other at the Git level.

```text
/Users/renhao/git/github/
  RenQuant/                  # umbrella — owns subrepos.lock.json + scripts
  renquant-common/           # standalone git repo
  renquant-base-data/        # standalone git repo
  renquant-pipeline/         # ...
  renquant-model/            # (post-P3 merge; today gbdt + patchtst)
  renquant-execution/
  renquant-backtesting/
  renquant-artifacts/
  renquant-strategy-104/
  renquant-orchestrator/
```

Each repo has independent history, independent CI, independent push
permissions, independent branches. The umbrella never *contains* the
subrepos; it points at them via `local_path` entries in the lock file.

### Layer 2: Code-to-code — standard Python imports

Every subrepo is a normal Python package (`pyproject.toml` + `src/<pkg>/`
layout). Cross-repo function calls are ordinary imports:

```python
# In renquant-pipeline source:
from renquant_common import Job, Pipeline, Task, load_scorer
from renquant_artifacts import validate_artifact_manifest
```

Two complementary mechanisms make these imports resolve:

**(a) `pyproject.toml` dependencies** — the production / CI path.

```toml
# renquant-pipeline/pyproject.toml
dependencies = [
  "renquant-common>=0.2,<0.3",
  "renquant-base-data>=0.1.0",
  "renquant-artifacts>=0.1.0",
]
```

In development, agents run `pip install -e ../renquant-common` (editable
install) so local edits in a sibling repo are picked up immediately
without reinstall.

**(b) `pyproject.toml` pytest `pythonpath`** — the in-place dev path.

```toml
[tool.pytest.ini_options]
pythonpath = [
  "src",
  "../renquant-common/src",
  "../renquant-base-data/src",
  "../renquant-artifacts/src",
]
```

This makes `pytest` find sibling-repo code without any pip install at
all — useful for fresh checkouts and CI. Production runs go through (a);
local dev tests can use either.

**No `git pull` ever happens during code execution.** Git is only
involved at install/sync time.

### Layer 3: Code-to-data / code-to-artifacts — manifest + URI

Data and model artifacts do *not* flow through Python imports. Their
producers commit a small **manifest file** (JSON / Pydantic-serialized)
in the owner repo; the actual large bytes live wherever the manifest's
URI points (today: local disk; Phase 5: DVC / LFS / S3 / object store).

```text
renquant-artifacts/
  manifests/
    panel_ltr_xgboost_20260520.json   # ArtifactManifest instance (small)
      {
        "kind": "panel_ltr_xgboost",
        "family": "gbdt",
        "artifact_uri": "file:///.../artifacts/gbdt/booster_20260520.json",
        "feature_fingerprint": "sha256:...",
        ...
      }

renquant-base-data/
  manifests/
    spy_ohlcv_daily.json              # DataManifest (small)
      {
        "kind": "ohlcv_daily",
        "symbols": ["SPY"],
        "uri": "file:///.../data/ohlcv/spy_daily.parquet",
        "sha256": "...",
        "schema_version": 3,
        ...
      }
```

Consumer code:

```python
# In renquant-pipeline or renquant-backtesting:
manifest = ArtifactManifest.parse_file(
    "manifests/panel_ltr_xgboost_20260520.json"
)
scorer = load_scorer(manifest)   # dispatches on manifest.kind via entry-point;
                                  # the chosen loader reads manifest.artifact_uri
                                  # and materializes the actual model bytes
```

**Properties of this layer:**

- Small manifests live in Git (auditable, diff-able, branchable per
  §"Branch Model"). Large bytes live outside Git.
- Consumers never hardcode paths. Every read goes through a manifest.
- Migrating from local disk to S3 changes only the URI; consumer code
  is untouched. This is what makes Phase 5 a flag-flip rather than a
  rewrite.
- Per §"Branch Model": manifests on `main` are promoted; manifests on
  `candidate/<run-id>` are R&D; promotion is the merge.

### Layer 4: Cross-repo Git operations — lock file + assemble script

Git operations across repos are owned by Python tooling in the umbrella,
not by ad-hoc bash:

```text
RenQuant/
  subrepos.lock.json              # single source of truth: name → commit + branch + path
  Makefile                        # make subrepo-assemble / subrepo-test / subrepo-doctor
  scripts/
    subrepo_assemble.py           # reads lock, fetches/checks out each subrepo
    subrepo_doctor.py             # verifies disk state == lock state
    subrepo_smoke.py              # runs cross-repo integration smoke
    subrepo_daily_contract.py     # daily-full entry that uses the assembly
```

`subrepo_assemble.py` is the load-bearing tool. Its workflow (verified
against current source):

```python
for entry in lock["subrepos"]:
    1. git clone <remote> <local_path>   # if missing and --sync
    2. assert origin remote matches entry["remote"]
    3. if HEAD ≠ entry["commit"] and --sync:
         refuse if working tree dirty
         git fetch origin
         git checkout <branch>
         git checkout <commit>
    4. else: error out, instruct operator to align manually

# Then build an assembly bundle:
.subrepo_assembly/<UTC-timestamp>/
  repos/
    renquant-common -> /Users/.../renquant-common      (symlink)
    renquant-pipeline -> /Users/.../renquant-pipeline  (symlink)
    ...
  env                     # PYTHONPATH=...:repos/renquant-common/src:...
  manifest.json           # what was assembled, with all pinned SHAs
```

Daily-full then sources `env` and runs Python — at which point everything
is normal `import renquant_pipeline` etc.

Per §"Branch Model" and §"Schema Versioning §5", the assemble script
refuses any pin that is not an ancestor of its subrepo's `main`. R&D
runs override via a separate lock file.

### Why not submodules

| Concern | Lock file + assemble script (current) | Submodules |
|---|---|---|
| Multi-agent parallel push to a subrepo | Free — each subrepo has its own remote and branches | Painful — every push to a sub requires a corresponding parent-repo SHA bump, leading to constant lock contention |
| Dev workflow: edit common, run pipeline tests | Edit in `../renquant-common`, immediate via editable install | Requires submodule update + parent commit + ... |
| Reproducibility for daily-full | `subrepos.lock.json` is one human-readable file | Embedded SHAs in tree state — readable via `git submodule status` but harder to PR-review |
| Rollback | `git revert` on lock file = atomic multi-repo rollback | `git submodule update` dance |
| New-contributor learning curve | Standard `pip install -e` + `pytest` | Submodule command set is its own tutorial |
| CI flexibility (test against any subrepo branch) | Trivial — point lock at a feature branch SHA in an experiment-lock | Each combination needs its own parent commit |

The bootstrap chose the lock-file path explicitly for these reasons.

### Day-to-day workflows

**Agent working on `renquant-pipeline`:**

```bash
cd /Users/renhao/git/github/renquant-pipeline
# one-time: pip install -e . && pip install -e ../renquant-common ...
git checkout -b feature/new-rotation-task
# edit code
pytest                   # uses pyproject.toml pythonpath; finds siblings
git commit && git push origin feature/new-rotation-task
gh pr create             # PR against renquant-pipeline/main
# CI runs: pytest + boundary tests + schema-compatibility against
#          currently-pinned renquant-common
```

**Agent working on `renquant-common` (most disruptive — every consumer
must accept):**

```bash
cd /Users/renhao/git/github/renquant-common
git checkout -b feature/add-Scorer-protocol
# edit + add api_snapshot/ entries
pytest
git commit && git push && gh pr create
# After merge to main, tag v0.2.0
# Then open follow-up PRs on every consumer that needs the new API
# Bump consumer pyproject.toml: "renquant-common>=0.2,<0.3"
# Finally, umbrella PR to advance subrepos.lock.json
```

**Daily-full production run (orchestrator cron):**

```bash
cd /Users/renhao/git/github/RenQuant
make subrepo-doctor          # verify lock == disk; abort if drift
make subrepo-assemble        # build .subrepo_assembly/<ts>/
source .subrepo_assembly/<ts>/env
python -m renquant_orchestrator.daily_full --strategy renquant_104
# Inside: imports renquant_pipeline, which imports renquant_common,
# which loads a Scorer via load_scorer(manifest) where the manifest
# was committed to renquant-artifacts/main by a prior promotion PR.
```

**Rolling forward a new promoted artifact:**

```bash
# 1. R&D agent finishes training, writes manifest to candidate branch
cd /Users/renhao/git/github/renquant-artifacts
git checkout -b candidate/panel_ltr_xgboost_20260601
# write manifests/panel_ltr_xgboost_20260601.json
git commit && git push

# 2. renquant-backtesting writes AcceptanceReport against the candidate
# 3. PR to renquant-artifacts/main; required CI: AcceptanceReport.tier == LIVE_PROMOTABLE
# 4. Merge to main = promotion (per §"Branch Model")
# 5. Umbrella daily-PR bot advances subrepos.lock.json:
cd /Users/renhao/git/github/RenQuant
# bot opens PR updating renquant-artifacts pin to new main HEAD
# orchestrator CI verifies main-ancestry, pip-resolve, integration smoke
# merge → next daily-full uses the new artifact
```

### What this means for new agents

A new agent joining the project needs to know:

1. Subrepos are independent sibling git repos. Clone the ones you'll
   touch; the umbrella's `subrepos.lock.json` tells you where.
2. Cross-repo function calls are `from <package> import <name>`. Pip
   editable installs make this work locally; pyproject `pythonpath`
   makes tests work without pip.
3. Data and artifacts flow through manifest files (small, in Git) +
   URIs (large, outside Git). Never hardcode a path.
4. Git operations across repos are owned by `scripts/subrepo_*.py` in
   the umbrella. Don't invent ad-hoc bash for the same thing.
5. Promotion = merge to `main` per §"Branch Model". The branch is the
   status; there is no `promotion_status` field anywhere.

## Schema Versioning

The whole multi-agent collaboration model depends on consumers being able to
detect that a producer's contract has shifted under them. Three layered
mechanisms:

### 1. Semantic versioning on `renquant-common`

- **Patch**: bug fix in a primitive, no public-API change.
- **Minor**: additive only. New Protocol method with default impl, new
  enum member, new optional schema field.
- **Major**: breaking. Renamed/removed Protocol method, removed enum
  member, required schema field added, type narrowed.

Adding or renaming a `RegimeLabel` member is major (it shifts the
closed-set contract for every consumer). Adding a new `Scorer` kind is
not — kinds are discovered via entry-points, not enumerated in common.

### 2. Consumer-side bounds in `pyproject.toml`

Every subrepo declares its accepted common range. Example:

```toml
[project]
dependencies = [
  "renquant-common>=2.3,<3.0",
]
```

Same pattern for `renquant-artifacts`, `renquant-base-data` if the
consumer depends on their APIs (most don't — they consume manifests, not
Python).

### 3. Public-API snapshot test in common's CI

Common's CI ships a `tests/api_snapshot/` directory containing a frozen
JSON snapshot of every public name and signature in the package. The CI
runs `inspect`-based diff vs the snapshot; any change must come with an
explicit snapshot update in the same PR. PRs that touch the snapshot are
labeled `breaking` or `additive` so reviewers and downstream agents see
the change classification.

### 4. Cross-repo pip-resolve test in orchestrator CI

`renquant-orchestrator` runs a daily CI job that resolves the full
dependency graph across every pinned subrepo commit. Fails if any pair
is incompatible (e.g., common is at 3.0.1 but pipeline still pins
`<3.0`). This catches the case where a subrepo's `pyproject.toml`
bounds and `subrepos.lock.json` commit hash silently disagree.

### 5. Subrepo lock-file extension

`subrepos.lock.json` extends each subrepo entry with:

```json
{
  "commit": "...",
  "branch": "main",
  "must_be_main_ancestor": true,
  "declared_common_range": ">=2.3,<3.0",
  "tested_against_common": "2.5.1"
}
```

The orchestrator's lock-update tooling enforces two rules:

1. Refuses to advance a pin to a commit whose `pyproject.toml` range
   excludes the currently-pinned common version unless that PR also
   bumps common.
2. Refuses to advance a pin to a commit that is **not an ancestor of
   that subrepo's `main`**. Per §"Branch Model", production lock can
   only point at `main`-ancestry commits. Experiment runs use a
   separate `experiment-lock.json` with this rule relaxed.

### 6. Breaking-change protocol

When a major bump on common is needed:

1. PR to common with both old and new APIs side by side, marked
   deprecated; minor version bump only.
2. Each consumer PR updates to the new API, ships, gets pinned.
3. Once every subrepo is on the new API, a follow-up PR to common
   removes the deprecated path; this is the major bump.

The major bump never lands before every consumer is on the new API. This
keeps multi-agent work unblocked.

## Scripts Distribution

The umbrella `RenQuant/scripts/` currently holds ~200 entries created over
the strategy's life. Every script needs an explicit subrepo home. Treating
the umbrella as their permanent address creates a hidden tail dependency:
every subrepo agent has to know about umbrella scripts to reproduce a
behavior.

Classification rules:

| Category | Examples | Target subrepo |
|---|---|---|
| Raw data ingestion / refresh | `backfill_forward_returns.py`, `daily_news_sentiment_refresh.sh`, `build_alpha158_*`, `build_extended_fundamentals.py`, `build_options_iv_features.py`, `daily_iv_snapshot.sh` | `renquant-base-data` |
| Model training entry points | `_train_BB_*`, `_train_fwd*`, `daily_retrain_alpha158_fund.sh`, `dlinear_baseline.py`, `enable_hourly_transformer.py`, all PatchTST/HF training drivers | `renquant-model` (under the right family subdir) |
| Walk-forward / acceptance evaluation | `eval_truly_oos.py`, `eval_paired_returns.py`, `eval_*_5cut_5seed.py`, `dsr_pbo_truly_oos.py`, `analyze_*`, `compare_sims.py`, `compare_panel_*`, `analyze_regime_stratified.py`, `analyze_experiments.py` | `renquant-backtesting` |
| Live runner / broker tooling | `live/`, broker-specific reconciliation, ntfy notification glue | `renquant-execution` |
| Strategy config tooling | `build_regime_overlay_configs.py`, `build_regime_reeval_configs.py`, `check_config_drift.py`, `build_sector_map.py` | `renquant-strategy-104` |
| Daily / weekly / monthly orchestration | `daily_104.sh`, `conditional_retrain_104.sh`, `event_sec_schema_change.sh`, `event_watchlist_change.sh`, `backup_state.sh`, `backup_to_github.sh`, `auto_revert_b1_regression.sh`, all launchd plists | `renquant-orchestrator` |
| Design-of-experiments machinery | `_doe_*`, `_phase1_run.sh`, BB-batch drivers | `renquant-backtesting` (DOE belongs with the sim layer); winning configs cross over to strategy-104 manually |
| Repo hygiene / audits | `audit_repo_hygiene.py`, `audit_oos_ic_drift.py`, `audit_regime_detector.py`, `audit_transformer_vs_lgbm_4way.py` | Split: detector audit → `renquant-pipeline`; OOS/training audits → `renquant-backtesting`; repo hygiene → `renquant-orchestrator` |
| Diagnostics / one-shots | `diagnose_calibrator_saturation.py`, `diagnose_funnel.sh`, `diagnose_regime_classifier.py` | Diagnoses move with the subsystem they diagnose (calibrator → model, regime → pipeline, funnel → backtesting) |
| Ad-hoc / exploratory | `bench_python_vs_rust.py`, `_meta_label_*`, anything that was a one-off | `renquant-research` (when that optional repo lands); until then, archive in umbrella `archive/` |

Migration approach: scripts move as part of each subrepo's first
post-backfill PR. A script that has not been touched in 90 days and has no
documented owner is a §5.7 "failed experiment" — move to
`doc/research/failed-experiments-log.md` with a reproduction recipe or
archive it. The umbrella `scripts/` should approach empty over the next two
quarters.

Until scripts move, each entry still in umbrella gets a one-line owner
comment at the top:

```bash
#!/usr/bin/env bash
# OWNER: renquant-backtesting (slated to move 2026-Q3; see RFC §"Scripts Distribution")
```

This prevents orphaned scripts from accumulating during the transition.

## Migration Strategy

Status notes added 2026-05-27 reflect what the bootstrap already shipped.

### Phase 0: Freeze and Classify — DONE

Snapshot, `.gitignore` tightening, and initial classification completed
during bootstrap. The umbrella working tree still has hundreds of
dirty/untracked entries (this is partly chronic; `git status` short form
should be the working baseline going forward, not a clean tree).

### Phase 1: Create Physical Repos from `HEAD` — DONE

All 10 sibling repos exist and are pinned in `subrepos.lock.json`:

```text
/Users/renhao/git/github/renquant-common
/Users/renhao/git/github/renquant-base-data
/Users/renhao/git/github/renquant-model-gbdt          # to be merged → renquant-model
/Users/renhao/git/github/renquant-model-patchtst      # to be merged → renquant-model
/Users/renhao/git/github/renquant-strategy-104
/Users/renhao/git/github/renquant-pipeline
/Users/renhao/git/github/renquant-execution
/Users/renhao/git/github/renquant-backtesting
/Users/renhao/git/github/renquant-artifacts
/Users/renhao/git/github/renquant-orchestrator
/Users/renhao/git/github/RenQuant                     # umbrella
```

The original Phase 1 acceptance gate ("Do not push until each has README,
package scaffold, CI command, import-boundary test, no secrets, no
accidental data blobs") was only partially met: READMEs and scaffolds are
present, but **import-boundary tests are missing or weak in most repos**.
The boundary gap is what allowed the drift in §"Bootstrap Drift Audit" to
ship. §"Backfill Plan" P2 closes this.

### Phase 2: Package Contracts First — PARTIAL

Done:
1. `renquant-common` published/installed; minimal scaffold ships.
2. `Task / Job / Pipeline / run_parallel / PipelineResult` extracted.
3. Cross-repo imports use package form (`from renquant_common import ...`).

Not done (this is what §"Bootstrap Drift Audit" surfaced):
4. **Typed contracts not written.** No `Scorer` Protocol, no
   `RegimeLabel` enum, no `DecisionTraceRow` / `ArtifactManifest` /
   `AcceptanceReport` schemas, no stats primitives, no canonical
   `PurgedKFold` location.
5. **Boundary tests too weak.** Import-boundary tests exist in some
   repos (`renquant-common/tests/test_import_boundaries.py`) but do not
   enforce the rules in the new §"Cross-Repo Contracts → Boundary test
   matrix".
6. **Compatibility shims** in umbrella never had a deletion plan; some
   old paths still resolve to umbrella code rather than failing fast.

§"Backfill Plan" P0 + P2 finish Phase 2 properly.

### Phase 3: Split Models — PARTIAL

Done:
1. GBDT training/scoring lives in `renquant-model-gbdt`.
2. PatchTST/HF lives in `renquant-model-patchtst`.

Wrong target architecture (per Decision item 7):
3. Two model repos, no shared scaffolding, duplicate `PurgedKFold` risk,
   no Scorer-registration entry point.

§"Backfill Plan" P3 merges the two repos into `renquant-model` after P0
contracts are in place (so the merged repo immediately registers Scorer
entry points against a stable Protocol).

### Phase 4: Split Runtime — PARTIAL

Done:
1. Pipeline / panel scoring / decision trace live in `renquant-pipeline`.
2. Broker code lives in `renquant-execution` (mostly — `live/` in the
   umbrella has not yet fully migrated).
3. Sim / LEAN wrappers live in `renquant-backtesting`.

Wrong:
4. `XGBoostPanelScorer` shipped inside `renquant-pipeline` instead of
   `renquant-model`. §"Backfill Plan" P1 reverses this.
5. Promotion gating not yet relocated to `renquant-backtesting`; still
   in umbrella `scripts/`. §"Backfill Plan" P0/P3 + §"Scripts
   Distribution" finish this.

### Phase 5: Data and Artifact Externalization — NOT STARTED

`data/`, `backtesting/data/`, and large `artifacts/` are still in
umbrella history. This phase remains future work; it is independent of
the contract-first backfill, but should not start until the contract
surface is stable (otherwise externalized manifests will have to be
rewritten when schemas land).

## Backfill Plan

The bootstrap shipped consumers before their contracts. The plan below
writes the contracts and retrofits the consumers. Sequencing is strict
because P1 needs P0 done, P2 needs P0 + P1 done, and P3 needs the entire
contract surface stable so the model-repo merge does not have to chase
moving APIs.

### Tracks at a glance

```text
[now]            [+1 day]              [+2 days]            [+1 week]
P0 contracts ──► P1 pipeline refactor ─► P2 boundary tests ─► P3 model merge
common pkg        scorer relocation      lock down drift       gbdt+patchtst→model
```

P0 is in `renquant-common`. P1 is in `renquant-pipeline` and
`renquant-model-gbdt`. P2 is in every consumer repo plus a new
orchestrator job. P3 is the most disruptive — it rewrites two repos into
one and updates `subrepos.lock.json` for every dependent.

### P0 — Backfill `renquant-common` contracts (P0, blocks everything)

Owner: single agent on `renquant-common`. PR sequence:

1. Add `renquant_common.contracts.scorer` with the `Scorer` Protocol and
   `load_scorer(manifest) -> Scorer` registry function (entry-point
   discovery via `importlib.metadata`).
2. Add `renquant_common.contracts.regime` with the `RegimeLabel` enum
   and `validate_regime_params(cfg) -> None` helper used by
   `renquant-strategy-104`.
3. Add `renquant_common.contracts.schemas` with `DecisionTraceRow`,
   `ArtifactManifest`, `OOSEvidence`, `AcceptanceReport`, `Tier`
   Pydantic models. Pydantic v2; `model_config = {"frozen": True}` for
   immutability. **No `PromotionStatus`** — derived from branch per
   §"Branch Model".
4. Add `renquant_common.stats` with `deflated_sharpe`, `pbo_cscv`,
   `wilcoxon_signed_rank`, `hac_mean`, `regime_stratified`. Pure
   functions; no I/O. Tests cover Bailey-López de Prado worked examples
   from the papers.
5. Relocate `PurgedKFold` from `renquant-model-gbdt` to
   `renquant-common.purged_cv`. Keep a deprecation shim in
   `renquant-model-gbdt` that re-exports and emits `DeprecationWarning`.
6. Add public-API snapshot test (`tests/api_snapshot/`).
7. Tag `renquant-common` v0.2.0 (minor bump from current bootstrap
   version; additive only).

Acceptance: `renquant-common` CI green; downstream pip-installs succeed;
`load_scorer` returns the bootstrap's XGBoostPanelScorer when the
fixture manifest is passed (uses a transitional shim — see P1).

### P1 — Refactor `renquant-pipeline` to use `Scorer` Protocol

Owner: single agent on `renquant-pipeline` + `renquant-model-gbdt`,
working in coordinated PRs.

1. In `renquant-model-gbdt`: move `XGBoostPanelScorer` from
   `renquant-pipeline/src/renquant_pipeline/xgboost_scorer.py` into
   `renquant_model_gbdt/scorer.py`. Implement common's `Scorer`
   Protocol. Add entry-point registration in `pyproject.toml`.
2. In `renquant-pipeline`: delete `xgboost_scorer.py`. Replace all
   `load_xgboost_panel_scorer(artifact)` call sites with
   `load_scorer(ArtifactManifest(**artifact))`.
3. In `renquant-pipeline`: also replace raw regime-string usage in
   `context.py` with `RegimeLabel` enum. The dataclass field becomes
   `regime: RegimeLabel = RegimeLabel.BULL_CALM`.
4. Bump `renquant-pipeline` to depend on `renquant-common>=0.2`.
5. Bump `renquant-model-gbdt` to depend on `renquant-common>=0.2`.
6. Coordinate the two PR merges so `subrepos.lock.json` advances both
   pins in one umbrella PR.

Acceptance: `renquant-pipeline` source contains zero references to
`xgboost`, `lightgbm`, `catboost`, `torch`, `transformers`,
`renquant_model_gbdt`, `renquant_model_patchtst`. A daily-full smoke run
still produces byte-identical decisions on the frozen fixture.

### P2 — Boundary tests + drift lockdown

Owner: agent on each consumer repo, plus a final integration PR on
`renquant-orchestrator`.

1. In `renquant-pipeline` and `renquant-backtesting`: add
   `tests/test_import_boundaries.py` that AST-scans the source tree and
   fails if any of the forbidden imports appear (`xgboost`, `torch`,
   `lightgbm`, `catboost`, `transformers`, `renquant_model.*`).
2. In `renquant-common`: add `tests/test_no_raw_regime_strings.py`
   that greps across the install paths of every other subrepo and
   fails if `"BULL_*"`, `"BEAR"`, or `"CHOPPY"` appear outside enum
   definitions or test fixtures.
3. In `renquant-strategy-104`: extend `config.py` to call
   `validate_regime_params` from common.
4. In `renquant-artifacts`: refactor `contracts.py` to import the
   schemas from common instead of redefining them. Move
   `SENTIMENT_DEFAULT_REGIME_POLICY` out — keys become `RegimeLabel`
   members; the policy itself moves to `renquant-strategy-104`.
5. In `renquant-orchestrator`: add the pip-resolve daily CI job
   described in §"Schema Versioning §4".
6. In `subrepos.lock.json`: extend the schema to record
   `declared_common_range` and `tested_against_common` per subrepo.

Acceptance: all boundary tests green in CI; orchestrator pip-resolve
job green; one forbidden-import PR opened deliberately to verify the
gate fails fast.

### P3 — Merge `renquant-model-gbdt` + `renquant-model-patchtst` into `renquant-model`

Owner: single agent, executed once P0 + P1 + P2 are green so the merge
sees a stable contract surface.

1. Create new repo `/Users/renhao/git/github/renquant-model` with the
   layout in §"Repository Set → renquant-model".
2. Use `git filter-repo --to-subdirectory-filter gbdt/` on the
   `renquant-model-gbdt` history, then `--to-subdirectory-filter
   patchtst/` on the `renquant-model-patchtst` history, then merge both
   into the new repo. Preserves authorship and per-file history.
3. Hoist shared utilities into `renquant_model/common/`:
   `training_ledger`, `acceptance`, `calibrator`, `feature_assembly`.
4. Wire entry-points in `pyproject.toml` so both `panel_ltr_xgboost`
   and `patchtst_panel` register `Scorer` loaders via `load_scorer`.
5. Add optional Python-extras: `pip install renquant-model[gbdt]`,
   `pip install renquant-model[patchtst]`.
6. Update `subrepos.lock.json`: remove `renquant-model-gbdt` and
   `renquant-model-patchtst` entries; add single `renquant-model`
   entry; bump every consumer's pin in one coordinated PR.
7. Archive (do not delete) the two old repos with a top-level
   `MIGRATED_TO_renquant-model.md` file. Keep the GitHub repos for
   rollback / archaeology.

Acceptance: every existing training entry point still runs; daily-full
still produces byte-identical decisions on the frozen fixture; the two
old repos return 404 to `import` attempts (intentional, via
deprecation shim).

### Out of scope for backfill (separate work)

- Phase 5 (data/artifact externalization to DVC/LFS) — independent of
  contracts; deferred.
- Strategy 105 (30-min model line) — once `renquant-model` lands, 105
  becomes a new family subdir; not a backfill concern.
- `renquant-research` optional repo — created on demand when notebook
  cleanup reaches threshold.

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

Original (resolved or stale):

1. ~~GitHub naming convention.~~ Resolved: lowercase hyphenated; in use.
2. ~~Submodules vs pinned manifest.~~ Resolved: pinned manifest
   (`subrepos.lock.json`); in use.
3. ~~Remote creation mechanism.~~ Resolved during bootstrap.

New (2026-05-27):

4. **`DecisionTraceRow` field set ratification.** The Pydantic schema in
   §"Cross-Repo Contracts" lists a starter field set. Before P0 lands,
   need a one-pass diff of every place that currently writes a trace row
   (live runner, sim, LEAN adapter) to confirm no field is dropped on
   the schema migration.
5. **Acceptance-report storage.** Once `AcceptanceReport` is pinned in
   common and written by backtesting, where are instances stored?
   Options: (a) JSON files in `renquant-artifacts` under
   `reports/<candidate-fingerprint>.json`; (b) sqlite ledger in
   artifacts; (c) DuckDB. Lean toward (a) for now; (b)/(c) when query
   volume justifies. Decide before P0 step 7 ships.
6. **Compatibility shim deletion criterion.** Phase 2 left shims in the
   umbrella ("Keep compatibility shims while callers migrate"). What
   triggers deletion? Proposal: each shim has a TODO with target
   removal date six weeks out; an audit cron lists shims past their
   date.
7. **PatchTST entry-point registration.** PatchTST currently writes
   "shadow" artifacts only. Until P1 lands, does `load_scorer` raise
   for `kind="patchtst_panel"`, or fall back to a no-op scorer? Lean
   toward raise (fail closed); revisit if shadow-evaluation tooling
   needs it earlier.

## Acceptance Criteria For The Split

The split is not done until:

1. A fresh machine can clone the umbrella repo, run one bootstrap command,
   and get all source repos at pinned commits.
2. `daily full` uses the split packages without changing decision-tree
   output versus the pre-split baseline on a frozen fixture.
3. Training a GBDT model writes a ledger row without importing execution
   code.
4. PatchTST WF sanity writes comparable ledger rows without importing
   GBDT internals.
5. Execution can place/skip/cancel orders without importing any training
   module.
6. Data/artifacts are referenced by fingerprinted manifest, not by
   accidental relative paths into a developer's local tree.

Backfill-specific additions (must hold before declaring the contract pass
done):

7. `renquant-pipeline` source tree contains zero references to
   `xgboost`, `torch`, `lightgbm`, `catboost`, `transformers`, or
   `renquant_model.*`. Enforced by boundary test in pipeline's CI.
8. `renquant-backtesting` source tree contains zero references to the
   same backend libraries. Enforced by boundary test.
9. Every regime label flowing across a repo boundary is a `RegimeLabel`
   enum value. Enforced by grep test in `renquant-common`'s CI across
   every other subrepo's install path.
10. Every `Scorer` implementation discoverable from common's
    `load_scorer` registry; no scorer hand-instantiated in pipeline or
    backtesting. Enforced by entry-point test in common.
11. `renquant-common` semver is honored: orchestrator's daily
    pip-resolve job green; no consumer's pyproject excludes the
    currently-pinned common commit.
12. Two model repositories merged into `renquant-model`; old repos
    archived; `subrepos.lock.json` reflects the single model entry.
13. Every pin in `subrepos.lock.json` is an ancestor of its subrepo's
    `main`. Enforced by orchestrator CI (`tools/check_lock_main_ancestry.py`).
    Production reads never accidentally pin a feature branch.
