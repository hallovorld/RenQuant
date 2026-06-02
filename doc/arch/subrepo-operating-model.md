# RenQuant Subrepo Operating Model

Date: 2026-05-25

This is the shared operating model for all RenQuant physical repositories.
Every subrepo README and CLAUDE.md must link here.

The current `/Users/renhao/git/github/RenQuant` repo is never deleted, emptied,
or rewritten as part of this split. It remains the umbrella/orchestrator,
integration harness, and rollback source.

## Repository Roles

End-to-end data flow:
`renquant-base-data` (data input) + `renquant-common`/`renquant-pipeline` (shared code)
→ **`renquant-model` (MODEL FACTORY: research + training)** → `renquant-artifacts` (model output)
→ consumed by `renquant-backtesting` / `renquant-strategy-104` / `renquant-orchestrator`.

| Repo | Role | Primary Output |
|---|---|---|
| `renquant-common` | Shared contracts + pipeline primitives + **shared training/eval utils** (`Task`/`Job`/`Pipeline`, `purged_cv`, `walk_forward_splits`, `hmm_regime_labels`, `config_consistency`) | Python package imported by the factory & pipeline |
| `renquant-base-data` | Data manifests + validation + **the training-data INPUT to the factory** | Fingerprinted data manifests / dataset handles |
| `renquant-model` | **MODEL FACTORY** — ingests base-data + common code, runs research/training (GBDT + PatchTST families in `renquant_model_gbdt` / `renquant_model_patchtst`), produces models | Model artifacts published to `renquant-artifacts` |
| `renquant-artifacts` | Artifact registry + validation; **RECEIVES factory models** | Fingerprinted artifact manifests; models consumed downstream |
| `renquant-strategy-104` | Active strategy policy/config; **consumes models from `renquant-artifacts`** | Versioned strategy config bundle |
| `renquant-pipeline` | Runtime decision tree + QP/order-intent generation; shares regime/config code with trainers | Decision trace and order intents |
| `renquant-execution` | Broker execution and order audit | Broker orders, cancel/reconcile/audit records |
| `renquant-backtesting` | Sim/LEAN/WF validation and forensics; **consumes models from `renquant-artifacts`** | Backtest reports, decision-quality diagnostics |
| `renquant-orchestrator` | Daily/full orchestration across pinned subrepos (wires factory output into daily/sim/backtest) | Run bundle, decision trace, order/audit bundle |
| `RenQuant` | Permanent umbrella/integration harness + canonical data store (`data/`, gitignored) | Pinned assembly in `subrepos.lock.json` |
| `renquant-model-gbdt`, `renquant-model-patchtst` | **ARCHIVED** — merged into `renquant-model` (RFC P3); empty pre-merge shells kept for rollback. Do NOT work there | — |

## Universal Rules

1. Every workflow is a pipeline. Training, inference, execution, data refresh,
   artifact validation, and backtesting must use `renquant-common` `Task` /
   `Job` / `Pipeline` primitives or a thin adapter over them.
2. Every repo has `CLAUDE.md`, `README.md`, `renquant_repo.yml`, `Makefile`,
   `RENQUANT_REPOS.md`, and GitHub Actions CI.
3. Every repo defines inputs, outputs, owner boundaries, and forbidden imports.
4. Large data and artifacts are referenced by manifest, fingerprint, and URI.
   They are not committed to normal Git history.
5. Production promotion requires immutable fingerprints: strategy config,
   data, model artifact, calibrator, code commits, and acceptance metrics.
6. `main` is the stable interface consumed by other repos and automation.
   Work on large changes, optimizations, or experiments happens on feature
   branches. `main` stays runnable.

## Branching And SDLC

Use branches for any non-trivial change:

```bash
git checkout -b feature/<repo>-<short-topic>
```

Allowed direct-to-main changes:

- typo/doc-only fixes that do not alter operating behavior
- manifest pin updates after the target repo commit is tested

Everything else needs:

- focused tests in the owning repo
- import-boundary check
- CI passing
- a clear commit message naming the invariant or workflow changed

Every subrepo agent should read local `RENQUANT_REPOS.md` first. That file is
duplicated deliberately so an agent launched in any repo has the full map:
roles, local paths, remotes, and system flow.

The umbrella repo pins subrepo commits in `subrepos.lock.json`. Updating a
subrepo alone is not enough for production use; the umbrella lock must be
advanced after integration checks.

## Training A New XGB/GBDT Model

Owner repo: `renquant-model-gbdt`.

Required inputs:

- strategy policy from `renquant-strategy-104`
- dataset manifest from `renquant-base-data`
- training config in `renquant-model-gbdt`
- common pipeline package from `renquant-common`

Required outputs:

- model artifact manifest
- calibrator artifact manifest
- metrics record with OOS IC, regime IC, train IC, calibration health,
  placebo/shuffle sanity, config/data/code fingerprints
- training ledger row or JSONL equivalent

Artifact storage:

- The physical model file goes to object storage, DVC remote, or local
  controlled artifact store.
- `renquant-artifacts` stores the manifest and acceptance metadata, not random
  checkpoint dumps.
- The umbrella repo updates `subrepos.lock.json` only after validation.

Minimal current automation:

```bash
cd /Users/renhao/git/github/RenQuant
make subrepo-smoke
```

This smoke proves the path: strategy config loads, data manifest validates,
`renquant-orchestrator` runs GBDT training, runtime inference, execution, and
backtest shell as one daily flow, then writes an auditable run bundle.

Branching:

- Big model changes use a feature branch in `renquant-model-gbdt`.
- Strategy threshold/config changes belong in `renquant-strategy-104`, not the
  model repo.
- Runtime decision-tree changes belong in `renquant-pipeline`.

## PatchTST/PatchTXT Research And Shadow

Owner repo: `renquant-model-patchtst`.

PatchTST must report both:

- declared-label IC/sanity
- raw expected-return IC/sanity

Promotion requires the same acceptance standard as GBDT: walk-forward manifest,
regime IC, SPY comparison where applicable, calibration health, and
placebo/shuffle sanity.

Shadow artifacts go to `renquant-artifacts` as `promotion_status: shadow` or
`diagnostic`. They are not prod until accepted and pinned by the umbrella repo.

## Daily Schedule / Inference / Live Flow

Owner repo: `renquant-orchestrator`; pinned and integration-tested by
`RenQuant`.

Runtime components:

1. Read `subrepos.lock.json`.
2. Checkout or verify pinned commits for common, strategy, pipeline,
   execution, data, and artifacts.
3. Validate strategy config from `renquant-strategy-104`.
4. Resolve model artifact manifest from `renquant-artifacts`.
5. Resolve data manifests from `renquant-base-data`.
6. Assemble a deterministic local runtime bundle.
7. For LEAN, export/prepare data and copy the assembled strategy into the LEAN
   working directory.
8. Run `renquant-orchestrator` to execute train -> inference -> execution ->
   optional backtest against pinned subrepos.
9. Run `renquant-pipeline` inside that orchestration to produce decision trace
   and order intents.
10. Run `renquant-execution` only after runtime gates pass and broker mode is
   explicit.
11. Persist run metadata: code commits, data fingerprints, model fingerprints,
    strategy fingerprint, order intents, and broker results.

The model comes from `renquant-artifacts`, not from a training repo working
directory. Data comes from `renquant-base-data` manifests, not ad hoc local
paths.

Strategy repo is intentionally policy-only. It does not submit orders directly.
`renquant-orchestrator` assembles strategy + data + artifact + pipeline +
execution;
that keeps policy, alpha, portfolio construction, and broker mutation separated.

## Data Refresh And Backup

Owner repo: `renquant-base-data`.

Data update workflow:

1. Fetch/update source data into local/object storage.
2. Validate schema and freshness.
3. Compute fingerprints.
4. Write or update a dataset manifest.
5. Run `make test`.
6. Push manifest changes.
7. Umbrella lock or deployment config references the new manifest only after
   consumers pass.

Fast and accurate API access:

- Consumers read local materialized cache when fingerprint matches.
- If cache is missing/stale, orchestrator materializes from manifest URI.
- Freshness rules live in manifests and validation tasks.
- API-specific fallback logic belongs in data materialization, not in model or
  execution repos.

Backup:

- Git backs up manifests and schemas.
- Object/DVC/LFS remote backs up large files.
- DB snapshots must be exported deliberately with fingerprint and retention
  class. WAL/SHM files are never source artifacts.

## Artifact Storage And Discovery

Owner repo: `renquant-artifacts`.

Every artifact manifest must include:

- `artifact_id`
- model family
- strategy
- URI
- SHA256/fingerprint
- data fingerprint
- config fingerprint
- code commit(s)
- metrics summary
- promotion status: `prod`, `shadow`, `candidate`, `diagnostic`, `rejected`
- retention class and owner

Finding history:

- Search by `artifact_id`, strategy, model family, promotion status, date, or
  metric keys.
- Accepted artifacts should be easy to list without reading large files.
- Rejected/diagnostic artifacts must keep the verdict so future work does not
  rerun known failures blindly.

## Local LEAN Assembly

Owner repo: `renquant-orchestrator` plus `renquant-backtesting`, pinned by
`RenQuant`.

The orchestrator builds a deterministic LEAN bundle from pinned repos and
manifests. The LEAN directory is an assembly output, not the source of truth.

The bundle records:

- repo commits
- strategy config fingerprint
- data manifest fingerprints
- artifact manifest fingerprints
- assembly timestamp

LEAN must not silently import code from a developer-local random path.

## Automation Requirements

Each subrepo CI runs at minimum:

```bash
make test
```

Each subrepo Makefile exposes:

- `make test`
- `make doctor`
- optionally `make lint`

Umbrella CI should eventually:

- clone or verify pinned subrepos
- install packages editable
- validate all manifests
- run a tiny inference fixture
- run a readonly daily-full smoke

Current umbrella local automation:

```bash
make subrepo-doctor   # required files, remotes, branch, lock commit
make subrepo-test     # doctor plus each subrepo test command
make subrepo-assemble # timestamped local assembly from pinned subrepos
make subrepo-smoke    # orchestrator train -> infer -> dry-run execute -> backtest
```

`make subrepo-assemble` writes `.subrepo_assembly/<timestamp>/` with symlinks
to the pinned repos, `manifest.json`, `pythonpath.txt`, and `env.sh`. It is an
assembly output, not source. It never deletes the umbrella repo.

For production launchd, prefer the isolated runtime root:

```bash
make subrepo-runtime-root
source .subrepo_assembly/current.env  # or source the timestamped env.sh
```

That command runs `scripts/subrepo_assemble.py --sync --runtime-root
.subrepo_runtime/repos`, cloning/fetching pinned repos under
`.subrepo_runtime/repos` instead of checking out developer worktrees. The
generated env exports `RENQUANT_SUBREPO_ROOT` and
`RENQUANT_STRICT_SUBREPO_PATHS=1`, so `daily_multirepo.py` and
`live_multirepo.py` import exactly the lock-pinned code. It also exports
`RENQUANT_OPS_FAIL_CLOSED=1`, which makes scheduled Python and shell
delegates fail closed instead of falling back to umbrella code when pinned
subrepo modules are unavailable.

## Open Migration Work

The first bootstrap created repo skeletons and contracts. `renquant-orchestrator`
now owns the full training-to-trading flow contract and run bundle persistence.
Remaining work is to
port real implementation slices with tests:

1. GBDT training implementation into `renquant-model-gbdt`
2. PatchTST implementation into `renquant-model-patchtst`
3. Runtime 104 tasks into `renquant-pipeline`
4. Broker adapters into `renquant-execution`
5. LEAN/sim tooling into `renquant-backtesting`
6. real data/artifact manifests into `renquant-base-data` and
   `renquant-artifacts`
