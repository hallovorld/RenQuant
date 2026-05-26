# RenQuant Subrepo Migration Manifest

Date: 2026-05-25

This manifest maps the current monorepo paths to the target physical repos.
It is intentionally conservative: move only committed source first, then bring
data and artifacts through explicit manifests.

## Target Mapping

| Current Path | Target Repo | Notes |
|---|---|---|
| `common/` | `renquant-common` | If present; code only. |
| `kernel/walk_forward_splits.py` | `renquant-common` | Shared split primitive. |
| `kernel/regime_labels.py` | `renquant-common` | Shared labels, not strategy logic. |
| `kernel/hmm_regime_labels.py` | `renquant-common` | Shared regime-label utility. |
| `backtesting/renquant_104/training_panel/` | `renquant-model-gbdt` | Keep PatchTST-specific helpers out during split. |
| `scripts/train_104.py` | `renquant-orchestrator` then thin wrapper | Long term: calls model/pipeline packages. |
| `scripts/train_panel*.py` | `renquant-model-gbdt` | After classifying each script. |
| `scripts/fit_*calibrator*.py` | model repo owning the scorer | GBDT calibrators to `model-gbdt`; PatchTST calibrators to `model-patchtst`. |
| `scripts/eval_xgb_*.py` | `renquant-model-gbdt` | Training/eval only. |
| `scripts/patchtst_hf.py` | `renquant-model-patchtst` | Sequence model source. |
| `scripts/fit_hf_patchtst_calibrator.py` | `renquant-model-patchtst` | PatchTST calibrator. |
| `scripts/eval_hf_*.py` | `renquant-model-patchtst` | Sequence-model eval. |
| `scripts/eval_dlinear_*.py` | `renquant-model-patchtst` | Sequence baseline repo. |
| `backtesting/renquant_104/kernel/pipeline/` | `renquant-pipeline` | Task/Job/Pipeline engine. |
| `backtesting/renquant_104/kernel/portfolio_qp/` | `renquant-pipeline` | Portfolio construction. |
| `backtesting/renquant_104/kernel/panel_pipeline/` | split: runtime scorer interfaces to `pipeline`; scorer impls to model repos | Needs careful second pass. |
| `backtesting/renquant_104/kernel/preflight.py` | `renquant-pipeline` | Runtime acceptance/preflight. |
| `backtesting/renquant_104/kernel/model_acceptance.py` | `renquant-pipeline` | Consumes model reports. |
| `backtesting/renquant_104/kernel/persistence.py` | `renquant-pipeline` | Decision trace and ledger schemas; may later split DB client to common. |
| `backtesting/renquant_104/adapters/` | split between `renquant-backtesting` and `renquant-execution` | Sim adapter to backtesting, runner adapter to execution/pipeline. |
| `live/` | `renquant-execution` | Broker adapters and live runner. |
| `scripts/live_only_104.sh` | `renquant-orchestrator` | Workflow wrapper. |
| `scripts/daily_104.sh` | `renquant-orchestrator` | Workflow wrapper. |
| `scripts/intraday_sell_104.sh` | `renquant-orchestrator` | Workflow wrapper. |
| `scripts/retrain_panel.sh` | `renquant-orchestrator` | Calls model-gbdt through pinned package. |
| `dagster_renquant/` | `renquant-orchestrator` | Workflow definitions. |
| `backtesting/renquant_104/main.py` | `renquant-backtesting` | LEAN strategy wrapper. |
| `backtesting/renquant_101/` | `renquant-backtesting` or archive | Legacy strategy. |
| `backtesting/renquant_102/` | `renquant-backtesting` or archive | Legacy strategy. |
| `backtesting/renquant_103/` | `renquant-backtesting` or archive | Legacy strategy. |
| `scripts/export_lean*.py` | `renquant-backtesting` | Data bridge, but data files external. |
| `scripts/analyze_backtest.py` | `renquant-backtesting` | Backtest report tool. |
| `scripts/analyze_wf_trade_forensics.py` | `renquant-backtesting` | If present; sim/forensics owner. |
| `data/` | `renquant-base-data` via DVC/LFS/object manifest | Do not normal-git copy. |
| `backtesting/data/` | `renquant-base-data` via DVC/LFS/object manifest | LEAN mirror. |
| `artifacts/` | `renquant-artifacts` via manifest | Exclude ad hoc experiments until classified. |
| `backtesting/renquant_104/artifacts/` | `renquant-artifacts` via manifest | Accepted/staging/shadow registries only. |
| `backtesting/renquant_104/models/` | `renquant-artifacts` or `model-gbdt` fixtures | Per-ticker policy metadata needs review; many files are generated. |
| `Notebooks/` | `renquant-research` | No prod dependency. |
| `doc/` | `renquant-orchestrator` initially | Later split to docs if useful. |
| `tests/` | split by owner | Integration tests stay in umbrella. |

## Must Exclude From All Source Repos

- `.env`
- `*.db-shm`
- `*.db-wal`
- local `live_state*.json` unless explicitly redacted fixture
- `live/logs/`
- `logs/`
- `__pycache__/`
- `.DS_Store`
- `.pytest_cache/`
- `.ruff_cache/`
- `catboost_info/`
- `_hf_trainer/` checkpoint directories unless registered artifact pointers
- ad hoc `artifacts/**` files without manifest owner/verdict

## First-Wave Repos

First wave should be enough for agent parallelism without exploding
coordination overhead:

1. `renquant-common`
2. `renquant-model-gbdt`
3. `renquant-model-patchtst`
4. `renquant-pipeline`
5. `renquant-execution`
6. `renquant-backtesting`
7. `renquant-base-data`
8. `renquant-artifacts`
9. `RenQuant` as umbrella/orchestrator

Defer `renquant-research` and `renquant-strategy-104` until the first wave can
run CI.

## Repo Creation Preflight

Before creating any physical repo:

```bash
git status --short > split-status-before.txt
git rev-parse HEAD > split-source-head.txt
git ls-files > split-tracked-files.txt
```

Keep those files in an operator archive, not necessarily committed.

Each generated repo should start from committed `HEAD` only. Dirty files must
be deliberately copied later by owner decision.

## Suggested Local Workflow

Use clean clones:

```bash
cd /Users/renhao/git/github
git clone RenQuant renquant-common
git clone RenQuant renquant-model-gbdt
git clone RenQuant renquant-model-patchtst
git clone RenQuant renquant-pipeline
git clone RenQuant renquant-execution
git clone RenQuant renquant-backtesting
git clone RenQuant renquant-base-data
git clone RenQuant renquant-artifacts
```

Then filter each clone to its owned paths using `git filter-repo` when
available. If `git filter-repo` is unavailable, use `git subtree split` for
single-prefix repos and do multi-prefix repos with a clean archive import
first, preserving source commit references in the initial commit message.

Do not run history-rewriting commands in the active `RenQuant` working tree.

## Boundary Tests To Add

Each repo needs an import-boundary test. Examples:

- `renquant-common`: importing the package must not import `alpaca`, `xgboost`,
  `torch`, `lean`, or broker modules.
- `renquant-model-gbdt`: must not import `live`, broker modules, or QP order
  emitters.
- `renquant-model-patchtst`: must not import `live`, broker modules, or GBDT
  training internals.
- `renquant-pipeline`: may import model scorer interfaces, but not model
  training modules.
- `renquant-execution`: must not import model training modules or notebooks.
- `renquant-backtesting`: must not import live broker credentials or order
  submitters.

## Data and Artifact Policy

`renquant-base-data` and `renquant-artifacts` are registry repos, not dumping
grounds.

Every large object needs:

- URI or DVC/LFS pointer.
- SHA256/fingerprint.
- schema version.
- created_at.
- owner repo.
- retention class: `prod`, `shadow`, `diagnostic`, `scratch`, or `expired`.
- verdict for experiments: `accepted`, `rejected`, `diagnostic-only`, or
  `unknown`.

`unknown` artifacts cannot be used by production workflows.
