# RenQuant Repository Map

> AUTO-GENERATED from `RenQuant/subrepos.lock.json` by `scripts/sync_subrepo_docs.py`.
> **Do not edit by hand** — edit the lock and re-run the sync. `subrepo_doctor.py` fails if this drifts.

Gives an agent local big-picture context when starting inside any RenQuant repo.
Canonical cross-repo docs (read, do not copy): `RenQuant/doc/arch/multirepo-sop.md` (architecture + SOP) and `subrepo-operating-model.md` (roles).

The umbrella `RenQuant` is never deleted, emptied, or rewritten.

## Repositories

| Repo | Remote | Role |
|---|---|---|
| `RenQuant` | `https://github.com/hallovorld/RenQuant` | permanent umbrella/orchestrator and rollback source |
| `renquant-common` | `https://github.com/hallovorld/renquant-common` | shared primitives + contracts + shared TRAINING/EVAL utilities (Task/Job/Pipeline, purged_cv, walk_forward_splits, hmm_regime_labels, config_consistency) — imported by the model factory & pipeline |
| `renquant-strategy-104` | `https://github.com/hallovorld/renquant-strategy-104` | active 104 strategy config; CONSUMES models from renquant-artifacts |
| `renquant-model` | `https://github.com/hallovorld/renquant-model` | MODEL FACTORY — ingests base-data + common code, runs research/training (GBDT + PatchTST families), and produces models published to renquant-artifacts |
| `renquant-pipeline` | `https://github.com/hallovorld/renquant-pipeline` | runtime inference/decision pipeline; shares regime/config code with trainers |
| `renquant-execution` | `https://github.com/hallovorld/renquant-execution` | broker execution and order-audit pipeline |
| `renquant-backtesting` | `https://github.com/hallovorld/renquant-backtesting` | backtesting, LEAN assembly, simulation, forensics; CONSUMES models from renquant-artifacts |
| `renquant-base-data` | `https://github.com/hallovorld/renquant-base-data` | base-data manifests + the training-data INPUT consumed by the model factory |
| `renquant-artifacts` | `https://github.com/hallovorld/renquant-artifacts` | artifact registry — RECEIVES trained models from the factory; consumed by sim/backtest/strategy-104 |
| `renquant-orchestrator` | `https://github.com/hallovorld/renquant-orchestrator` | pinned-subrepo daily orchestration + run bundles (wires factory output into daily/sim/backtest) |

## System flow — the model-factory pipeline

1. `renquant-base-data` publishes the training-data input + freshness/fingerprint contracts.
2. `renquant-common` + `renquant-pipeline` provide the shared code (Task/Job/Pipeline,
   purged_cv, walk_forward_splits, hmm_regime_labels, config_consistency).
3. `renquant-model` is the **model factory**: the `renquant_model_gbdt` and
   `renquant_model_patchtst` packages research + train models from base-data input and
   shared code, then publish artifact manifests to `renquant-artifacts`.
4. `renquant-artifacts` is the model registry (contracts + promotion status).
5. `renquant-strategy-104`, `renquant-pipeline` (runtime), `renquant-backtesting`, and
   `renquant-orchestrator` CONSUME models by `artifact_path` — never by importing the factory.
6. `renquant-execution` consumes order intents and performs broker actions with audit records.
7. `RenQuant` (umbrella) pins the whole assembly in `subrepos.lock.json` and stays the
   permanent integration harness + rollback source.

## Model lifecycle (build → validate → publish → consume)

See `RenQuant/doc/arch/multirepo-sop.md` §3 for the full SOP. In brief: the factory
BUILDs a candidate through the canonical engine, VALIDATEs it placebo-clean (WF IC +
shuffle/time-shift placebos, DSR/PBO, 3-tier gate), PUBLISHes the fingerprinted model to
`renquant-artifacts`, then the consuming side is PINned (`artifact_path` + lock pin). No
live flip without Tier 3. The factory never writes into a consumer.

## Cross-repo rules

- New code goes in the repo that OWNS the subject; never duplicate across repos; never add
  code to the umbrella `RenQuant` (integration/rollback only).
- Use `renquant-common` pipeline primitives for every workflow.
- Cross-repo docs (architecture/SOP/roles) live ONCE in `RenQuant/doc/arch/` and are
  referenced, never copied — replication is what causes stale-doc drift.
- Workflow: feature branch → `make test` green → local `git merge --no-ff` into `main` →
  push. **No GitHub PRs.** After a subrepo merge, advance its pin in `subrepos.lock.json`.
- Large data, checkpoints, DBs, experiment dumps are referenced by manifest + fingerprint,
  not committed. A subrepo commit is not production-active until the umbrella pins it.
