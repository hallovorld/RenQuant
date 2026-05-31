# RenQuant Multi-Repo — How It Works & SOP

Canonical cross-repo reference. **One source of truth** — subrepos do NOT copy this;
they link to it (see §5). Companion: `subrepo-operating-model.md` (role table),
`subrepo-split-rfc.md` (the original split decision).

## 1. Architecture — the model-factory pipeline

```
 renquant-base-data ── data manifests / dataset input ─┐
 renquant-common ───── shared code (Task/Job/Pipeline,  │
   purged_cv, walk_forward_splits, hmm_regime_labels,   ├─► renquant-model  ── trained model ──► renquant-artifacts
   config_consistency)                                  │   = MODEL FACTORY                      = model registry
 renquant-pipeline ── shared regime/config code ────────┘   (renquant_model_gbdt +                     │
                                                              renquant_model_patchtst                   │
                                                              + renquant_model_common)                  ▼
                                                                          consumed by ── strategy-104 / backtesting / pipeline (runtime) / orchestrator
```

The umbrella **RenQuant** owns no model/runtime code — it is the integration harness:
it pins every subrepo in `subrepos.lock.json`, holds the canonical (gitignored) `data/`
store, and hosts these cross-repo docs.

## 2. Active repos (9) — each owns ONE subject

| Repo | Owns |
|---|---|
| renquant-common | shared primitives + contracts + shared training/eval utils |
| renquant-base-data | data manifests + the training-data input |
| renquant-pipeline | runtime inference/decision pipeline (+ shared regime/config) |
| **renquant-model** | **MODEL FACTORY** — research + train GBDT & PatchTST families, produce models |
| renquant-artifacts | model/artifact registry + contracts (receives factory output) |
| renquant-strategy-104 | active strategy config; consumes models from artifacts |
| renquant-backtesting | sim / LEAN / WF / forensics; consumes models |
| renquant-execution | broker execution + order audit |
| renquant-orchestrator | daily run-bundle orchestration across pins |

There is **no `renquant-model-xgb` / `renquant-model-patchtst` repo** — the families are
*packages* (`renquant_model_gbdt`, `renquant_model_patchtst`) inside the one merged
`renquant-model` (RFC P3). Work on either family happens in `renquant-model`.

## 3. SOP — model lifecycle (the "xgb just promoted a model" flow)

A model family in `renquant-model` going from idea → live:

1. **BUILD** — train a candidate through the canonical engine, never ad-hoc:
   GBDT → `renquant_model_gbdt.ModelTrainingJob` (driver: orchestrator `train_gbdt`);
   PatchTST → `renquant_model_patchtst` (trainer + `research.py` harness).
   Inputs: data from base-data store, shared code from common/pipeline.
2. **VALIDATE** (CLAUDE.md §5.2 / §5.13.4a) — placebo-clean walk-forward IC (shuffle +
   time-shift ≈ 0), WF gate vs SPY, then DSR/PBO. **Tiers**: T1 reject / T2 screen
   (not live) / T3 live-promotable (DSR>0.5 or PBO<0.5).
3. **PUBLISH** — write the model + a **fingerprinted manifest** to **renquant-artifacts**
   (`promotion_status: candidate`). The artifact MUST carry `config_fingerprint` equal to
   what the runtime scorer computes from the strategy config (`config_consistency`), or
   the scorer fail-closes. T3 winners flip to `active`.
4. **CONSUME** — consumers don't import the factory; they load the artifact:
   `renquant-strategy-104` config `ranking.panel_scoring.artifact_path` points at it;
   sim/backtest/live/orchestrator resolve it through the artifacts registry.
5. **PIN** — advance the consuming side (strategy config `artifact_path` + the
   `subrepos.lock.json` pin for the producing repo). No live flip without §3 Tier 3.

So "renquant-model-xgb promoted a model" really means: the GBDT family in
`renquant-model` produced a Tier-3 model → published to `renquant-artifacts` → the
strategy-104 config (and the WF-gate metadata) is updated to point at it → daily/sim
loads it. The factory NEVER writes directly into a consumer.

## 4. Dev workflow (every repo)
- **Placement**: new code goes in the repo that OWNS the subject (model research →
  renquant-model; runtime → pipeline; shared util → common). Never the umbrella; never
  duplicate across repos.
- **Merge**: PR-based for ALL 13 repos per umbrella `CLAUDE.md` §3.1
  (2026-05-30 mandate, reverses the deleted 2026-05-27 verbal-merge convention).
  Feature branch → `make test` green → `git push -u origin <branch>` →
  `gh pr create --base main` → after verbal approval, `gh pr merge --merge --delete-branch`.
  NEVER `git push origin main` from a branch. Per umbrella `CLAUDE.md` §3.2,
  also `git fetch origin && git rebase origin/main` before opening any PR
  and before declaring merge-ready.
- **Pins**: after a subrepo merge, advance its commit in the umbrella `subrepos.lock.json`.
- **Tests**: each subrepo's `make test` (incl. import-boundary + no-raw-regime-string
  scans) must pass before merge.

## 5. Doc / RFC sharing policy  ←  (answer: NOT replicated)
Cross-repo docs (this SOP, the RFC, the operating model, role definitions) live in
**exactly one place — `RenQuant/doc/arch/`** — and are referenced, never copied.
Replication causes drift: the archived model shells carried stale `CLAUDE.md`/`AGENTS.md`
copies that misdirected an agent into a dead repo. Therefore:
- Cross-repo architecture/RFC/SOP/roles → **umbrella `doc/arch/` only**.
- Each subrepo keeps a **thin `AGENTS.md`** = a one-line pointer to the umbrella docs +
  ONLY repo-local specifics (its package layout, its `make test`). No copied architecture.

## 6. Archived repos
`renquant-model-gbdt`, `renquant-model-patchtst` — merged into `renquant-model` (RFC P3);
local checkouts + GitHub remotes deleted. Rollback, if ever needed, is the pre-merge
history reachable in `renquant-model`'s own log. Do not recreate per-family repos without
revisiting RFC P3.
