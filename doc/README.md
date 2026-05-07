# RenQuant Documentation

Active strategy: **`renquant_104`** (panel-LTR cross-sectional ranking with cvxpy + CLARABEL portfolio QP). `renquant_103` kept for rollback only.

> **Where to start**: [`STATUS.md`](STATUS.md) → [`arch/strategy-104.md`](arch/strategy-104.md) → [`arch/overview.md`](arch/overview.md) → [`components/portfolio-qp.md`](components/portfolio-qp.md) → [`ops/usage.md`](ops/usage.md).

---

## arch/ — architecture & strategy specs

| Doc | Contents |
|-----|----------|
| [strategy-104.md](arch/strategy-104.md) | **Active strategy** — panel-LTR end-to-end (XGBoost ranker, NGBoost head, calibrator, portfolio QP, acceptance gates) |
| [overview.md](arch/overview.md) | Pipeline + data flow; `kernel/pipeline/` and `kernel/panel_pipeline/` composition |
| [decision-graph-103.md](arch/decision-graph-103.md) | Decision flowchart (shared 103/104 trunk; 104 layer extends) |
| [strategy-103.md](arch/strategy-103.md) | Rollback strategy spec — kept for reference only |
| [indicators.md](arch/indicators.md) | Indicator catalog with parameters |
| [models.md](arch/models.md) | Per-backend model type reference + decision guide |

## components/ — subsystem deep dives

| Doc | Contents |
|-----|----------|
| [panel-ltr.md](components/panel-ltr.md) | Panel-LTR primer + glossary |
| [buy-logic.md](components/buy-logic.md) | 3 quality gates + portfolio QP integration |
| [sell-logic.md](components/sell-logic.md) | SellGateB + LimitSellsPerBar |
| [calibration.md](components/calibration.md) | Score-DB + isotonic + global calibrator |
| [rotation.md](components/rotation.md) | Holdings rotation logic |
| [transformer.md](components/transformer.md) | Daily + hourly transformer (kept for future, not active) |
| [portfolio-qp.md](components/portfolio-qp.md) | **cvxpy + CLARABEL convex QP** (Boyd/Stanford cvxportfolio idiom; 2026-05-06 refactor) |
| [databases.md](components/databases.md) | runs.db schema + role split |
| [training-pipeline.md](components/training-pipeline.md) | FullTrainingPipeline + PanelTrainingPipeline orchestration |
| [trade-evaluation.md](components/trade-evaluation.md) | RL off-policy evaluation (deferred) |
| [macro-factor-frame-design.md](components/macro-factor-frame-design.md) | Macro factor design — currently disabled, see STATUS.md |

## ops/ — operations & runbooks

| Doc | Contents |
|-----|----------|
| [usage.md](ops/usage.md) | 5 workflow modes (research / validation / analysis / live / scheduled) |
| [setup.md](ops/setup.md) | Apple Silicon environment setup |
| [environment.md](ops/environment.md) | Reproducibility: conda env, pinned versions, Docker |
| [golden-config.md](ops/golden-config.md) | Current golden state + drift policy |
| [transformer-promotion.md](ops/transformer-promotion.md) | Transformer promotion checklist |
| [maintenance-103.md](ops/maintenance-103.md) | 103 rollback maintenance workflow |

## research/ — background reading

| Doc | Contents |
|-----|----------|
| [papers-implemented.md](research/papers-implemented.md) | Academic papers wired into production code |
| [scoring-research.md](research/scoring-research.md) | Calibrated scoring + panel-LTR notes |
| [rotation-research.md](research/rotation-research.md) | Rotation literature scan |
| [watchlist-100.md](research/watchlist-100.md) | Watchlist construction methodology |
| [panel-sunday-sweep.md](research/panel-sunday-sweep.md) | Panel hyperparameter sweep findings |
| [alpaca-crypto-btc.md](research/alpaca-crypto-btc.md) | Alpaca crypto integration evaluation |
| [failed-experiments-log.md](research/failed-experiments-log.md) | **Mandatory log** (CLAUDE.md §5.7): every failed experiment + why |

## experiments/ — measured A/B results

| Doc | Contents |
|-----|----------|
| [ab-journal.md](experiments/ab-journal.md) | Running A/B comparison journal |
| [panel-training-runs.md](experiments/panel-training-runs.md) | Per-run training results table |
| [post-tier1-followups.md](experiments/post-tier1-followups.md) | Investigation queue post-Tier-1 |
| [sim-ab-results.md](experiments/sim-ab-results.md) | Sim-level A/B with APY/Sharpe deltas |

## top-level

| Doc | Contents |
|-----|----------|
| [STATUS.md](STATUS.md) | **Read first** — current state, recent results, open priorities |
| [roadmap.md](roadmap.md) | Living roadmap — goals, open decisions, deferred work |

## archives/

- `archives/sessions/` — daily handoff notes
- `archives/audits/` — historical audits + post-mortems
- `archives/assessments/` — periodic system assessments
- `archives/shelved/` — closed-experiment design docs (preserved for `git log --follow` provenance)
