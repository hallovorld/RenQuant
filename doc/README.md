# RenQuant Documentation

Themed index. The active strategy is `renquant_104` (panel-LTR cross-sectional ranking). `renquant_103` is the rollback. Doc tree last reorganized 2026-04-26.

> **Where to start as a new contributor**: read [`arch/strategy-104.md`](arch/strategy-104.md) → [`arch/overview.md`](arch/overview.md) → [`components/panel-ltr.md`](components/panel-ltr.md) → [`components/model-selection.md`](components/model-selection.md) → [`ops/usage.md`](ops/usage.md). That sequence covers ~90% of the operational surface.

---

## arch/ — Architecture & strategy specs

| Doc | Contents |
|-----|----------|
| [strategy-104.md](arch/strategy-104.md) | **Current active strategy** — panel-LTR end-to-end, including XGBoost ranker, NGBoost head, calibrator, portfolio QP, acceptance gates, macro factors |
| [overview.md](arch/overview.md) | Pipeline + data flow + adapter isolation; how `kernel/pipeline/` and `kernel/panel_pipeline/` compose |
| [decision-graph-103.md](arch/decision-graph-103.md) | Decision flowchart — every branch in the inference pipeline (shared 103/104 trunk; 104 layer extends rather than replaces) |
| [strategy-103.md](arch/strategy-103.md) | Rollback strategy spec — 3-layer regime detector + per-symbol scanner |
| [indicators.md](arch/indicators.md) | Indicator catalog with parameters (uniform `(df, **params) -> DataFrame` API) |
| [models.md](arch/models.md) | Per-backend model type reference + decision guide |

---

## components/ — Subsystem deep dives

| Doc | Contents |
|-----|----------|
| [model-selection.md](components/model-selection.md) | **★ 4-tier SOP** — acceptance gates (Phase 1+2), backend tournament (Phase 3), shadow/challenger (Phase 4) |
| [panel-ltr.md](components/panel-ltr.md) | Panel-LTR primer + glossary (cross-section, panel matrix, OOS IC, ranker mechanics) |
| [buy-logic.md](components/buy-logic.md) | 3 quality gates + portfolio QP (operator runbook merged) |
| [sell-logic.md](components/sell-logic.md) | SellGateB + LimitSellsPerBar (round-7 additions) |
| [calibration.md](components/calibration.md) | Saturation finding + score-DB design + global calibrator mechanics |
| [transformer.md](components/transformer.md) | Daily + hourly transformer; Bug #21/#23/#24 + acceptance protections |
| [macro-factor-frame-design.md](components/macro-factor-frame-design.md) | VIX/HYG/UUP/DBC/GLD/TLT/XLV/XLU/KRE/MTUM/USMV cross-asset broadcast |
| [metadata-db-and-backup-plan.md](components/metadata-db-and-backup-plan.md) | Plan: model metadata DB columns + cloud backup (Backblaze B2). DEFERRED — see roadmap |
| [portfolio-qp.md](components/portfolio-qp.md) | QP solver for rotation under correlation + sector + concentration constraints |
| [rotation.md](components/rotation.md) | Holdings rotation logic (joint actions, greedy fallback) |
| [databases.md](components/databases.md) | runs.db schema reference + role split (live vs sim) |
| [db-design-decision-factors.md](components/db-design-decision-factors.md) | Per-decision factor logging design |
| [training-pipeline.md](components/training-pipeline.md) | FullTrainingPipeline + PanelTrainingPipeline orchestration |
| [trade-evaluation.md](components/trade-evaluation.md) | RL off-policy evaluation (OPE) design — DEFERRED |

---

## ops/ — Operations & runbooks

| Doc | Contents |
|-----|----------|
| [usage.md](ops/usage.md) | **5 workflow modes** — research / validation / analysis / live / scheduled. Includes scripts/ CLI reference |
| [setup.md](ops/setup.md) | Apple Silicon environment setup, prerequisites, daily activation |
| [environment.md](ops/environment.md) | Reproducibility: conda env, pip pins, version lockfile, Docker requirements |
| [tech-stack.md](ops/tech-stack.md) | Tool choices and rationale (xgboost vs lgbm vs torch, etc.) |
| [golden-config.md](ops/golden-config.md) | Current golden state — strategy_config.golden.json snapshot + drift policy |
| [transformer-promotion.md](ops/transformer-promotion.md) | Transformer-specific promotion checklist (separate from XGBoost gate flow) |
| [maintenance-103.md](ops/maintenance-103.md) | 103 maintenance workflow (review / alignment / validation / commit cycle) |

---

## research/ — Research notes (background reading)

| Doc | Contents |
|-----|----------|
| [papers-implemented.md](research/papers-implemented.md) | Index of academic papers wired into production code (Lo 2002, Garleanu-Pedersen 2013, …) |
| [scoring-research.md](research/scoring-research.md) | Calibrated scoring + panel-LTR + feature neutralization notes |
| [rotation-research.md](research/rotation-research.md) | Rotation literature scan |
| [watchlist-100.md](research/watchlist-100.md) | Watchlist construction methodology + selection criteria |
| [panel-sunday-sweep.md](research/panel-sunday-sweep.md) | Panel hyperparameter sweep findings |
| [alpaca-crypto-btc.md](research/alpaca-crypto-btc.md) | Alpaca crypto integration evaluation (BTC) |

---

## experiments/ — Measured A/B results

| Doc | Contents |
|-----|----------|
| [ab-journal.md](experiments/ab-journal.md) | Running journal of A/B comparisons + verdicts |
| [panel-training-runs.md](experiments/panel-training-runs.md) | Per-run training results table (OOS IC, train IC, panel shape, features) |
| [panel-backend-comparison.md](experiments/panel-backend-comparison.md) | Cross-backend OOS IC comparison (XGBoost vs LightGBM vs Transformer) |
| [panel-ic-improvement.md](experiments/panel-ic-improvement.md) | Tier 1 / Tier 1.5 retrain history toward higher OOS IC |
| [post-tier1-followups.md](experiments/post-tier1-followups.md) | Investigation queue post-Tier-1 retrain |
| [sim-ab-results.md](experiments/sim-ab-results.md) | Simulation-level A/B with APY/Sharpe deltas |
| [rust-transformer-ic.md](experiments/rust-transformer-ic.md) | Rust transformer scorer parity check + perf benchmarks |

---

## Top-level

| Doc | Contents |
|-----|----------|
| [roadmap.md](roadmap.md) | Living roadmap — current goals, open decisions, deferred items, completed work |
| [REORG_PLAN.md](REORG_PLAN.md) | Doc-reorg plan history (Phase 1-5 mega-refactor 2026-04-26) |

---

## archives/

Historical session logs and audit reports live in `archives/sessions/` and `archives/audits/`. Browse if you need provenance for a specific decision; not part of normal operating documentation.
