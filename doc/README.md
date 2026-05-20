# RenQuant Documentation

Active strategy: **`renquant_104`** (panel-LTR cross-sectional ranking, **172 features**, cvxpy + CLARABEL portfolio QP, **NGBoost head ON in prod / σ-wire dormant**, HF PatchTST shadow active). `renquant_103` archived.

> **Where to start:** [`roadmap.md`](roadmap.md) → [`arch/strategy-104.md`](arch/strategy-104.md) → [`arch/overview.md`](arch/overview.md) → [`components/portfolio-qp.md`](components/portfolio-qp.md) → [`ops/usage.md`](ops/usage.md).
>
> For PatchTST capability roadmap: [`research/2026-05-19-patchtst-improvement-plan.md`](research/2026-05-19-patchtst-improvement-plan.md) + lessons [`experiments/2026-05-19-hf-trainer-refactor-journal.md`](experiments/2026-05-19-hf-trainer-refactor-journal.md).

---

## Top-level

| Doc | Contents |
|---|---|
| [roadmap.md](roadmap.md) | **Read first** — current state, ROI-ranked active items, closed/rejected items, paper citations |
| [STATUS.md](STATUS.md) | Thin pointer (was duplicating roadmap.md "Current state"; now lists deltas since 2026-05-09) |

## arch/ — architecture & strategy specs

| Doc | Contents |
|---|---|
| [strategy-104.md](arch/strategy-104.md) | **Active strategy** end-to-end (XGB ranker, calibrator, QP, gates, bug-fix lineage) |
| [overview.md](arch/overview.md) | Pipeline + data flow (kernel/pipeline/, kernel/panel_pipeline/) |
| [indicators.md](arch/indicators.md) | Indicator catalog with parameters |
| [models.md](arch/models.md) | Per-backend model type reference |

## components/ — subsystem deep dives

| Doc | Contents |
|---|---|
| [panel-ltr.md](components/panel-ltr.md) | Panel-LTR primer + glossary |
| [buy-logic.md](components/buy-logic.md) | Quality gates + cost-aware wash-sale + portfolio QP integration |
| [sell-logic.md](components/sell-logic.md) | SellGateB + LimitSellsPerBar |
| [calibration.md](components/calibration.md) | Score-DB + isotonic + global calibrator |
| [rotation.md](components/rotation.md) | Holdings rotation logic |
| [portfolio-qp.md](components/portfolio-qp.md) | **cvxpy + CLARABEL convex QP** (cvxportfolio idiom; σ-band cap per BUG #7) |
| [databases.md](components/databases.md) | runs.db schema + role split |
| [training-pipeline.md](components/training-pipeline.md) | FullTrainingPipeline + PanelTrainingPipeline orchestration |

## ops/ — operations & runbooks

| Doc | Contents |
|---|---|
| [usage.md](ops/usage.md) | 5 workflow modes (research / validation / analysis / live / scheduled) |
| [setup.md](ops/setup.md) | Apple Silicon environment setup |
| [environment.md](ops/environment.md) | Reproducibility: conda env, pinned versions |
| [tech-stack.md](ops/tech-stack.md) | Languages, frameworks, infra dependencies |
| [golden-config.md](ops/golden-config.md) | Current golden state + drift policy |
| [insider-trades-setup.md](ops/insider-trades-setup.md) | SEC EDGAR User-Agent setup |
| [cloud-backup-setup.md](ops/cloud-backup-setup.md) | Backup configuration |

## research/ — background

| Doc | Contents |
|---|---|
| [failed-experiments-log.md](research/failed-experiments-log.md) | **Mandatory log** (CLAUDE.md §5.7): every NO-GO with reason (E1–E55) |
| [papers-implemented.md](research/papers-implemented.md) | Academic papers wired into production code |
| [ic-evaluation-methodology.md](research/ic-evaluation-methodology.md) | Walk-forward CV + sanity battery rules |
| [scoring-research.md](research/scoring-research.md) | Calibrated scoring + panel-LTR notes |
| [rotation-research.md](research/rotation-research.md) | Rotation literature scan |
| [watchlist-100.md](research/watchlist-100.md) | Watchlist construction methodology |
| [panel-sunday-sweep.md](research/panel-sunday-sweep.md) | Panel hyperparameter sweep findings |
| [alpaca-crypto-btc.md](research/alpaca-crypto-btc.md) | Alpaca crypto integration evaluation |

## experiments/ — measured A/B results

| Doc | Contents |
|---|---|
| [ab-journal.md](experiments/ab-journal.md) | Running A/B comparison journal |
| [panel-training-runs.md](experiments/panel-training-runs.md) | Per-run training results table |
| [sim-ab-results.md](experiments/sim-ab-results.md) | Sim-level A/B with APY/Sharpe deltas |
| [post-tier1-followups.md](experiments/post-tier1-followups.md) | Post-Tier1 follow-up checklist |

## archives/ — preserved for provenance

- `archives/sessions/` — daily handoff notes
- `archives/audits/` — historical audits + post-mortems
- `archives/assessments/` — periodic system assessments
- `archives/shelved/` — closed-feature design docs (kept for `git log --follow`)
   - `strategy-103.md` (rollback ref)
   - `decision-graph-103.md`
   - `macro-factor-frame-design.md` (REJECTED variants)
   - `transformer.md` + `transformer-promotion.md` (DISABLED)
   - `trade-evaluation.md` (RL-OPE deferred)
   - `maintenance-103.md` (old strategy)
