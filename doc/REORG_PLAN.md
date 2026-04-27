# Doc Reorg Plan (2026-04-26)

**Trigger:** User mega-ask 2026-04-26 — *"prune all docs, merge dated → non-dated, structure each clearly, files not too long, docs cross-reference, CLAUDE.md / arch / roadmap concise, doc/ split into logical subdirs"*

**Status:** Phase 1 (plan committed). Execute via Phases 2-5. **Delete this file after Phase 5.**

## Goals

1. **Eliminate date-suffix sprawl** for topical content. Sessions/audits stay dated (they're history), but design docs lose dates and live in topical homes.
2. **Logical subdirs**: `arch/`, `components/`, `ops/`, `research/`, `experiments/`, `archives/{sessions,audits}/`.
3. **Per-file caps**: aim for ≤300 lines for design docs, ≤500 for theory primers. Split when above.
4. **Cross-references**: every doc points to its prerequisites + related docs (footer "See also" section).
5. **CLAUDE.md, architecture.md, roadmap.md**: aggressive trim — table of contents + links, not encyclopedia entries.
6. **Explanatory docs (panel-ltr-primer, papers-implemented) can be longer** but no repetition.

## Current state — 63 docs, 17,978 lines

```
38 dated     (38 × _2026-04-* or _2025-*)
25 non-dated
8 docs > 500 lines (longest: research_scoring.md @ 1389)
```

## Target structure

```
doc/
├── README.md                # top-level index (regen after move)
├── roadmap.md               # was improvement_roadmap.md, trimmed
│
├── arch/                    # foundational architecture — read first
│   ├── overview.md          # was architecture.md, trimmed
│   ├── strategy-103.md      # was renquant_103_design.md
│   ├── strategy-104.md      # was renquant_104_design.md
│   ├── decision-graph-103.md# was logic_graph_103.md
│   ├── indicators.md        # unchanged
│   ├── models.md            # unchanged
│   └── data-stores.md       # split from old database.md (architecture only)
│
├── components/              # subsystem designs (theory + impl)
│   ├── panel-ltr.md         # was panel_ltr_primer.md (theory)
│   ├── panel-ltr-impl.md    # was renquant_104_design.md scoring section + transformer_design.md merged
│   ├── transformer.md       # transformer_hourly_stage_c2_design + renquant_104_transformer_design
│   ├── buy-logic.md         # buy_logic_redesign_2026-04-26 + buy_logic_portman_ops merged
│   ├── sell-logic.md        # sell_quality_floor_2026-04-26 (#1 SellGateB + #3 LimitSells)
│   ├── portfolio-qp.md      # unified_portfolio_action_design_2026-04-26
│   ├── calibration.md       # calibrator_saturation + calibration_db_design merged
│   ├── rotation.md          # rotate_algorithm_design + rotation_research_2026-04-24
│   ├── trade-evaluation.md  # trade_evaluation_rl_design_2026-04-26 (RL-OPE)
│   ├── training-pipeline.md # model_training_design.md trimmed
│   └── databases.md         # database.md + db_design_decision_factors merged (schema + queries)
│
├── ops/                     # operator manuals
│   ├── runbook.md           # buy_logic_portman_ops + renquant_103_maintenance_workflow merged
│   ├── golden-config.md     # golden_config_2026-04-23 (kept as living "current best")
│   ├── setup.md             # unchanged
│   ├── usage.md             # unchanged
│   ├── environment.md       # unchanged
│   ├── tech-stack.md        # unchanged
│   ├── backup-plan.md       # extracted from roadmap (cloud backup section)
│   └── transformer-promotion.md  # unchanged
│
├── research/                # papers + idea exploration
│   ├── papers-implemented.md # unchanged
│   ├── scoring-research.md  # was research_scoring.md
│   ├── alpaca-crypto-btc.md # was alpaca_crypto_btc_feasibility_2026-04-26
│   ├── watchlist-100.md     # was watchlist_100_candidates_2026-04-24
│   └── panel-sunday-sweep.md # was panel_sunday_sweep_2026-04-26
│
├── experiments/             # measured results, not designs
│   ├── ab-journal.md        # was ab_experiments.md
│   ├── panel-training-runs.md # unchanged
│   ├── panel-backend-comparison.md # was panel_backend_comparison_2026-04-25
│   ├── panel-ic-improvement.md # was panel_ic_improvement_2026-04-24
│   ├── sim-ab-results.md    # was sim_ab_results_2026-04-26
│   ├── rust-transformer-ic.md # was rust_transformer_ic_baseline
│   └── post-tier1-followups.md # was post_tier1_followups_2026-04-25
│
└── archives/                # historical record — read for context, not source of truth
    ├── sessions/
    │   ├── 2026-04-23.md    # session_handoff_2026-04-23
    │   ├── 2026-04-24.md    # session_summary_2026-04-24 + session_bug_audit
    │   ├── 2026-04-25.md    # session_handoff_2026-04-25
    │   └── 2026-04-26.md    # session_summary_2026-04-26 + session_self_audit
    └── audits/
        ├── 2026-04-24-bugs.md          # bug_audit_2026-04-24 + bug_audit_r3
        ├── 2026-04-24-panel-ltr.md     # panel_ltr_audit_2026-04-24
        ├── 2026-04-24-panel-pipeline.md # panel_pipeline_audit_2026-04-24
        ├── 2026-04-25-ngboost-tx.md    # ngboost_transformer_audit_2026-04-25
        ├── 2026-04-26-3hour.md         # audit_3hour_2026-04-26
        ├── 2026-04-26-mega.md          # mega_audit_phase{1-6} consolidated
        ├── 2026-04-26-transformer.md   # transformer_audit_2026-04-26
        └── 2026-04-26-xgboost-rotation.md
```

**Final count:** ~50 files (was 63), avg ~270 lines (was ~285).

## Migration map (file-by-file)

### MERGE — dated → topical home

| From (dated) | To | Notes |
|---|---|---|
| `buy_logic_redesign_2026-04-26.md` (363 L) | `components/buy-logic.md` | Theory + 3 gates A/B/C |
| `buy_logic_portman_ops_2026-04-26.md` (369 L) | `ops/runbook.md` | Operator-facing |
| `unified_portfolio_action_design_2026-04-26.md` (481 L) | `components/portfolio-qp.md` | 7-stage QP design |
| `sell_quality_floor_2026-04-26.md` (269 L) | `components/sell-logic.md` | SellGateB + LimitSells |
| `calibrator_saturation_2026-04-26.md` | `components/calibration.md` | Plus the round-7 fix section |
| `calibration_db_design_2026-04-26.md` | `components/calibration.md` | DB schema for calibration |
| `db_design_decision_factors_2026-04-26.md` | `components/databases.md` | TDS schema |
| `trade_evaluation_rl_design_2026-04-26.md` | `components/trade-evaluation.md` | Rename only |
| `rotation_research_2026-04-24.md` | `components/rotation.md` | Lit review section |
| `transformer_hourly_stage_c2_design.md` | `components/transformer.md` | Stage C-2 design |
| `transformer_audit_2026-04-26.md` | `archives/audits/2026-04-26-transformer.md` | Audit history |
| `xgboost_rotation_audit_2026-04-26.md` | `archives/audits/2026-04-26-xgboost-rotation.md` | Audit history |
| `bug_audit_2026-04-24.md` + `bug_audit_r3` | `archives/audits/2026-04-24-bugs.md` | Single combined |
| `mega_audit_phase{1-6}_findings_2026-04-26.md` + `_plan_` | `archives/audits/2026-04-26-mega.md` | All 7 collapsed; preserve phase-headers |
| `panel_ltr_audit_2026-04-24.md` | `archives/audits/2026-04-24-panel-ltr.md` | Audit history |
| `panel_pipeline_audit_2026-04-24.md` | `archives/audits/2026-04-24-panel-pipeline.md` | |
| `ngboost_transformer_audit_2026-04-25.md` (1012 L) | `archives/audits/2026-04-25-ngboost-tx.md` | Long; split if >500 after pruning |
| `audit_3hour_2026-04-26.md` | `archives/audits/2026-04-26-3hour.md` | |
| `session_self_audit_2026-04-26.md` | merge into `archives/sessions/2026-04-26.md` | Same date |
| `session_bug_audit_2026-04-24.md` | merge into `archives/sessions/2026-04-24.md` | Same date |
| `session_summary_2026-04-26.md` | `archives/sessions/2026-04-26.md` | |
| `session_summary_2026-04-24.md` | `archives/sessions/2026-04-24.md` | |
| `session_handoff_2026-04-23.md` | `archives/sessions/2026-04-23.md` | |
| `session_handoff_2026-04-25.md` | `archives/sessions/2026-04-25.md` | |
| `panel_backend_comparison_2026-04-25.md` | `experiments/panel-backend-comparison.md` | Rename only |
| `panel_ic_improvement_2026-04-24.md` | `experiments/panel-ic-improvement.md` | |
| `sim_ab_results_2026-04-26.md` | `experiments/sim-ab-results.md` | |
| `post_tier1_followups_2026-04-25.md` | `experiments/post-tier1-followups.md` | |
| `alpaca_crypto_btc_feasibility_2026-04-26.md` | `research/alpaca-crypto-btc.md` | Rename |
| `watchlist_100_candidates_2026-04-24.md` | `research/watchlist-100.md` | Rename |
| `panel_sunday_sweep_2026-04-26.md` | `research/panel-sunday-sweep.md` | Rename |
| `golden_config_2026-04-23.md` | `ops/golden-config.md` | Living doc, keep date in body |

### MOVE only — non-dated → subdir

| From | To |
|---|---|
| `architecture.md` | `arch/overview.md` |
| `renquant_103_design.md` | `arch/strategy-103.md` |
| `renquant_104_design.md` | `arch/strategy-104.md` |
| `renquant_104_transformer_design.md` | `components/transformer.md` (merge) |
| `logic_graph_103.md` | `arch/decision-graph-103.md` |
| `indicators.md` | `arch/indicators.md` |
| `models.md` | `arch/models.md` |
| `database.md` | split → `arch/data-stores.md` (overview) + `components/databases.md` (schema) |
| `panel_ltr_primer.md` | `components/panel-ltr.md` |
| `model_training_design.md` | `components/training-pipeline.md` |
| `rotate_algorithm_design.md` | `components/rotation.md` (merge with rotation_research) |
| `renquant_103_maintenance_workflow.md` | `ops/runbook.md` (merge) |
| `transformer_promotion_plan.md` | `ops/transformer-promotion.md` |
| `setup.md`, `usage.md`, `environment.md`, `tech-stack.md` | `ops/` |
| `papers_implemented.md` | `research/papers-implemented.md` |
| `research_scoring.md` | `research/scoring-research.md` |
| `panel_training_runs.md` | `experiments/panel-training-runs.md` |
| `ab_experiments.md` | `experiments/ab-journal.md` |
| `rust_transformer_ic_baseline.md` | `experiments/rust-transformer-ic.md` |
| `improvement_roadmap.md` | `roadmap.md` (top-level, trim) |
| `README.md` | unchanged at `doc/README.md` (regen index) |

### EXTRACT — split sections out

| Source | Extract to |
|---|---|
| `improvement_roadmap.md` (cloud backup section) | `ops/backup-plan.md` |
| `database.md` (schema details) | `components/databases.md` |
| `database.md` (architectural decisions) | `arch/data-stores.md` |

### TRIM aggressively (target -50% lines)

- `CLAUDE.md` (root) — keep just sections 1-4 (project + env + workflow + library), move detailed indicator/model/strategy descriptions to per-doc links.
- `arch/overview.md` — was 584L, target ~250L.
- `roadmap.md` — was 740L, target ~400L (move history to archives/sessions).
- `arch/strategy-103.md` — was 833L, target ~400L (move algorithm details to components/).

## Phases

| # | Phase | Reversibility | Commit before |
|---|---|---|---|
| 1 | **THIS DOC** | trivial | now |
| 2 | Create subdirs + git-mv all files (no merging yet) | revert via git revert | each subdir batch |
| 3 | Merge dated → topical, delete originals | irreversible — need git revert + manual reconstruction | each merge |
| 4 | Trim CLAUDE.md / arch / roadmap | text-only edits | each trim |
| 5 | Add cross-references + regen README index | text-only | final commit |
| 6 | Delete this REORG_PLAN.md | trivial | done |

## Rules during execution

1. **Always `git mv`** (never `mv` then `add`) so blame survives.
2. **One commit per logical move/merge** — easy to bisect if something breaks.
3. **Update CLAUDE.md doc paths in same commit as the move** — so the test that checks doc references doesn't fail mid-reorg.
4. **Run the doc-index test (if any) after each commit**.
5. **Keep this file updated** — strike through completed items as we go.

## Open questions for user (resolve before Phase 2)

1. **mega_audit consolidation**: collapse all 7 phase files into one `2026-04-26-mega.md` (~2000 line monolith), or keep as `archives/audits/2026-04-26-mega/{plan,phase1,...,phase6}.md` (8-file subdir)? **Default plan: consolidate into one with `## Phase N` headers.**
2. **Sessions naming**: `archives/sessions/2026-04-26.md` (just date) vs `archives/sessions/2026-04-26-summary.md` (date + suffix)? **Default plan: just date.**
3. **README.md index**: regenerate with full link tree, OR keep terse with subdir pointers? **Default plan: terse pointers + each subdir gets its own README.md.**
