# Mega Audit — All Components, Industry-Leading Quality (2026-04-26)

**User mandate**: largest possible deep audit on ALL code/config/docs.
Multiple methods + tools. Burn tokens. Don't fear refactoring.

**Goal**: Catch every lurking bug, design flaw, untested code path,
config-drift hazard, doc rot, dead code, security risk, performance
trap, race condition, type confusion, before they bite production.

This doc is the MASTER PLAN. Per-component findings live in numbered
follow-on docs (`mega_audit_<component>_2026-04-26.md`).

---

## Audit methods (apply each to every component)

| Method | What it catches |
|---|---|
| **M1 — Static AST scan** | Type confusion, dead code, undefined names |
| **M2 — Grep patterns** | Hard-coded paths, magic numbers, TODO/FIXME, copy-paste sites |
| **M3 — Test coverage gap** | Functions with 0 tests, files with 0 tests |
| **M4 — Type-vs-runtime divergence** | Mock dicts in tests vs real dataclass in prod (this session's recurring pattern) |
| **M5 — Cross-call-site consistency** | Same function called with different conventions in different places |
| **M6 — Edge case enumeration** | NaN/inf/empty/None handling at every level |
| **M7 — Doc-vs-code consistency** | Docstring claims vs actual behavior |
| **M8 — Config schema drift** | Strategy config keys present in code but missing in docs/golden |
| **M9 — Import graph** | Circular imports, dead imports, optional imports without try-except |
| **M10 — Performance traps** | O(n²) loops, redundant work, missing memoization, file I/O in hot path |
| **M11 — Concurrency pitfalls** | Shared state, GIL releases, file locking, DB contention |
| **M12 — Error swallowing** | bare `except`, `except Exception: pass` |
| **M13 — Backward compat** | Public API changes, schema changes, config breakage |
| **M14 — Security** | Path traversal, eval/exec, SQL injection (less applicable here), env var leaks |

---

## Component coverage

### Tier 1 — CRITICAL PATH (used in production right now)

| # | Component | LOC | Methods | Doc owner | Status |
|---|---|---:|---|---|---|
| 1 | `kernel/portfolio_qp/qp_solver.py` | 290 | M1-M14 | A1 | ⏳ |
| 2 | `kernel/portfolio_qp/task_joint_qp.py` | 245 | M1-M14 | A1 | ⏳ |
| 3 | `kernel/portfolio_qp/signal_combiner.py` | 110 | M1-M14 | A1 | ⏳ |
| 4 | `kernel/panel_pipeline/task_quality_floor.py` | 250 | M1-M14 | A2 | ⏳ |
| 5 | `kernel/panel_pipeline/job_panel_scoring.py` | 620 | M1-M14 | A2 | ⏳ |
| 6 | `kernel/persistence.py` | 980 | M1-M14 | A3 | ⏳ |
| 7 | `kernel/pipeline/task_score_distribution.py` | 160 | M1-M14 | A2 | ⏳ |
| 8 | `kernel/pipeline/task_joint_actions.py` | 920 | M1-M14 | A1 | ⏳ |
| 9 | `kernel/pipeline/job_joint_actions.py` | 50 | M1-M14 | A1 | ⏳ |
| 10 | `adapters/sim.py` | 750 | M1-M14 | A4 | ⏳ |
| 11 | `adapters/runner.py` | 970 | M1-M14 | A4 | ⏳ |
| 12 | `live/alpaca_broker.py` | 600 | M1-M14 | A4 | ⏳ |
| 13 | `sim/runner.py` | 250 | M1-M14 | A4 | ⏳ |
| 14 | `kernel/intraday_wash.py` | 180 | M1-M14 | A5 | ⏳ |
| 15 | `training_panel/hourly_resolution_panel.py` | 220 | M1-M14 | A5 | ⏳ |
| 16 | `training_panel/pp_panel_training.py` | 2100 | M1-M14 | A6 | ⏳ |
| 17 | `training_panel/ngboost_head.py` | 360 | M1-M14 | A6 | ⏳ |
| 18 | `training_panel/global_calibrator.py` | 320 | M1-M14 | A6 | ⏳ |
| 19 | `training_panel/transformer_model.py` | 1300 | M1-M14 | A7 | ⏳ |
| 20 | `kernel/regime.py` | 480 | M1-M14 | A8 | ⏳ |

**Tier 1 total**: ~10,495 LOC across 20 files.

### Tier 2 — INFRA / TRAINING (Sunday-only, less critical)

| Component | Status |
|---|---|
| `scripts/validate_buy_logic.py` (just gained 27 tests but more lurking) | ⏳ |
| `scripts/run_validation_matrix.sh` | ⏳ |
| `scripts/sunday_panel_sweep.py` | ⏳ |
| `scripts/train_104.py` | ⏳ |
| `scripts/recalibrate_scores.py` | ⏳ |
| `scripts/fit_panel_calibrator.py` | ⏳ |
| `scripts/monitor_training_resources.py` | ⏳ |
| `scripts/plot_training_resources.py` | ⏳ |
| `scripts/enable_hourly_transformer.py` | ⏳ |
| `scripts/fetch_*.py` (5 scripts) | ⏳ |
| `scripts/ab_harness.py` | ⏳ |
| `scripts/compare_panel_vs_baseline.py` | ⏳ |
| `scripts/backtest_and_analyze.py` | ⏳ |
| `scripts/analyze_backtest.py` | ⏳ |
| `scripts/export_lean_*.py` | ⏳ |
| `daily_104.sh` / `live_only_104.sh` / `intraday_sell_104.sh` | ⏳ |
| `retrain_panel.sh` | ⏳ |

### Tier 3 — CONFIG + DOCS

| Component | Status |
|---|---|
| `strategy_config.json` schema audit | ⏳ |
| `strategy_config.golden.json` drift check | ⏳ |
| `CLAUDE.md` accuracy audit | ⏳ |
| `doc/golden_config_2026-04-23.md` outdated check | ⏳ |
| `doc/database.md` schema drift | ⏳ |
| All 9 session docs (this convo) cross-ref | ⏳ |
| LaunchAgent plists (4 files) | ⏳ |
| `requirements.lock.txt` vulnerability scan | ⏳ |

### Tier 4 — TESTS (audit the auditors)

| Component | Status |
|---|---|
| Test files using mock dicts where prod uses dataclass | ⏳ |
| Tests with `time.sleep` / racey assertions | ⏳ |
| Tests skipping with `@pytest.mark.skip` without justification | ⏳ |
| Coverage gap: which functions have ZERO tests | ⏳ |

---

## Execution order

**Phase 1 (NOW, ~1 hour)**: A1+A4 — portfolio_qp + sim/live adapters.
These touch live trades. Highest risk if buggy.

**Phase 2**: A2+A3 — panel_pipeline tasks + persistence. These shape
decisions; bugs here affect every bar.

**Phase 3**: A5+A6 — intraday_wash + panel training. Stage C-3 prep.

**Phase 4**: A7+A8 — transformer + regime detector. Lower urgency
(transformer not in production; regime stable for months).

**Phase 5**: Tier 2 scripts. Today's session bit us 5× through scripts;
audit them all.

**Phase 6**: Config + docs + tests of tests.

Each phase produces 1 doc with structured findings:
- 🔴 P0: production bug, fix immediately
- 🟠 P1: latent bug, fix this week
- 🟡 P2: cosmetic / future-proofing
- 🟢 P3: noted, no action

---

## Phase 1 audit log (start)

Auditing now. Checkpoint: 2026-04-26 15:21 PT.

Will produce findings doc per component. Expected first commit: <30 min.
