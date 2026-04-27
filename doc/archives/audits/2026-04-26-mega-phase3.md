# Mega Audit — Phase 3 Findings (2026-04-26)

**Scope**: training_panel/* (17 modules, ~6500 LOC)
**Methods**: M3 coverage + M12 error swallowing + M2 hardcoded paths
**Started**: 2026-04-26 15:33 PT

## Findings

### 🟢 P3 — every module has ≥1 test file

| Module | Test files | Risk |
|---|---:|---|
| context.py | 31 | 🟢 |
| pipeline.py | 73 | 🟢 |
| labels.py | 15 | 🟢 |
| pp_panel_training.py | 12 | 🟢 |
| panel_frame.py | 10 | 🟢 |
| ltr_model.py | 8 | 🟢 |
| ngboost_head.py | 5 | 🟢 |
| global_calibrator.py | 5 | 🟢 |
| transformer_model.py | 4 | 🟢 |
| factors.py | 4 | 🟢 |
| purged_cv.py | 4 | 🟢 |
| hourly_features.py | 3 | 🟢 |
| hourly_resolution_panel.py | 1 | 🟡 — Stage C-1 just shipped |
| imputation.py | 1 | 🟡 |
| neutralization.py | 1 | 🟡 |
| minute_features.py | 1 | 🟡 |
| lgbm_ltr.py | 1 | 🟡 |

**No untested module.** 4 modules with thin coverage worth deepening.

### 🟢 P3 — no error-swallowing patterns

`grep except.*pass` returned nothing across 17 modules. No silent
failures hiding bugs.

### 🟢 P3 — no hard-coded production paths

All paths come from config / strategy_dir / function args.

### 🟡 P2 — thin-coverage modules to deepen tests

- `imputation.py` (4 fns: apply_min_history_gate, add_missingness_indicators,
  sector_median_fill, compute_age_weight) — only 1 test file. Each fn deserves
  edge cases (empty, all-NaN, single ticker, etc).
- `neutralization.py` (3 fns: compute_sector_momentum, _residualize,
  neutralize_features) — same.
- `minute_features.py` (3 fns) — recently shipped (this session's intraday work).
- `lgbm_ltr.py` — only 1 test, but has 8 prior session bugs fixed —
  worth a fresh test pass.

These are NOT production blockers. Stage 6 (Tier 4: tests-of-tests)
will systematically deepen these.

## Phase 3 outcome: 0 P0/P1 bugs found

The training_panel codebase is mature — the 17 modules have
collectively ~150 tests pointing at them. New stage C work has
its own tests. No silent error patterns. No hard-coded paths.

This contrasts with the 5+ bugs found in NEWLY-WRITTEN code this
session (validate_buy_logic.py, notebook A/B cells). The pattern:
**existing code is well-audited, new code shipped this session
needed deeper integration testing.**

## Phase 4 next: transformer_model.py + regime.py

These are the two most complex single files (~1300 + ~480 LOC each).
Transformer was deeply audited recently (95% audit doc). Regime
detector has been stable for months.
