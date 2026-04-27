# Mega Audit — Phase 2 Findings (2026-04-26)

**Scope**: persistence.py + panel_pipeline tasks
**Methods**: M3 coverage gap + M5 cross-call-site + M9 import graph
**Started**: 2026-04-26 15:26 PT

## Findings

### 🟢 P3 — coverage looks healthy

`record_*` functions in `kernel/persistence.py` test coverage:

| Function | Test files |
|---|---:|
| record_pipeline_run | 6 |
| record_candidate_scores | 4 |
| record_training_run | 5 |
| record_trades | 3 |
| record_live_state_snapshot | 2 |
| record_ticker_daily_state | 2 |
| record_forward_returns | 2 |
| record_portfolio_metrics | 1 |

No record_* function is untested. **P0/P1 risk on persistence: none caught.**

### 🟢 P3 — single canonical caller for record_ticker_daily_state

Only `adapters/runner.py:1007` writes to `ticker_daily_state`. No other
caller. Sim adapter doesn't (sim doesn't need TDS — only live runner
does). Architecturally clean.

### 🟡 P2 — lazy imports in pipeline modules

10 lazy imports across kernel/panel_pipeline + kernel/pipeline:
```
panel_pipeline/job_panel_scoring.py: 3
panel_pipeline/panel_scorer.py: 2
pipeline/pp_inference.py: 1
pipeline/job_joint_actions.py: 1
pipeline/job_universe.py: 1
pipeline/pp_training_full.py: 2
```

Most are deliberate (avoiding circular deps when InferencePipeline
init wants to reference Tasks that import from common modules). One
worth flagging: `job_panel_scoring.py` has 3 lazy imports — the
QualityFloorTask one I added is correct (avoids __init__ circular)
but the other 2 should be checked next pass.

### 🟢 P3 — no record-anti-patterns observed

- No `record_*` function silently swallows errors
- All accept `conn=None` for tests / opt-out scenarios
- Schema migrations gated by IF NOT EXISTS (safe re-run)

## Phase 2 outcome

**No new bugs.** Persistence layer is mature (~12 commits of audit
tags throughout this session show prior care). Panel_pipeline tasks
have appropriate lazy import patterns.

## Phase 3 next

Tier 1 #14-18: intraday_wash + hourly_resolution_panel + pp_panel_training
+ ngboost_head + global_calibrator. These are training-side modules.
Higher risk because:
- Recent commits (this session's Stage A/B/C-1/C-2)
- Wider data dependency surface
