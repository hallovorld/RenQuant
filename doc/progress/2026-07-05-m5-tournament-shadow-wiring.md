# M5 Tournament Shadow Admission -- Umbrella Integration Wiring

**Date:** 2026-07-05
**PR:** (this PR)
**Companion:** renquant-orchestrator PR #395 (tournament shadow admission module)

## What

Wired the orchestrator's `log_shadow_admission()` entry point into
`RunnerAdapter.commit()` so the daily pipeline logs both per-ticker
tournament and panel admission verdicts in parallel.

## Changes

| File | Change |
|------|--------|
| `backtesting/renquant_104/adapters/runner.py` | Added `_build_tournament_shadow_ticker_scores()` helper (module-level) + config-gated call to `log_shadow_admission()` after decision_ledger write |
| `backtesting/renquant_104/strategy_config.json` | Added `tournament_shadow: {enabled: false}` config section |
| `tests/test_tournament_shadow_wiring.py` | 11 tests: ticker_scores builder + config gate + fail-open + lazy import |

## Safety

- Default OFF (`tournament_shadow.enabled: false`)
- Fail-open: try/except wraps the entire block; logged warning, never raised
- Lazy import: `renquant_orchestrator.tournament_shadow_admission` imported
  only inside the enabled branch
- No production data writes; output goes to `data/shadow/` (non-canonical)
- No orders, no model changes
