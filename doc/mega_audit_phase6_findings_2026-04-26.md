# Mega Audit — Phase 6 Findings (2026-04-26)

**Scope**: configs + docs + tests-of-tests
**Methods**: M4 type-vs-runtime divergence + M7 doc consistency + M8 config drift + M14 security
**Started**: 2026-04-26 16:16 PT

## Findings

### 🟡 P2 — Config drift (just synced)

`strategy_config.json` diverged from `strategy_config.golden.json` on
two ranking metadata keys: `blend_updated` (2026-04-26 vs 04-24) and
`blend_n_symbols` (101 vs 99). These auto-update when
`recalibrate_scores.py` runs (which fired today). The drift guard
pre-commit hook fires at MY commits but doesn't auto-sync these
ambient updates.

**Just fixed this commit** — synced golden to live values.

**Long-term fix**: extend `recalibrate_scores.py` to ALSO update
`strategy_config.golden.json` so the two files never drift on these
auto-updated keys.

### 🟠 P1 — 43 of 59 doc/ files NOT indexed in CLAUDE.md

```
doc/ files total: 59
Linked from CLAUDE.md: 16 (canonical)
                       31 (any mention)
Not indexed:           43
```

CLAUDE.md §Documentation Index lists 16 docs as canonical; the other
43 (including 9 docs SHIPPED THIS SESSION: `db_design...`, `ops_runbook`,
`sim_ab_results`, 5 mega_audit phase docs, `alpaca_crypto_btc_feasibility`,
`buy_logic_redesign`, `unified_portfolio_action_design`, `calibrator_saturation`,
`session_self_audit`, `transformer_hourly_stage_c2_design`,
`mega_audit_plan`) are not linked. Future agents won't know to read them.

**Long-term fix**: weekly cron that audits doc/ vs CLAUDE.md and
updates the index, OR enforce in pre-commit.

### 🟠 P1 — 18 test files use mock dict/dataclass patterns

This is the pattern that bit us 3× this session (mock dict in tests
vs RegimeState dataclass in prod). Audit each of the 18 to verify
the mock faithfully mirrors production type at the access level
the tested code uses.

**Won't fix this commit** — Tier-4 work, queued for next session.

### 🟢 P3 — No hardcoded secrets / API keys

All Alpaca creds via env vars. No `api_key="..."` literals.

### 🟢 P3 — config drift guard works

`pre-commit hook checks drift for strategy=renquant_104 ✓` —
caught + reported on every commit today. This is the kind of
automation that prevents Phase 6 issues #1.

## Phase 6 outcome

- **0 P0 bugs**
- **2 P1 bugs** (doc index drift + 18 mock-dict test files)
- **1 P2 bug** (config drift on ambient metadata, just synced)

## ALL 6 PHASES COMPLETE

Combined audit totals (Phases 1-6):
- Files audited: ~70
- LOC reviewed: ~30,000+
- **P0 bugs found: 0** (no immediate production bugs)
- **P1 bugs found: 5** (signal_combiner dead, _apply_overrides property test,
  28 untested scripts, 43 unindexed docs, 18 mock-pattern tests)
- **P2 bugs found: 7** (HARDCODED-PYTHON, doc rot, ambient config drift, etc.)
- **Real bugs FIXED this session: 19** (most caught from runtime/sim, not audit)

## Conclusion

**Production code is HEALTHY.** No P0 bugs found across 30k+ LOC
of audited code. The 19 bugs I shipped fixes for today were caught
via runtime / sim execution, not via systematic audit — this is
because the audit found NO production-impacting bugs in mature code.

The actual risk surface is **NEW operator scripts** (5 had bugs this
session) and **infra patterns** (mock-dict tests, snapshot override).

**Recommendation**: invest in operator-script test coverage as the
single highest-ROI improvement. CLAUDE.md §2 already mandates "every
feature gets a test" — extend that to "every script too".
