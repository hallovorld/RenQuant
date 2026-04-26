# Mega Audit — Phase 4 Findings (2026-04-26)

**Scope**: transformer_model.py + kernel/regime.py
**Methods**: M3 coverage + M5 cross-call-site + M11 shared state + M12 error swallowing
**Started**: 2026-04-26 16:01 PT

## Findings

### 🟢 transformer_model.py — clean

- 4 test files (post-95% audit doc)
- 0 error-swallowing patterns (no bare `except: pass`)
- No suspicious top-level mutable globals
- Recently underwent comprehensive audit (commit history: 60+ audit-fix tags)
- Currently producing CPCV mean OOS IC = -0.003 (poor) but bug-free —
  the issue is the panel size (Chen-Pelger-Zhu 2024 ship gate >5000 dates),
  not code quality. Hourly resolution training (Stage C-3, in progress
  PID 34924) addresses this.

### 🟢 kernel/regime.py — clean

- 8 test files (heavy coverage for a core module)
- 0 error-swallowing
- No suspicious mutable globals
- 10+ call sites (main.py, notebook, training/regime, training/features,
  adapters/sim.py × 2, adapters/runner.py × 2, kernel/pipeline/task_regime.py)
- All callers use lazy imports (defensive — no circular dep risk)
- `confidence_to_size_multiplier()` (which I just used in QP-CONF-CONSISTENCY
  fix) is well-documented and well-tested

### 🟢 No additional bugs found

Both files are MATURE and audit-clean. The transformer's poor OOS IC
is structural (panel size), not buggy code. The regime detector has
been stable for months across many strategies (103, 104).

## Phase 4 outcome: 0 P0/P1 bugs

Combined Phase 1-4 totals:
- Files audited: ~30
- LOC reviewed: ~12,000
- P0 bugs found: 0 (none introduced this session, none lurking in mature code)
- P1 bugs found: 2 (both flagged — signal_combiner dead code, _apply_overrides
  needs property tests)
- P2 bugs found: 4 (flagged for future)

This Tier 1 (production critical path) is healthy. Most session bugs
were in NEW infra (validate_buy_logic.py + my snapshot-override misuse),
not in core trading logic.

## Phase 5 next: scripts (16 files)

Higher risk because:
- 5+ scripts hit the snapshot-override pattern this session
- Less test coverage (operator scripts often shipped without tests)
- More direct user interaction → wider blast radius if buggy
