# Mega Audit — Phase 1 Findings (2026-04-26)

**Scope**: Tier 1 #1-3 (`kernel/portfolio_qp/*.py`) + Tier 1 #10-13 (adapters + sim runner)
**Methods applied**: M1 AST + M2 grep + M5 cross-call-site + M10 perf + M11 concurrency + M12 error-swallowing
**Started**: 2026-04-26 15:24 PT

## Findings (rank-ordered)

### 🔴 P0 — none in this batch

### 🟠 P1

#### P1-1: `signal_combiner.combine_signals` is DEAD CODE
- **Location**: `backtesting/renquant_104/kernel/portfolio_qp/signal_combiner.py`
- **Evidence**: `grep -rn combine_signals` returns only the module's own def + log + 12 tests. NO production caller.
- **Risk**: Confusion when reading the codebase. Ops doc claims "Phase 6: Treynor-Black signal combiner" but the call site to wire it doesn't exist.
- **Fix**: Either (a) wire it into PanelScoringJob with a config flag, or (b) move to `archive/`. Recommend (a) since it's the entry-point for Stage 6 of the unified portfolio QP roll-out.
- **Won't fix this commit** — needs sim verification first per CLAUDE.md §2a.

#### P1-2: `_apply_overrides` deepcopy thoroughness
- **Location**: `scripts/validate_buy_logic.py::_apply_overrides`
- **Pattern**: After this session's bugs, the function is correct, but it has 11 mutations across nested dicts. A single missed `setdefault()` chain would re-introduce the silent-override class. Recommend a unit test that runs `_apply_overrides` end-to-end on the FULL production config and asserts no key is unexpectedly missing/added.
- **Fix**: Add property-based tests that compare keys-in vs keys-out.

### 🟡 P2

#### P2-1: SimAdapter DB injection — confirmed CORRECT
- **Audit**: I just shipped DB-PATH-WRONG-KEY fix; verified it's correct.
  - `adapters/sim.py:152` calls `get_connection(config, role="sim")`
  - `kernel/persistence.py:381`: when `role="sim"`, reads `persistence.sim_db_path`
  - My fix sets that key → per-sim DB is honored ✓

#### P2-2: No hard-coded paths in production critical path
- M2 grep on adapters/sim.py + adapters/runner.py + live/alpaca_broker.py: clean.
- Paths come from config or `_strategy_dir`.

#### P2-3: No error-swallowing patterns in adapters
- M12 grep on adapters: clean. (`except` blocks all do logging or fallback).

### 🟢 P3

- portfolio_qp module: clean static AST, no syntax issues
- Cross-call-site for solve_portfolio_qp / JointPortfolioQPTask: consistent
- No GIL-release / concurrency hazards in qp_solver (single-process scipy.minimize)
- O(n) loops only in task_joint_qp (4 iterations of 1 loop each, no nesting)

## Patterns observed (this audit + this session)

These are **systemic** — apply across the whole codebase:

1. **Mock-dict tests vs dataclass production** — already bit 3× this session. Need a session-wide audit (Phase 2-4) to find every test fixture that could diverge from prod.
2. **Snapshot context overriding in-memory mutations** — already bit 4× (validate.py + notebook + ab_harness + compare_panel). Already added SNAPSHOT-OVERRIDE-WARN guardrail. **Phase 2** should grep ALL run_backtest call sites.
3. **Wrong-key config injection** — DB-PATH-WRONG-KEY. Suggests we should consolidate config-key constants into `kernel.config` so it's a typo-error-not-runtime-bug.

## Next phases

- Phase 2: panel_pipeline tasks + persistence
- Phase 3: hourly resolution panel + intraday wash + training pipeline
- Phase 4: transformer + regime detector
- Phase 5: scripts (16 files)
- Phase 6: configs + docs + tests-of-tests

Estimated total wall time for full mega-audit: 4-6 hours of focused work.
This commit ships Phase 1 findings; next interval continues.
