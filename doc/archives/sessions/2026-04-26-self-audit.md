# Session Self-Audit (2026-04-26 PT 14:10)

**Trigger**: user — "you need self deep audit". Writing this BEFORE
relaunching sims, so if anyone (including future me) is wondering
where the bugs are likely lurking, this is the answer.

## What I shipped this session — risk-graded

| File / commit | Lines | Unit tested | Integration tested | Production tested | Risk |
|---|---:|---|---|---|---|
| `kernel/portfolio_qp/qp_solver.py` (Stages 0-7) | ~250 | ✅ 29 | ✅ 12 | ❌ | 🟡 medium |
| `kernel/portfolio_qp/task_joint_qp.py` | ~230 | ✅ 15 | ✅ 1 | ❌ | 🟠 high |
| `kernel/portfolio_qp/signal_combiner.py` | ~100 | ✅ 12 | ❌ | ❌ | 🟢 low (not wired into pipeline) |
| `kernel/panel_pipeline/task_quality_floor.py` (3 gates) | ~200 | ✅ 40 | ❌ direct sim | ❌ | 🟠 high |
| `kernel/intraday_wash.py` | ~180 | ✅ 20 | ❌ | ❌ | 🟢 low (functions, not pipeline) |
| `training_panel/hourly_resolution_panel.py` (Stage C-1) | ~170 | ✅ 15 | ❌ | ❌ | 🟢 low (not yet wired) |
| `kernel/persistence.py` (`record_ticker_daily_state`) | ~70 | ✅ 10 | ✅ 5 source | ❌ | 🟢 low (additive) |
| `adapters/runner.py` (TDS writer wiring) | ~80 | ❌ | ✅ 5 source | ❌ | 🟠 high — runtime path |
| `scripts/validate_buy_logic.py` | ~302 | ❌ | ❌ | 🔴 **CRASHED** | 🔴 critical |
| `scripts/run_validation_matrix.sh` | ~95 | ❌ | ❌ | ❌ | 🟡 medium |
| `scripts/monitor_training_resources.py` | ~150 | ❌ | ❌ | ❌ | 🟢 low |
| `scripts/plot_training_resources.py` | ~530 | ❌ | ❌ chart-rendered manually | ❌ | 🟢 low |
| `backtesting/.../strategy_config.json` (production turn-on) | flag flips | n/a | ❌ | ⏳ today 1:55 PM PT first run | 🔴 high |

## Bugs caught LATE this session (confidence-reduces)

These are bugs unit tests couldn't catch — only triggered at
runtime / sim / production:

1. **CACHE-DIR-SNAPSHOT** (commit `e63bac5`) — sim runs silently loaded
   0 fundamentals. Lurked because no test covered the snapshot path.
2. **QP-REGIME-STATE-DUCK** (commit `d4ed1a4`) — sim crashed at first
   bar because `regime_state.get()` failed on dataclass. Lurked because
   unit tests used dict ctx; only sim used real RegimeState.
3. **VALIDATE-BUYS-CALL** (just now) — `result.buys()` called as method
   on a list (it's a `@property`). Lurked because `validate_buy_logic.py`
   has zero tests.

**Pattern**: my unit tests use dict / mock objects; production uses
dataclasses + real adapter state. **Three bugs in one session caused
by this gap.**

## Bugs LIKELY still lurking (top suspicions)

### 🔴 P0 — `validate_buy_logic.py` has more landmines

I just fixed `result.buys()` but never sanity-tested the rest of
`_summarise()`. Other suspects:
- `equity_df["portfolio"]` — does this column always exist?
- `result.equity_df.pct_change()` — what if `equity_df` is empty?
- `_apply_overrides` does `cfg.setdefault(...).setdefault(...)` —
  works on dicts, but if config has a list at any level → `.setdefault()`
  raises. Strategy config is all dicts, but worth verifying.
- `_diff_table` formats with `f"{cv:+.4f}"` — fails if cv is a string.

Fix: add a unit test for `_summarise()` with a fake SimResult.

### 🟠 P1 — JointPortfolioQPTask not real-sim verified

The task ran in unit tests + my smoke `test_qp_integration.py`. But:
- `ctx.last_sell_dates` — what's its type in real ctx? My code does
  `last_sells.get(t)` → if it's a `dict`, fine. If it's `set` or
  `list`, crash. Need to grep adapters/runner.py + sim to see what
  type goes in.
- `ctx.prices` — confirmed dict in both unit + integration tests, OK.
- `ctx.holdings[t].shares` — Holdings is a dataclass; `getattr(hs,
  "shares", 0.0)` is safe.
- `ctx.exits.append((t, sig))` — exits is a list. Safe.
- The sim adapter populates `ctx.regime_state` as a `RegimeState`
  dataclass — duck-typed in QP-REGIME-STATE-DUCK fix. ✅

### 🟠 P1 — QualityFloorTask has not run in real sim path

Same risk pattern. The integration test (`test_qp_integration.py`)
uses real `InferenceContext` but a small synthetic ctx. Hasn't run
through full SimAdapter once.

Specifically untested in real sim:
- `ctx._db` for Gate A — sim ctx has `_db` set (sim_runs.db); never
  exercised it.
- `ctx.candidates` ordering — Gate B/C iterate; if cand order matters
  to other downstream tasks, my filter could have subtle effects.

### 🟠 P1 — Production config flip is untested in live

Today's 1:55 PM PT daily_104 will be the FIRST live run with QP
solver + Gate B. If anything about the live runner ctx differs from
sim ctx (e.g. `live_state.json` shape vs sim's persistence), it could
crash production.

Mitigation: ops doc has explicit rollback procedures (30-second CLI
edit). But a crash mid-bar would still lose that bar's decision
quality.

### 🟡 P2 — Hourly resolution panel: no real-data test

`build_hourly_resolution_panel` only tested with synthetic
`_hourly_bars()`. Real `HourlyBarStore.load(symbol)` data has:
- Different timezone (NY market time vs UTC)
- Possible DST transitions
- Sessions with <7 bars (early close, like Black Friday half-day)
- Tickers with sparse history
- 18 tickers (out of 101) MISSING entirely

When wired into Stage C-2, these real-world quirks could trip.

### 🟢 P3 — Notebook updates: not run end-to-end yet

Cells 14, 15, 16 are syntactically clean per Python parse, but the
A/B comparison cell (16) accesses `panel_sim.attribute` — needs to
verify the SimResult attributes I'm reading are real (just fixed for
validate_buy_logic; same risk).

## Action queue

### NOW (this interval) — fix validate_buy_logic.py issues

1. Test `_summarise` with empty SimResult (no equity_df) — does it
   handle gracefully?
2. Re-run smoke sim with fix → confirm summary file lands on disk
3. Then relaunch the matrix

### Next 30 min

4. Audit `task_joint_qp.py` for `last_sell_dates` type assumptions
5. Check if `live_state.json` shape differs from sim's regime_state
6. Sanity-check production config drift (golden + live both have new
   flags)

### Sunday next week

7. Sunday sweep with `panel_ltr.training_resolution: hourly` once
   Stage C-2 lands
8. Live run trace audit — verify Gate B fires + QP solves without
   crash on first daily_104

## Lessons for next session

- **Unit tests with mock dicts are not enough** for tasks that consume
  real adapter ctx. Always include at least one integration test that
  builds a real `InferenceContext` (or close enough to it). 3 bugs
  this session would have been caught by integration tests.
- **Operator scripts need tests too.** `validate_buy_logic.py` had
  zero tests and shipped a method-vs-property bug that would have
  been caught by literal one-line test.
- **Defer flipping production flags until ≥1 real sim run completed.**
  I flipped flags BEFORE seeing sim verdict. If validate_buy_logic
  hadn't crashed at _summarise, I would have had data. If a live run
  crashes today at 1:55 PT, I have ops-doc rollback but that's after
  the bad bar.
