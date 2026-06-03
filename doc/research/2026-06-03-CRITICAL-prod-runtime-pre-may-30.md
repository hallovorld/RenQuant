# 🔴 CRITICAL — Production runtime is pre-2026-05-30; QP architecture refactor is NOT in prod

**Date**: 2026-06-03
**Status**: ACTIVE — operational concern surfaced by PR #143 drift audit.
**Severity**: P0 — silent functional drift; not a crash, but the prod
behavior diverges from what the config + tests describe.
**Author**: Claude
**References**:
- PR #143 (drift audit) — root finding
- `memory/project_subrepo_runtime_vendored_snapshot_2026-06-01.md` — vendored-runtime mechanism

## TL;DR

The production daily runner reads QP code from
`.subrepo_runtime/repos/renquant-pipeline/`, which is a **vendored
snapshot** that has NOT been refreshed since before 2026-05-30. That
snapshot's `tasks.py` is **3,419 LOC vs subrepo-main's 3,671 LOC**
and predates the entire 2026-05-30+ QP-architecture refactor — including:

1. **`davis_norman.py`** module (added 2026-05-30 commit `87773e6`).
2. **`proportional_trade.py`** module (added 2026-05-30 commit `bfc08b9`).
3. **PR #123 v4** — hard-cap snapshot fix (`_qp_w_upper_hard`) merged 2026-06-03.
4. **PR #126** — `ConstraintSnapshot` contract merged 2026-06-03.
5. **PR #127** — `solve_portfolio_qp_from_snapshot` wrapper merged 2026-06-03.
6. **PR #129** — `BuildConstraintSnapshotTask` wired into Job merged 2026-06-03.

**Practical effect**: production today is running pre-2026-05-30
portfolio_qp code. The `qp_band_method='davis_norman'` config flag in
`strategy_config.{golden,prod,shadow}.json` is set but **the runtime
code does not read that branch** — it falls through to the legacy
no-trade band. The `_qp_w_upper_hard` separation that closed the
`#123` bug class IS NOT in the deployed runtime; if the runtime ever
hits the over-cap holding + low-conviction case, it would re-surface
the original bug.

This has been the deployed state for **at least 4 days without a
crash** — silently, because none of the missing-code paths fired
under typical bar conditions.

## Why this happens

Per `memory/project_subrepo_runtime_vendored_snapshot_2026-06-01.md`:

> Umbrella's subrepo delegate uses `.subrepo_runtime/repos/`
> vendored snapshot, NOT live sibling clone; `make subrepo-runtime-root`
> refreshes after subrepo merges.

The mechanism is correct — `make subrepo-runtime-root` is the
documented refresh path. It just hasn't been invoked since before
2026-05-30. Since then:
- 6+ portfolio_qp PRs have merged in renquant-pipeline (subrepo) — #19, #20, #21, #22, #26, #27 (Step 1e mirror).
- `davis_norman.py` + `proportional_trade.py` are NOT YET in the
  subrepo at all (drift audit P0 findings #1 + #2; subrepo mirror PR
  in flight per Task #79).
- Even after that mirror lands, the vendored runtime needs
  `make subrepo-runtime-root` to pick up everything.

## Why prod has not crashed

The missing-code paths fail-soft, not fail-hard:

1. **`qp_band_method='davis_norman'`** — the runtime's pre-2026-05-30
   code does not have an `if method == 'davis_norman'` branch. It
   silently uses the default ad-hoc 5% band.
2. **`_qp_w_upper_hard`** — pre-2026-05-30 code never reads this
   attribute, so the absence is invisible. The cap-compliance retry
   path keys off `infeasible` status which still fires correctly via
   the legacy w_upper.
3. **`ConstraintSnapshot`** — pre-refactor consumers use kwargs
   directly. The contract just isn't exercised.

So the runtime is functionally "older, weaker, but stable". Not
catastrophic, but DEFINITELY not what the test suite + the config
imply is running.

## Sequencing the fix

**Must happen in this order** to avoid pushing a partially-applied
refactor to prod:

1. ✅ Land `davis_norman.py` + `proportional_trade.py` mirror to
   subrepo (Task #79 — subagent in flight as of 2026-06-03).
2. ✅ Land all currently-open subrepo portfolio_qp mirror PRs (#23, #24,
   #26, plus the davis_norman/proportional_trade PR from step 1).
3. ⏳ Verify subrepo-main `tests/test_no_bare_kernel_imports.py` is
   green AND the test suite passes against the subrepo's HEAD.
4. ⏳ Run `make subrepo-runtime-root` from the umbrella to refresh the
   vendored snapshot to subrepo-main HEAD.
5. ⏳ Run a `daily_104.sh` smoke through paper / shadow broker BEFORE
   re-enabling prod cron. Verify:
   - `davis_norman` code path actually fires (look for "Davis-Norman
     no-trade band" log line, or an explicit count in `_qp_diagnostics`).
   - `_qp_w_upper_hard` is stamped on the snapshot.
   - `BuildConstraintSnapshotTask` short-circuits cleanly on a
     synthetic over-cap fixture (the integration test in
     `tests/test_build_constraint_snapshot_task.py::test_failure_path_stamps_qp_attribution_fields`).
6. ⏳ Only then, re-enable the prod daily cron (or let the next
   natural run take it).

## Rollback path

The vendored snapshot is just a git checkout of subrepo-main. If
step 4 produces a broken state, `cd .subrepo_runtime/repos/renquant-pipeline
&& git checkout <pre-refresh-sha>` reverts the snapshot atomically;
prod resumes the old behavior on the next bar. **No data corruption
risk** — the QP runs are stateless.

## Recommendation

1. **Don't refresh the snapshot today.** Subagent (Task #79) is still
   creating the davis_norman + proportional_trade mirror PR. Wait
   for it to merge + the broader subrepo mirror queue (#23, #24, #26)
   to clear.
2. **Schedule the refresh as the FIRST step of the 2026-06-04 daily
   cycle**, after a paper-broker smoke run confirms the new code
   paths fire as expected.
3. **Add a runtime sanity check** to the daily cycle (follow-up PR):
   verify `kernel.portfolio_qp.davis_norman` is importable and that
   `_qp_diagnostics["band_method"]` is "davis_norman" on at least one
   bar per day. Fail loud if not — catches future drift.
4. **Long-term**: tighten the umbrella `Makefile` so that
   `make subrepo-runtime-root` runs as part of CI on any
   `renquant-pipeline` merge to main. The current manual cadence is
   the source of the drift.

## Why this memo exists

The drift audit (PR #143) catalogued the finding. This memo escalates
it so the prod operator (you) sees it in the daily review queue
rather than buried in an audit doc. The §8 plan A/B replay assumes
the refactor IS in prod; until step 4 above runs, that assumption is
false.

## Open question for codex / user

Do we want CI to BLOCK on stale vendored snapshots (force-refresh on
every subrepo merge), or KEEP the manual cadence with a louder daily
warning? Current "manual but documented" lost us 4 days; either
option fixes it.
