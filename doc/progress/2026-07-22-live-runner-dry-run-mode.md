# Progress — live.runner explicit dry-run mode (GOAL-5 AC5 dawn preflight, r2)

**Date:** 2026-07-22
**Goal:** GOAL-5 P0 AC5 (month-1) — companion to `renquant-orchestrator`#565.
**Type:** bug fix + regression tests. No production path touched (additive,
default-off flag; existing call sites unchanged).

## STATUS:
Implemented and tested here. Not yet consumed — `renquant-orchestrator`#565
(the dawn-preflight shell script + analyzer) is the consumer and needs this
merged first; that PR will be updated to pass `--dry-run` once this lands.

## WHAT:
`live/runner.py` gains an explicit `--dry-run` CLI flag, threaded through
`run_once_multi()` → `_run_once_multi_pipeline(..., dry_run: bool = False)`:
- Runs the real scoring/selection funnel (preflight checks, `adapter.make_context()`,
  `pipeline.run(ctx)`) exactly as a normal cycle does.
- **Never calls `adapter.commit(ctx)`** — the one place that places broker
  orders, writes `live_state.json`, and persists run-bundle records. `commit()`
  itself (`backtesting/renquant_104/adapters/runner.py`, ~1000 lines of
  audited order/state logic) is untouched by this change.
- Logs a greppable `DRY_RUN_ATTESTATION commit=skipped ...` line and sends a
  distinctly-labeled `[DRY-RUN PREFLIGHT]` ntfy via a new `_notify_dry_run_probe()`
  — separate from `_notify_decision()`, which reads `ctx.orders_placed` /
  `ctx.exits_placed`, fields only populated inside `commit()`. Reusing
  `_notify_decision` in dry-run mode would have silently reported every probe
  cycle as a real "0 orders" decision instead of "no decision was applied."

## WHY/DIR:
`renquant-orchestrator`#565 fixed the dawn preflight's `ModuleNotFoundError`
(a `cd` before `-m live.runner`) but codex's re-review correctly rejected it:
`--broker readonly-alpaca` only constrains **broker** access — `commit()`
still writes `live_state.json`, allocates run IDs, updates artifact/status
records, and notifies, regardless of broker choice. `cd` is not side-effect-free.
A working import is necessary but not sufficient for a read-only operational
probe.

Design choice: skip `commit()` entirely rather than adding a write-suppression
flag *inside* `commit()`. That function is the live order-execution path with
~2 years of audited edge-case fixes (tax lots, fractional shares, partial
exits, NaN guards); threading a dry-run branch through it would touch the
highest-risk code in the repo for a feature that doesn't need to be in there
at all — the funnel is already cleanly split into "compute" (`pipeline.run`,
read-only) and "persist+notify" (`commit` + `_notify_decision`). Skipping the
second half at the call site in `live/runner.py` is a small, additive,
default-off change with zero new surface inside `commit()`.
`dry_run` is orthogonal to `--broker` (works with any broker) so the two are
defense-in-depth, not either/or, matching codex's framing ("readonly-alpaca
only constrains broker access").

## EVIDENCE:
Ran the new + existing runner test suites (`--strategy renquant_104` code
paths) via the shared venv:
- `tests/test_runner_dry_run.py` (new, 4 tests) + `tests/test_runner_preflight_fail_closed.py`
  (existing, unmodified): **8 passed**. `[VERIFIED]`
- `pytest tests/ -k runner` (445 tests total): **442 passed, 2 failed, 2 errors**.
  Confirmed all 4 non-passes are pre-existing and unrelated: reran the same
  2 failing tests against a clean `origin/main` worktree (no changes) — same
  2 failures there (`test_state_store.py` / `test_runner_trade_ntfy.py`, both
  asserting on unrelated source text in `adapters/runner.py` /
  `scripts/live_only_104.sh`, not touched by this PR). The 2 collection errors
  (`test_correlation_guard.py`, `test_per_regime_sigma_wire.py`) are import-time,
  unrelated to `live/runner.py`. `[VERIFIED — pre-fix reproduction on unmodified
  origin/main]`

Key regression covered: `test_dry_run_never_calls_commit_on_any_broker` is
parametrized over `paper` and `alpaca_shadow` broker names specifically to
pin codex's finding that `readonly-alpaca` alone is not sufficient — dry_run
must skip `commit()` regardless of broker.

## NEXT:
- Merge this, then update `renquant-orchestrator`#565:
  `ops/renquant104/dawn_funnel_preflight.sh` adds `--dry-run` to the runner
  invocation; `ops/renquant104/dawn_funnel_analyze.py` requires the
  `DRY_RUN_ATTESTATION` marker in the log (fail-closed if a future regression
  silently drops the flag) — this is the "assert it in an integration test /
  fail closed if the runner does not attest" half of codex's #565 review.
- Deploy to `renquant-orchestrator-run` / sync the live umbrella checkout
  after both land (launchd runs from the live tree, not a pin).
