# Live-state save NameError (`config` → `self._config`) + regression guard

2026-06-26. Severity: P0 (live trading). Customer impact: intraday cron
fail-closed for ~1.5h of the session (18 failed runs) before recovery; no bad
orders, no capital impact.

## Symptom
The `renquant_104` intraday-sell cron fail-closed every 12 minutes on the hard
preflight `P-BROKER-FILL-FRESHNESS`:

> no runner-driven activity in 21 trading days (hard cap 20); last_activity=2026-05-27.
> Strategy is dormant.

All 18 other preflight checks passed (config-fp, broker connect, holdings load).
The broker, however, showed real runner-driven fills on 2026-06-24 and
2026-06-25 — the strategy was **not** dormant. `last_activity_date` was stale.

## Root cause
Commit `64f6b46` (2026-06-16, *"wire config flag through save path"*) changed
`RunnerAdapter.commit()`'s live-state save to:

```python
save_live_state_atomic(state_file, self._state, config)   # `config` is undefined here
```

`config` is not a local in `commit()` — every other reference in the method uses
`self._config`. The third argument is evaluated at the call site, so this raises
`NameError: name 'config' is not defined` on **every** commit, *after* orders are
already placed at the broker. The state file (`live_state.alpaca.json`) was
therefore never rewritten: `monitor_state.last_activity_date` froze at its last
good value (2026-05-27). ~20 trading days later that crossed the dormancy cap and
`P-BROKER-FILL-FRESHNESS` started fail-closing the cron.

The bug shipped to `main` and ran in production undetected for ~10 days because
**no test exercises `commit()`'s save section**: the only existing `commit()`
test uses `SimAdapter` (a different code path), and any test that *did* reach the
save line on the buggy code would itself `NameError` on the argument before
`save_live_state_atomic` ran.

`config` is an optional arg of `save_live_state_atomic` (it only feeds the
opt-in, default-OFF `RQ_LIVE_STATE_V2` typed-state flag and is byte-identical
output otherwise), so the correct, intent-preserving fix is `self._config`.

## Fix
- `backtesting/renquant_104/adapters/runner.py` — `config` → `self._config` in
  the `save_live_state_atomic(...)` call (one line).
- `tests/test_runner_commit_save_state_config_arg.py` — AST invariant guard
  pinning that `commit()` passes `self._config` (an attribute access), never a
  bare `config`, to `save_live_state_atomic`. Verified: PASS on the fix, FAIL on
  the reintroduced bug. AST is used deliberately — `commit()` reads ~all of
  self/ctx, so a behavioral test would need a full live context; the invariant
  that broke is exactly "the save call's config arg must be `self._config`".

## Verification (live)
The fix was hot-applied to the live tree and confirmed on the running cron:
8 consecutive intraday runs (10:00–11:24 PT) `finished` cleanly, `State saved →
live_state.alpaca.json` with no NameError, `P-BROKER-FILL-FRESHNESS` green
(`last activity 2026-06-25, 1 trading day ago`), and holdings reconciled from the
broker (MU/CSCO/CRWD/PANW/AMZN). This PR lands the same fix on `main` so a fresh
deploy / runtime sync no longer ships the NameError.

## Note on discovery
The frozen state + the lost in-tree hotfix were surfaced while recovering from
the 2026-06-25 live-tree `git reset --hard` incident: that reset reverted the
uncommitted working-tree hotfix (`self._config`) back to the committed-buggy
`config`, which is why live saves had been succeeding 06-17/24/25 but then began
to crash. Committing the fix here removes the dependency on an uncommitted
working-tree patch.
