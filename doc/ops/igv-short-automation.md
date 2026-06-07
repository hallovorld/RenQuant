# IGV short-plan automation (98/90 put spread)

Cron-driven monitor that runs the operator's discretionary IGV put-spread plan
as a deterministic state machine and — **only when explicitly armed** — places
the live multi-leg order on Alpaca.

> ⚠️ **Real-money options on a timer.** This overrides the CLAUDE.md §4.1
> paper-cron mandate, and *only* when the operator arms it (see §3). The entry
> triggers are discretionary by nature; review the alerts. Default posture is
> dry-run (alerts, no orders).

## The plan (encoded in `live/igv_short_state.py`)

| IGV action | Plan response |
|---|---|
| bounce $97.5–99, hourly close back < 97.5 (rejection) | **enter** 98/90 put debit spread |
| break < $94.8, then bounce $95–96 rejects (close < 95) | **enter** (path B) |
| reclaim $100 | stand down — do not enter this bounce |
| recover $101.5–102 | **void** the plan |
| in position, $92–93 | take profit: close **half** |
| in position, $88–90 | take profit: close **most** → done |
| in position, ≥ $100.5 | stop: cut **half** |
| in position, daily **close** ≥ $101.5 | stop: **exit all** → done |

"Rejection" = the zone was touched in the recent window AND the latest **closed
hourly bar** closes back below the zone low (operator-chosen rule).

## Architecture

```
launchd (every 15m) ─► scripts/igv_short_monitor.py --once
   ├─ kill-switch + market-clock guard (no-op outside RTH)
   ├─ fetch IGV: hourly bars + latest price + daily close   (alpaca data)
   ├─ live/igv_short_state.step()  ── pure, unit-tested ──► actions
   ├─ persist state (live/state/igv_short_state.json, idempotent)
   ├─ ntfy alert for every action (always)
   └─ live/options_executor  ── ONLY if armed ──► Alpaca multi-leg limit order
```

- **State machine** (`live/igv_short_state.py`) is pure (no I/O); 15 tests in
  `tests/test_igv_short_monitor.py` cover every transition.
- **Executor** (`live/options_executor.py`) resolves the 98/90 puts at the
  nearest weekly expiry from the live chain (no hand-built OCC), submits a
  **limit** multi-leg order, enforces a hard contract cap + debit sanity bound,
  and uses a deterministic `client_order_id` (idempotent — re-runs never
  double-submit).

## Safety gates (ALL required for a live order)

1. `config.mode == "live"`
2. env `IGV_LIVE_ARMED == "1"`
3. no kill-switch: env `IGV_KILL != "1"` **and** no file `live/state/IGV_KILL`

Any gate off ⇒ **dry-run**: state advances and alerts fire, but no order is sent.
Additional always-on rails: limit orders only (never market), `IGV_MAX_CONTRACTS`
hard ceiling (default 5), one-entry (the machine never re-enters), debit bound
`0 < debit < strike width`, auto-void at recovery and terminal stop/target.

## Setup

1. **Enable options** (level 2 / spreads) on the Alpaca account.
2. `cp live/igv_short_plan.example.json live/igv_short_plan.json` and edit:
   set `contracts`, `max_debit`, and `dte_min/max`. Leave `mode: "paper"` to
   start. (`live/igv_short_plan.json` + `live/state/` are gitignored.)
3. Dry-run / paper first:
   ```bash
   set -a; source .env; set +a
   .venv/bin/python scripts/igv_short_monitor.py --once   # paper, alerts only
   ```
4. Install the cron (review the plist first):
   ```bash
   cp ops/launchd/com.renquant.igv-monitor.plist ~/Library/LaunchAgents/
   # edit RENQUANT_NTFY_URL + paths, then:
   launchctl load ~/Library/LaunchAgents/com.renquant.igv-monitor.plist
   ```

## Arming live (deliberate)

```bash
# in live/igv_short_plan.json: set "mode": "live"
# in the plist EnvironmentVariables: set IGV_LIVE_ARMED = 1, reload the agent
```

## Kill switch (instant halt)

```bash
touch live/state/IGV_KILL        # or: export IGV_KILL=1
```
Removes nothing already filled — cancel/close open orders in the broker if needed.

## Operational notes

- The monitor only manages the spread per the table; it does not chase or
  re-enter after a void/close (one-entry by design — start a new plan_id for a
  new setup).
- Post-close stop (daily close ≥ 101.5): add a second launchd entry ~5 min after
  the close with `IGV_IGNORE_HOURS=1`.
- §3.5 follow-up: the options-exec primitive should migrate to
  `renquant-execution`; it lives in umbrella `live/` for v1 alongside
  `alpaca_broker.py`.
