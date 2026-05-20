# Insider trades setup — SEC EDGAR User-Agent

## What

`kernel/insider_trades.py` reads SEC EDGAR (`data.sec.gov/...`) to populate
the `insider_net_buy_90d` feature column in the panel. SEC blocks
generic User-Agents — a UA must include a name and a contact email.

The module reads `RENQUANT_SEC_UA` from the environment. If unset, it
falls back to a generic string that **SEC currently rejects with HTTP 403**.

## Why this matters

`insider_net_buy_90d` is a documented alpha factor (Lakonishok-Lee 2001,
Cohen-Malloy-Pomorski 2012). When the env var is unset:
- All SEC fetches return 403 silently.
- `LoadInsiderTradesTask` reports `0 / N tickers with insider rows`.
- Panel feature `insider_net_buy_90d` is all-NaN.
- XGBoost trees never split on it; effective signal contribution = 0.

Symptom: training logs show
`LoadInsiderTradesTask: 0 / 142 tickers with insider rows` even though
the panel config has the feature enabled. (Universe is now wl200 — 142 tickers
post-2026-05-18 promotion; pre-promotion was wl103.)

**Note (2026-05-18)**: Insider features (E22) were REJECTED via retest with
placebo persistence 100-142%. The infrastructure stays wired for future
re-evaluation but `panel_ltr.insider_features` may be disabled in golden.
Refresh cron still useful for the cache if re-enabled.

## One-time setup (interactive shells)

Add to `~/.zshrc` (or `~/.zshenv` for non-interactive):

```bash
export RENQUANT_SEC_UA="YourName your_email@example.com"
```

Reload: `source ~/.zshrc`. Verify: `python -c "import os; print(os.environ.get('RENQUANT_SEC_UA'))"`.

## Production launchd setup (cron-style retrain)

`launchd` does NOT inherit shell env vars by default. Each plist that
spawns a Python process touching insider data must declare the var
in `EnvironmentVariables`.

**Affected plists** (any plist whose Python process touches `kernel/insider_trades.py`):
- `~/Library/LaunchAgents/com.renquant.daily104.plist` — daily refresh
- `~/Library/LaunchAgents/com.renquant.weekly-wf-promote.plist` — Saturday 04:00 PT weekly walk-forward promote (replaced Sunday retrain-panel104 post-2026-05-17 walk-forward gate enforcement)
- `~/Library/LaunchAgents/com.renquant.retrain-panel104.plist` — daily STAGES retrain (NOTE: NOT checked into repo `scripts/launchd/`; must be installed manually)

For each plist, ensure the `EnvironmentVariables` block contains:

```xml
<key>EnvironmentVariables</key>
<dict>
    <key>PYTHONPATH</key>
    <string>/Users/renhao/git/github/RenQuant</string>
    <key>RENQUANT_SEC_UA</key>
    <string>YourName your_email@example.com</string>
</dict>
```

**Apply**: edit the plist with the UA value, then reload:

```bash
launchctl unload ~/Library/LaunchAgents/com.renquant.retrain-panel104.plist
launchctl load   ~/Library/LaunchAgents/com.renquant.retrain-panel104.plist
```

**Verify**: trigger a manual run and tail the log for the
`LoadInsiderTradesTask: N / N tickers with insider rows` line —
the count should match the watchlist size.

## Cold backfill (first-time fetch)

The default 240s budget in `fetch_insider_trades_watchlist` is sized
for daily incremental refresh — a cold backfill of 100+ tickers needs
60-90 min wall time at SEC's 8 req/sec rate cap.

```bash
export RENQUANT_SEC_UA="YourName your_email@example.com"
python scripts/fetch_insider_trades.py \
  --strategy renquant_104 \
  --total-budget-sec 5400 \
  --per-ticker-sec 60
```

Wallclock: ~85-125 min for wl200 142 tickers fresh (scales linearly from
prior 60-90 min for 103 tickers). Subsequent daily refreshes run in <30 s
because the cache only updates filings newer than 7 days.

## SEC rate limit reference

- SEC EDGAR rate cap: 10 req/sec (per IP). We self-throttle at 8 req/sec
  in `kernel/insider_trades.py::_MIN_SLEEP_S = 0.125`.
- 403 responses on missing/generic UA are **immediate** — no exponential
  backoff fixes them. Only a valid UA fixes it.
- SEC publishes their fair-access policy at:
  https://www.sec.gov/os/accessing-edgar-data
