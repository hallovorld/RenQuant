# RenQuant Cadence — Single Source of Truth

**Last updated:** 2026-05-09 (audit FIX-C). Authoritative table of every scheduled or event-triggered job in the system.

## Why this doc exists

Pre-2026-05-09 the cadence was implicit — `daily_104.sh` did EVERYTHING (retrain + promote + live trade + dashboard) and the WF gate was bypassed via `RQ_ALLOW_NO_WF=1`. Result: bad models shipped silently, single-cut acceptance gates let through models that failed walk-forward. Post-audit, the cadence is split:

- **Daily** = ops only (live trade + monitor + heartbeat) — no model changes
- **Weekly** = trust boundary (full WF + sanity battery before promote)
- **Monthly** = drift maintenance (calibrator only, no booster touch)
- **Event-triggered** = manual operator-initiated overrides

Every job below documents: **what it does, what files it touches, what alerts on failure**.

---

## 📅 Daily (every NYSE trading day, ~14:00 PT)

| Field | Value |
|---|---|
| **Script** | `scripts/daily_104.sh` |
| **Plist** | `scripts/launchd/com.renquant.conditional-retrain104.plist` (the existing "daily 104" cron) |
| **Trigger** | Mon-Fri 14:00 PT (existing existing) |
| **Wallclock** | ~5-15 min |
| **Touches (mutates)** | `live_state.alpaca.json`, broker positions (Alpaca), `runs.alpaca.db`, `doc/dashboard.md`, `data/portfolio_daily_metrics` rows |
| **Touches (read-only)** | `panel-ltr.alpha158_fund.json`, `panel-rank-calibration.json`, OHLCV cache |
| **Does NOT touch** | model artifacts (no retrain, no promote) |
| **Alert** | ntfy "RenQuant 104 SMOKE-FAIL" if pre-flight smoke test fails (live trade is then ABORTED) |

**Steps:**
1. Lock + NYSE-open guard + drift check
2. **Smoke test** (`scripts/smoke_test_model.py`) — load active artifact, score 2 synthetic rows, assert finite + diverse outputs (BUG #6 class guard)
3. Active-model age check (alert if > 14 days — weekly cron may be failing)
4. Export LEAN watchlist data
5. Backfill forward returns + recompute portfolio metrics (broker-tagged DB)
6. Run `live.runner --strategy renquant_104 --broker alpaca --once`
7. Refresh `doc/dashboard.md`

---

## 📅 Weekly (Sat 04:00 PT — trust boundary for promote)

| Field | Value |
|---|---|
| **Script** | `scripts/weekly_wf_promote.sh` |
| **Plist** | `scripts/launchd/com.renquant.weekly-wf-promote.plist` |
| **Trigger** | Saturday 04:00 PT (07:00 ET, NYSE closed weekend) |
| **Wallclock** | ~90 min (75 min train + 15 min WF + sanity) |
| **Touches (mutates)** | `panel-ltr.alpha158_fund.json` (only on WF pass), `panel-rank-calibration.json` (refit), `data/sec_fundamentals_daily.parquet` (if `daily_retrain_alpha158_fund.sh` regenerates) |
| **Touches (read-only)** | training panel, OHLCV cache |
| **Alert (success)** | "RenQuant 104 WEEKLY-PROMOTE ✓" with WF Sharpe + APY + sanity IC |
| **Alert (failure)** | "RenQuant 104 WEEKLY-FAIL" — prior model preserved |

**Steps:**
1. Pre-flight smoke test (abort if model broken before 90-min commit)
2. Retrain via `daily_retrain_alpha158_fund.sh` (panel-LTR + alpha158 + fund + PEAD + SUE)
3. `scripts/run_wf_gate.py --strict` — 3-cut WF + §5.2 sanity (shuffled-label + time-shift placebo)
4. **`promote()` WITHOUT `RQ_ALLOW_NO_WF` override** — `_check_wf_gate` refuses if metadata missing or `passed=False`
5. Refresh dashboard
6. ntfy with verdict

**Trust invariant:** EVERY model that ships to production (live trading) passes through this gate. The daily cron has no path to promote. The only override is `scripts/manual_promote.sh` (3-confirmation manual) — and even that requires a follow-up weekly run within 24h.

---

## 📅 Monthly (1st of month, 03:00 PT — calibrator drift)

| Field | Value |
|---|---|
| **Script** | `scripts/monthly_calibrator_refresh.sh` |
| **Plist** | `scripts/launchd/com.renquant.monthly-calibrator-refresh.plist` |
| **Trigger** | 1st of every month, 03:00 PT |
| **Wallclock** | ~10-15 min |
| **Touches (mutates)** | `panel-rank-calibration.json` (if smoke + non-collapse passes) |
| **Touches (read-only)** | active panel-LTR model, OHLCV cache |
| **Alert (success)** | "RenQuant 104 MONTHLY-CAL ✓" with knot count + n_unique_prob_y |
| **Alert (failure)** | "RenQuant 104 MONTHLY-FAIL" — prior calibrator preserved |

**Why monthly:** isotonic calibrator's knot positions can drift as score distribution shifts (regime change, watchlist evolution). Refit catches drift without model change. n_unique_prob_y ≥ 10 invariant prevents the "calibrator collapsed to 7 buckets" failure mode (acceptance gate G2).

**Steps:**
1. Pre-flight smoke test
2. `scripts/fit_panel_calibrator.py --strategy renquant_104` — refit isotonic on current (model, panel) pair
3. Post-fit smoke test — abort if calibrator collapsed
4. ntfy summary

---

## 🔔 Event-triggered (manual, no cron)

### `scripts/event_watchlist_change.sh`
**When:** after `strategy_config.json` watchlist changes (e.g. wl103 → wl162). Watchlist edits require panel rebuild on the new universe.
**Behavior:** prompts for confirmation, then delegates to `weekly_wf_promote.sh` (same trust boundary).

### `scripts/event_sec_schema_change.sh`
**When:** after `scripts/fetch_sec_fundamentals.py` is modified (e.g. period change, new field). Parquet on disk is stale until regenerated.
**Behavior:** prompts to regenerate `data/sec_fundamentals_daily.parquet` (~60 min, hits SEC EDGAR rate-limited), then triggers `weekly_wf_promote.sh`. `--skip-fetch` skips regeneration.

### `scripts/manual_promote.sh`
**When:** EMERGENCY ONLY (regulatory, hotfix, disaster recovery). Bypasses WF gate via `RQ_ALLOW_NO_WF=1`.
**Behavior:** requires three explicit confirmations:
1. Artifact path
2. Reason (must be one of: `emergency_bugfix` / `regulatory` / `hotfix` / `disaster_recovery`)
3. Rollback rehearsed (per CLAUDE.md §5.5)

After every emergency promote, operator MUST run `weekly_wf_promote.sh` within 24h to validate against the proper trust boundary.

### `scripts/conditional_retrain_104.sh` (existing)
**When:** Mon-Fri 13:10 PT, only if SPY |Δ| > 2% OR VIX |Δ| > 5%.
**Behavior:** anomaly-triggered conditional retrain. Pre-existing pre-FIX-C; preserved for fast reaction to regime shifts.

---

## 🔧 Install / verify

```bash
# Install all 3 launchd plists (weekly + monthly + existing daily)
cp scripts/launchd/com.renquant.weekly-wf-promote.plist ~/Library/LaunchAgents/
cp scripts/launchd/com.renquant.monthly-calibrator-refresh.plist ~/Library/LaunchAgents/

launchctl load ~/Library/LaunchAgents/com.renquant.weekly-wf-promote.plist
launchctl load ~/Library/LaunchAgents/com.renquant.monthly-calibrator-refresh.plist

# Verify all loaded
launchctl list | grep renquant
```

---

## 📊 Failure observability

| Failure mode | Where you find it | Alert mode |
|---|---|---|
| Daily smoke test fails | `logs/daily_104/<date>.log` | ntfy "SMOKE-FAIL", live trade aborted |
| Active model > 14 days old | dashboard headline + ntfy | ntfy "STALE-MODEL" |
| Weekly WF gate rejects | `logs/weekly_wf_promote/<date>.log` | ntfy "WEEKLY-FAIL", prior preserved |
| Weekly cron didn't run | dashboard age field shows N+ days | (caught by next daily age check) |
| Monthly calibrator collapses | `logs/monthly_calibrator/<date>.log` | ntfy "MONTHLY-FAIL", prior preserved |
| Live broker error | `logs/daily_104/<date>.log` + ntfy "ERROR" | ntfy "ERROR" |

---

## 🛡 Cadence invariants (must remain true)

1. **No production promote without WF gate** — except `manual_promote.sh` with 3-confirmation + 24h follow-up rule.
2. **Daily script never touches model artifacts** — only ops (live, dashboard, monitor).
3. **Weekly always pre-flights smoke test** — never commits 90-min compute on already-broken pipeline.
4. **Every cron writes to `logs/<job>/<date>.log`** — operator can grep one place to debug.
5. **Every cron has lock file** — concurrent invocations no-op cleanly.
6. **NYSE holiday guard** — daily skips on closed days.
7. **ntfy on every fail path** — silent failure is forbidden.

If any of these ever stops being true, audit the responsible script + add a regression test that pins the invariant.

---

## Pre-FIX-C cadence (historical, for reference)

```
daily_104.sh → retrain (90 min, 3x/week) + auto-promote (RQ_ALLOW_NO_WF=1) + live trade
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              this entire block is gone post-FIX-C
```

Why this was wrong (audit doc/AUDIT_2026-05-09.md):
- 60-day forward label means daily retrain learns from 0.014% new info per day
- Auto-promote on G1-G11 acceptance gates let single-cut "wins" reach production
- WF gate code shipped but bypassed in production = theatrical
- 5 RED bugs from 2026-05-09 audit were daily-retrain-introduced (silent corruption + auto-promote)

Post-FIX-C: weekly cadence is the trust boundary. Daily is ops only.
