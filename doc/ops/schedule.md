# RenQuant Cadence — Single Source of Truth

**Last updated:** 2026-05-24 (weekly same-recipe manifest selection + repo hygiene mainline). Authoritative table of every scheduled or event-triggered job in the system.

> **Broker mode** (2026-05-11 safety mandate + live-account operator override):
> - `scripts/daily_104.sh` currently runs `--broker alpaca` by operator mandate, so it can submit LIVE orders when buy-side preflight passes.
> - `--broker alpaca-paper` remains the paper API mode (real Alpaca endpoints, no real money) for manual safety testing.
> - `.env` holds LIVE credentials; paper-API calls 401

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
| **Plist** | `~/Library/LaunchAgents/com.renquant.daily104.plist` (separate from `conditional-retrain104.plist`) |
| **Trigger** | Mon-Fri 14:00 PT |
| **Wallclock** | ~5-15 min on M4 Pro 14c (was 5-15 min on M2 Pro 10c too — daily is I/O-bound not CPU) |
| **Touches (mutates)** | `live_state.alpaca.json`, broker positions (Alpaca), `runs.alpaca.db`, `doc/dashboard.md`, `data/portfolio_daily_metrics` rows |
| **Touches (read-only)** | `panel-ltr.alpha158_fund.json`, `panel-rank-calibration.json`, OHLCV cache |
| **Does NOT touch** | model artifacts (no retrain, no promote) |
| **Alert** | ntfy "RenQuant 104 SMOKE-FAIL" if pre-flight smoke test fails (live trade is then ABORTED); ntfy "BUY-BLOCKED" if the active artifact has failed WF evidence and daily falls back to sell-only risk exits |

**Steps:**
1. Lock + NYSE-open guard + drift check
2. **Smoke test** (`scripts/smoke_test_model.py`) — load active artifact, score 2 synthetic rows, assert finite + diverse outputs (BUG #6 class guard)
3. Active-model age check (alert if > 14 days — weekly cron may be failing)
4. Export LEAN watchlist data
5. Backfill forward returns + recompute portfolio metrics (broker-tagged DB)
6. Run `live.runner --strategy renquant_104 --broker alpaca --once`; if P-WF-GATE blocks full mode, rerun `--sell-only` so exits/risk controls still execute while new buys remain blocked
7. Refresh `doc/dashboard.md`
8. Run PatchTST shadow read-only e2e with a wall-clock timeout (`RENQUANT_SHADOW_TIMEOUT_SEC`, default 1800s); timeout/failure is non-fatal after the primary path has completed

---

## 📅 Weekly (Sat 04:00 PT — trust boundary for promote)

| Field | Value |
|---|---|
| **Script** | `scripts/weekly_wf_promote.sh` |
| **Plist** | `scripts/launchd/com.renquant.weekly-wf-promote.plist` |
| **Trigger** | Saturday 04:00 PT (07:00 ET, NYSE closed weekend) |
| **Wallclock** | ~60-70 min on M4 Pro 14c (75-90 min on prior M2 Pro 10c) |
| **Touches (mutates)** | unique `*.weekly_<RUN_ID>.staging.json` scorer/calibrator first; active `panel-ltr.alpha158_fund.json` and `panel-rank-calibration.json` only after WF pass; `data/sec_fundamentals_daily.parquet` if the retrain wrapper regenerates inputs |
| **Touches (read-only)** | training panel, OHLCV cache |
| **Alert (success)** | "RenQuant 104 WEEKLY-PROMOTE ✓" with WF Sharpe + APY + sanity IC |
| **Alert (failure)** | "RenQuant 104 WEEKLY-FAIL" — prior model preserved |

**Steps:**
1. Pre-flight smoke test (abort if model broken before 60-min commit)
2. Retrain `daily_retrain_alpha158_fund.sh` into unique staging paths via `--xgb-artifact-out` + `--calibrator-out`; active production remains untouched.
3. `scripts/run_wf_gate.py --derive-config-from-prod --strategy-config strategy_config.sim_wl200_172_sentiment.calibrated_causal.json --strict --jobs 3` — WF + §5.2 sanity + trade-ledger gates. The derived WF config keeps production decision semantics but inherits sim-scoped regime/correlation/calibration artifacts. If the base config points at a stale manifest, the gate auto-selects the same-recipe manifest with the widest retrain coverage; if none exists, it keeps the base manifest so recipe validation fails closed.
4. Active swap only if staged scorer has `wf_gate_metadata.passed=True`; scorer and paired calibrator are copied through `.incoming.json` then `os.replace`.
5. Emergency shell-env `RQ_ALLOW_NO_WF=1` remains confined to `scripts/manual_promote.sh`.
6. Refresh dashboard
7. ntfy with verdict

**Trust invariant:** EVERY model that ships to production (live trading) passes through this gate. The daily cron has no path to promote (staging only). The only override is `scripts/manual_promote.sh` (3-confirmation manual) — and even that requires a follow-up weekly run within 24h.

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

**Why monthly:** Platt-scaling calibrator (switched from isotonic 2026-05-18) parameters drift as score distribution shifts (regime change, watchlist evolution). Refit catches drift without model change. n_unique_prob_y ≥ 10 invariant prevents the "calibrator collapsed to 7 buckets" failure mode (acceptance gate G2). 2026-05-17 monthly cron added H2a (non-collapse) + H2b (IC-regression) hard gates with auto-rollback (commit `637594e`).

**Steps:**
1. Pre-fit pre-refit backup (commit `637594e`)
2. Pre-flight smoke test
3. `scripts/fit_panel_calibrator.py --strategy renquant_104 --method platt` — refit Platt scaling on current (model, panel) pair; clip `expected_return.y` to [-0.20, +0.20] at train-site (per 2026-05-15 P0 fix)
4. H2a non-collapse hard gate (n_unique_prob_y ≥ 10)
5. H2b IC-regression hard gate (pool_ic drop > 2pp → rollback)
6. Post-fit smoke test
7. ntfy summary

---

## 🔔 Event-triggered (manual, no cron)

### `scripts/event_watchlist_change.sh`
**When:** after `strategy_config.json` watchlist changes (e.g. 2026-05-18 wl103 → wl200 promotion). Watchlist edits require panel rebuild on the new universe.
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
| Daily full mode blocked by failed active WF evidence | `logs/daily_104/<date>.log` | ntfy "BUY-BLOCKED"; daily reruns sell-only risk exits |
| Daily shadow e2e timeout | `logs/daily_104/<date>_shadow.log` | ntfy "SHADOW-TIMEOUT"; primary path already completed |
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
