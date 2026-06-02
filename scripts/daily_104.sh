#!/usr/bin/env bash
# daily_104.sh — DAILY OPS for renquant_104.
# ─── LIVE MODE (restored 2026-05-11 PM) ─────────────────────────────────
# 2026-05-11 PM: live trading restored per user request (Bug C fix shows
# strategy is profitable: +11.6% APY mean Sharpe 0.77 across 3 windows).
# To restore PAPER mode for safety testing:
#   sed -i "" "s/--broker alpaca/--broker paper/g" scripts/*.sh
# Or add ALPACA_PAPER_API_KEY/SECRET to .env + switch to --broker alpaca-paper
# for Alpaca's paper-trading sandbox (real API, no real money).
# ─────────────────────────────────────────────────────────────────────────
#
# 2026-05-09 refactor (audit FIX-C): retrain + promote MOVED to
# weekly_wf_promote.sh (Sat 04:00 NYC) so the WF + §5.2 sanity gate is
# actually enforced. Daily ops now does:
#   - smoke test (load model, score 1 row, assert non-NaN) — pipeline heartbeat
#   - LEAN data export (panels for next intraday backtest)
#   - forward-returns backfill + portfolio metrics compute
#   - live trade once via Alpaca
#   - dashboard refresh
#
# Rationale (per doc/ops/schedule.md):
#   * production label is fwd_60d → only 1 new label-row/ticker/day = 0.014%
#     of 700k-row panel = statistical noise. Daily retrain mostly cargo-cult.
#   * 5 RED bugs from 2026-05-09 audit were daily-retrain-introduced
#     (silent corruption + auto-promote on noise, no WF gate enforcement).
#   * Weekly retrain + WF gate + §5.2 sanity = real trust boundary.
#
# Designed for launchd/cron on macOS. Runs unattended.
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
# 2026-05-11 audit M-env: switched conda → .venv per feedback_python_env.md
VENV_DIR="/Users/renhao/git/github/RenQuant/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$REPO_DIR/logs/daily_104"
NTFY_TOPIC="renquant"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"

notify() {
    local title="$1" body="$2"
    if command -v terminal-notifier &>/dev/null; then
        terminal-notifier -title "$title" -message "$body" -sound Glass 2>/dev/null || true
    fi
    curl -s -H "Title: $title" -d "$body" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
}

# Load Alpaca credentials
CRED_FILE="$REPO_DIR/.env"
if [ -f "$CRED_FILE" ]; then
    set -a
    source "$CRED_FILE"
    set +a
else
    echo "ERROR: $CRED_FILE not found. Create it with ALPACA_API_KEY and ALPACA_SECRET_KEY." | tee -a "$LOG"
    exit 1
fi

# Resolve pinned subrepo runtime before invoking daily_multirepo.py or
# live_multirepo.py. Missing assembly env falls back to sibling checkouts.
GITHUB_DIR="$(dirname "$REPO_DIR")"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh"
renquant_load_subrepo_env "$REPO_DIR"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR")"
export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"
if ! PROD_STRATEGY_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json)"; then
    if { [ "${RENQUANT_STRICT_SUBREPO_PATHS:-0}" = "1" ] || [ "${RENQUANT_OPS_FAIL_CLOSED:-0}" = "1" ]; } \
        && [ "${RQ_DAILY_RUNNER:-multirepo}" != "umbrella" ]; then
        echo "ERROR: pinned renquant-strategy-104 strategy_config.json unavailable" | tee -a "$LOG"
        exit 1
    fi
    PROD_STRATEGY_CONFIG="$REPO_DIR/backtesting/renquant_104/strategy_config.json"
fi

exec >> "$LOG" 2>&1
echo "=== daily_104 started at $(date) ==="

# ── Lock file — prevent concurrent invocations ────────────────────────────────
# Audit fix LOCK-STALE (Round 2 deep audit, 2026-04-25): pre-fix, a
# stale lock with a dead PID (left over after a SIGKILL / kernel panic
# / hard reboot — when the EXIT trap doesn't fire) blocked every
# subsequent run silently. After a 6 PM crash, the next morning's
# 6:32 AM run would see the dead-PID lock, log "Another daily_104 run
# is active", and exit 0. No models retrained, market opens, positions
# never exit. Now: when we hit a lock conflict, validate the recorded
# PID is actually alive via `kill -0`. If not, clear the stale lock
# and retry once.
LOCK_FILE="/tmp/renquant_104_daily.lock"
_acquire_lock() {
    ( set -C; echo $$ > "$LOCK_FILE" ) 2>/dev/null
}
if ! _acquire_lock; then
    EXISTING_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "?")
    if [ "$EXISTING_PID" != "?" ] && [ -n "$EXISTING_PID" ] && \
            ! kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "Stale lock detected (PID=$EXISTING_PID is dead) — clearing and retrying."
        rm -f "$LOCK_FILE"
        if ! _acquire_lock; then
            echo "Failed to acquire lock after clearing stale — aborting."
            notify "RenQuant 104 ERROR" "Lock acquire retry failed"
            exit 1
        fi
    else
        echo "Another daily_104 run is active (PID=$EXISTING_PID, lock=$LOCK_FILE) — skipping."
        notify "RenQuant 104 SKIP" "Duplicate daily run blocked (PID=$EXISTING_PID already running)"
        exit 0
    fi
fi
trap "rm -f '$LOCK_FILE'" EXIT

# NYSE calendar guard — skip on market holidays
TODAY_DATE=$(date +%Y-%m-%d)
if ! "$PYTHON" -c "
import sys, pandas_market_calendars as mcal, pandas as pd
cal = mcal.get_calendar('NYSE')
sched = cal.schedule('$TODAY_DATE', '$TODAY_DATE')
sys.exit(0 if len(sched) > 0 else 1)
"; then
    echo "NYSE closed today ($TODAY_DATE) — skipping run."
    notify "RenQuant 104" "Skipped — NYSE holiday ($TODAY_DATE)"
    exit 0
fi
echo "NYSE open today ($TODAY_DATE) — proceeding."

# Drift guard (2026-04-24): alert if strategy_config.json has drifted from
# strategy_config.golden.json. Non-fatal — the run continues — but WARN
# ntfy fires so flag regressions are caught before a bad run completes.
# Common causes: manual edits left behind after an A/B, or a promoted
# change where golden wasn't updated in the same commit.
#
# Audit fix DAILY-DRIFT (Round 2 deep audit, 2026-04-25): pre-fix used
# `|| true` to suppress the script's exit code. That also masked Python
# import errors (e.g., a syntax error in check_config_drift.py would
# silently exit 0 and the run proceeded with NO drift check). Now:
# capture stdout AND exit code separately. Drift script's documented
# exit codes:
#   0 = clean / no drift
#   1 = drift detected (we look at stdout below for the diff)
#   2+ = the script itself crashed (import error, missing golden, etc.)
DRIFT_OUT=$("$PYTHON" "$REPO_DIR/scripts/check_config_drift.py" --strategy renquant_104 2>&1)
DRIFT_RC=$?
if [ "$DRIFT_RC" -ge 2 ]; then
    notify "RenQuant 104 DRIFT-CHECK FAILED" \
        "check_config_drift.py crashed (rc=$DRIFT_RC) — config NOT validated. First 200 chars: ${DRIFT_OUT:0:200}"
    echo "$DRIFT_OUT"
    # Continue the run, but the operator now sees the crash explicitly
    # instead of silent skip.
elif [ "$DRIFT_RC" -eq 1 ] || echo "$DRIFT_OUT" | grep -q "drift detected"; then
    # Only surface booleans + the first 2 numeric lines in ntfy to keep it short
    SHORT=$(echo "$DRIFT_OUT" | grep -E "→" | head -5 | sed 's/^  *//')
    notify "RenQuant 104 DRIFT" "strategy_config.json drifted from golden — $SHORT"
    echo "$DRIFT_OUT"
else
    echo "Config drift OK."
fi

# Step 1: SMOKE TEST — pipeline heartbeat (replaces daily retrain).
# 2026-05-09 audit FIX-C: retrain moved to weekly_wf_promote.sh.
# Daily smoke test verifies the model artifact loads + scores correctly
# WITHOUT consuming 75 min compute or risking BUG-#1/#6-class corruption.
# A failure here means upstream data or artifact storage is broken —
# operator must investigate before market open.
echo "--- Step 1: Model smoke test (pipeline heartbeat) ---"
cd "$REPO_DIR"
if "$PYTHON" scripts/smoke_test_model.py --strategy renquant_104; then
    echo "Smoke test PASS at $(date)"
else
    echo "Smoke test FAILED at $(date)"
    notify "RenQuant 104 SMOKE-FAIL" "Daily model smoke test failed — see $LOG. Live trade WILL NOT proceed."
    exit 1
fi

# Step 1b: Surface model age — alert if active artifact is > 14 days old.
# 2026-05-09 audit FIX-C: replaces the old "Models retrained" ntfy that
# fired on every daily promote. Now: weekly cadence is the source of
# trust; this just surfaces the active artifact's age + IC so the
# operator sees "model is N days old, last WF mean IC = X" each day.
PANEL_INFO=$("$PYTHON" -c "
import json, datetime
from pathlib import Path
sd = Path('$REPO_DIR/backtesting/renquant_104')
cfg = json.loads(Path('$PROD_STRATEGY_CONFIG').read_text())
p_rel = cfg['ranking']['panel_scoring']['artifact_path']
try:
    p = json.loads((sd / p_rel).read_text())
except Exception:
    p = {}
td = p.get('trained_date') or '—'
ic = p.get('oos_mean_ic')
ic_s = f'{ic:+.4f}' if isinstance(ic, (int, float)) else '—'
nf = len(p.get('feature_cols', []))
try:
    age = (datetime.date.today() - datetime.date.fromisoformat(td)).days
except Exception:
    age = -1
print(f'panel@{td} ({age}d old) IC={ic_s} n_feat={nf}')
" 2>/dev/null || echo "panel info unavailable")
echo "Active model: $PANEL_INFO"

# Stale-model alert: if active artifact is > 14 days old, the weekly
# WF cron either didn't run or rejected every promote. Operator must
# investigate. 14d = 2 weekly cycles + buffer.
MODEL_AGE_DAYS=$("$PYTHON" -c "
import json, datetime
from pathlib import Path
sd = Path('$REPO_DIR/backtesting/renquant_104')
try:
    cfg = json.loads(Path('$PROD_STRATEGY_CONFIG').read_text())
    p = json.loads((sd / cfg['ranking']['panel_scoring']['artifact_path']).read_text())
    age = (datetime.date.today() - datetime.date.fromisoformat(p['trained_date'])).days
    print(age)
except Exception:
    print(-1)
" 2>/dev/null || echo "-1")
if [ "$MODEL_AGE_DAYS" -gt 14 ] 2>/dev/null; then
    notify "RenQuant 104 STALE-MODEL" "Active artifact ${MODEL_AGE_DAYS}d old — weekly WF cron may be failing. Check logs/weekly_wf_promote/."
fi

# Step 2: Export LEAN data for all watchlist symbols
echo "--- Step 2: Exporting LEAN watchlist data ---"
if "$PYTHON" scripts/export_lean_watchlist.py --strategy renquant_104; then
    echo "LEAN data export finished at $(date)"
else
    echo "LEAN data export FAILED at $(date)"
    notify "RenQuant 104 ERROR" "LEAN data export failed — check $LOG"
    exit 1
fi

# Step 2b: Backfill forward returns (yesterday's and older candidates now have
# enough future bars for fwd_1d / fwd_5d / fwd_10d / fwd_20d) + recompute
# portfolio risk metrics (Sharpe/DD/VaR tracking → goal: APY=1.41, Sharpe=2.0).
echo "--- Step 2b: Backfill forward returns + portfolio metrics ---"
"$PYTHON" scripts/backfill_forward_returns.py --source live --broker alpaca 2>&1 | tail -5 || echo "forward_returns backfill failed (non-fatal)"
"$PYTHON" scripts/compute_portfolio_metrics.py --source live --strategy renquant-104 2>&1 | tail -15 || echo "portfolio metrics compute failed (non-fatal)"

# Step 2c: Refresh news sentiment (2026-06-01 audit fix #4).
# The standalone com.renquant.daily-news-sentiment launchd cron silently
# stopped firing — sentiment data was 12+ trading days stale, causing
# ApplyScoresTask to log hit=0/142 every run (sentiment features all-null).
# Inlining the refresh here makes the daily wrapper self-sufficient: if the
# cron is broken or hasn't loaded, daily still gets fresh sentiment before
# the live trader's panel scoring step. Fast skip (<10s) when the data is
# already fresh; full refresh ~30min when behind.
echo "--- Step 2c: Refresh news sentiment ---"
if [ -x "$REPO_DIR/scripts/daily_news_sentiment_refresh.sh" ]; then
    if "$REPO_DIR/scripts/daily_news_sentiment_refresh.sh" 2>&1 | tail -3; then
        echo "sentiment refresh finished at $(date)"
    else
        echo "sentiment refresh failed (non-fatal — daily continues with stale sentiment)"
    fi
else
    echo "sentiment refresh script missing — skip (non-fatal)"
fi

# Step 3: Run live trading (Alpaca, single pass)
echo "--- Step 3: Running live trader (alpaca) ---"
TRADE_LOG="$REPO_DIR/live/logs/renquant-104/$DATE.json"

# Snapshot trade log length before this run so the notification only shows
# trades placed by THIS run, not earlier runs from the same day.
PRE_COUNT=$("$PYTHON" -c "
import json
from pathlib import Path
log_path = Path('$TRADE_LOG')
try:
    trades = json.loads(log_path.read_text()) if log_path.exists() else []
    print(len(trades))
except Exception:
    print(0)
" 2>/dev/null || echo "0")

BUY_BLOCKED_BY_PREFLIGHT=0
PREFLIGHT_SYSTEM_FAILURE=0

# 2026-05-27: route the daily decision pipeline through the pinned subrepos
# (multi-repo). scripts/daily_multirepo.py aliases lifted kernel.* modules to
# sibling subrepos (+ common/model/execution/backtesting/...) then delegates to
# live.runner.main() with the same argv. Instant rollback (§5.5):
#   RQ_DAILY_RUNNER=umbrella  → plain `-m live.runner` (the untouched baseline).
if [ "${RQ_DAILY_RUNNER:-multirepo}" = "umbrella" ]; then
    RUNNER_ARGS=(-m live.runner)
else
    RUNNER_ARGS=("$REPO_DIR/scripts/daily_multirepo.py")
fi

FULL_RUN_LOG=$(mktemp "/tmp/renquant_104_daily_full.XXXXXX")
if RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1 \
        "$PYTHON" "${RUNNER_ARGS[@]}" --strategy renquant_104 --broker alpaca --once \
        > "$FULL_RUN_LOG" 2>&1; then
    cat "$FULL_RUN_LOG"
    rm -f "$FULL_RUN_LOG"
    echo "=== daily_104 finished at $(date) ==="
else
    FULL_RC=$?
    cat "$FULL_RUN_LOG"
    BUY_SIDE_PREFLIGHT_PATTERN="P-WF-GATE|P-REGIME-IC|P-CONFIG-FP|P-SECTOR-MAP|P-PANEL-CONTRACT|P-CALIBRATOR-HEALTH|P-CALIBRATOR-FLAT-REGION|P-FEATURE-COVER|P-WATCHLIST|P-MODEL-ARTIFACT|P-CORR-METADATA|P-BEST-ITER|P-RUN-ID|P-META-LABEL"
    PREFLIGHT_SYSTEM_FAILURE_PATTERN="P-PREFLIGHT-IMPORT|P-PREFLIGHT-EXCEPTION|P-BROKER-CONNECT"
    PREFLIGHT_FALLBACK_PATTERN="$BUY_SIDE_PREFLIGHT_PATTERN|$PREFLIGHT_SYSTEM_FAILURE_PATTERN"
    if grep -Eq "$PREFLIGHT_FALLBACK_PATTERN" "$FULL_RUN_LOG"; then
        if grep -Eq "$PREFLIGHT_SYSTEM_FAILURE_PATTERN" "$FULL_RUN_LOG"; then
            PREFLIGHT_SYSTEM_FAILURE=1
            echo "Full live trader hit preflight system failure — rerunning sell-only so exits/risk controls still execute."
        else
            BUY_BLOCKED_BY_PREFLIGHT=1
            echo "Full live trader blocked by buy-side preflight gate — rerunning sell-only so exits/risk controls still execute."
        fi
        SELL_ONLY_LOG=$(mktemp "/tmp/renquant_104_daily_sell_only.XXXXXX")
        if "$PYTHON" "${RUNNER_ARGS[@]}" --strategy renquant_104 --broker alpaca --once --sell-only > "$SELL_ONLY_LOG" 2>&1; then
            cat "$SELL_ONLY_LOG"
            rm -f "$SELL_ONLY_LOG" "$FULL_RUN_LOG"
            echo "=== daily_104 finished sell-only fallback at $(date) ==="
        else
            SELL_RC=$?
            cat "$SELL_ONLY_LOG"
            rm -f "$SELL_ONLY_LOG" "$FULL_RUN_LOG"
            echo "=== daily_104 FAILED sell-only fallback at $(date) (full_rc=$FULL_RC sell_rc=$SELL_RC) ==="
            notify "RenQuant 104 ERROR" "Full run blocked by WF gate, and sell-only fallback failed — check $LOG"
            exit 1
        fi
    else
        rm -f "$FULL_RUN_LOG"
        echo "=== daily_104 FAILED at $(date) ==="
        notify "RenQuant 104 ERROR" "Live trader failed — check $LOG"
        exit 1
    fi
fi

# live.runner is the single source of success/trade ntfy. The wrapper only
# sends failure alerts plus this buy-blocked fallback alert; otherwise raw
# wrapper success ntfy duplicates runner alerts and can mis-summarize trades.
if [ "$PREFLIGHT_SYSTEM_FAILURE" -eq 1 ]; then
    notify "RenQuant 104 ERROR" "Full run hit preflight system failure; sell-only fallback completed. Check $LOG"
elif [ "$BUY_BLOCKED_BY_PREFLIGHT" -eq 1 ]; then
    HOLDINGS=$("$PYTHON" -c "
import os
try:
    from alpaca.trading.client import TradingClient
    client = TradingClient(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_SECRET_KEY'], paper=False)
    positions = client.get_all_positions()
    parts = [f\"{p.symbol}{float(p.unrealized_plpc)*100:+.0f}%\" for p in sorted(positions, key=lambda x: x.symbol)]
    print('Held: ' + ' '.join(parts) if parts else 'No positions')
except Exception:
    print('')
" 2>/dev/null || echo "")
    BUY_BLOCKED_ALERT_STAMP="$LOG_DIR/.buy_blocked_alert_stamp"
    BUY_BLOCKED_COOLDOWN_SEC="${RENQUANT_BUY_BLOCKED_ALERT_COOLDOWN_SEC:-21600}"
    NOW_SEC=$(date +%s)
    LAST_SEC=0
    if [ -f "$BUY_BLOCKED_ALERT_STAMP" ]; then
        LAST_SEC=$(cat "$BUY_BLOCKED_ALERT_STAMP" 2>/dev/null || echo 0)
    fi
    if [ $((NOW_SEC - LAST_SEC)) -ge "$BUY_BLOCKED_COOLDOWN_SEC" ]; then
        notify "RenQuant 104 BUY-BLOCKED" "Full run blocked new buys; sell-only fallback completed.${HOLDINGS:+ | $HOLDINGS}"
        echo "$NOW_SEC" > "$BUY_BLOCKED_ALERT_STAMP"
    else
        echo "BUY-BLOCKED ntfy suppressed by cooldown (${BUY_BLOCKED_COOLDOWN_SEC}s)."
    fi
else
    echo "Wrapper success ntfy suppressed; live.runner already posted the cycle decision."
fi

# Sustainability audit (Plan D, 2026-04-23): append one JSONL row
# to logs/live_104/audit.jsonl summarizing today's live state.
# scripts/weekly_apy_check.py consumes this stream to compute
# rolling 30-day APY and fire ntfy alerts when live deviates from
# the golden backtest baseline.
AUDIT_DIR="$REPO_DIR/logs/live_104"
AUDIT_LOG="$AUDIT_DIR/audit.jsonl"
mkdir -p "$AUDIT_DIR"
"$PYTHON" -c "
import json, os
from datetime import datetime
from pathlib import Path

def _safe(fn, default=None):
    try: return fn()
    except Exception: return default

# Account snapshot from Alpaca. Use non-margin buying power for cash parity
# with live broker sizing; settled_cash is logged separately for audit.
equity = cash = settled_cash = None
n_positions = 0
try:
    from alpaca.trading.client import TradingClient
    client = TradingClient(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_SECRET_KEY'], paper=False)
    acct = client.get_account()
    equity       = float(acct.equity)
    settled_cash = float(acct.cash)
    cash         = float(getattr(acct, 'non_marginable_buying_power', acct.cash))
    n_positions = len(client.get_all_positions())
except Exception as exc:
    pass

# Regime + HWM from broker-isolated live_state file
# (daily_104 always runs --broker alpaca, so use the alpaca-tagged path;
# fall back to legacy live_state.json during the migration window)
_alpaca = Path('$REPO_DIR/backtesting/renquant_104/live_state.alpaca.json')
_legacy = Path('$REPO_DIR/backtesting/renquant_104/live_state.json')
state_path = _alpaca if _alpaca.exists() else _legacy
hwm = regime = confidence = None
try:
    s = json.loads(state_path.read_text())
    hwm         = float(s.get('high_water_mark', 0) or 0) or None
    regime      = s.get('regime')
    confidence  = float(s.get('regime_confidence', 0) or 0) or None
except Exception:
    pass

# Count of orders placed THIS run (trade_log entries past the PRE_COUNT snapshot)
n_orders = 0
try:
    tl = Path('$TRADE_LOG')
    if tl.exists():
        n_orders = max(0, len(json.loads(tl.read_text())) - $PRE_COUNT)
except Exception:
    pass

drawdown = None
if equity and hwm and hwm > 0:
    drawdown = round(max(0.0, (hwm - equity) / hwm), 4)

row = {
    'date':            '$DATE',
    'timestamp':       datetime.utcnow().isoformat(timespec='seconds') + 'Z',
    'equity':          round(equity, 2) if equity is not None else None,
    'cash':            round(cash, 2)   if cash   is not None else None,
    'settled_cash':    round(settled_cash, 2) if settled_cash is not None else None,
    'hwm':             round(hwm, 2)    if hwm    is not None else None,
    'drawdown_pct':    drawdown,
    'n_positions':     n_positions,
    'n_orders_today':  n_orders,
    'regime':          regime,
    'confidence':      round(confidence, 3) if confidence is not None else None,
}
with open('$AUDIT_LOG', 'a') as f:
    f.write(json.dumps(row) + '\n')
print(f\"audit: equity={equity}  hwm={hwm}  drawdown={drawdown}  n_orders_today={n_orders}  regime={regime}\")
" 2>&1 || echo "audit write failed (non-fatal)"

# Refresh the metrics dashboard (non-fatal — purely informational).
# Reads from runs.alpaca.db + live_state.alpaca.json that the live runner
# just updated above. Output: doc/dashboard.md (auto-rendered on GitHub).
"$PYTHON" "$REPO_DIR/scripts/build_dashboard.py" --broker alpaca \
    --out "$REPO_DIR/doc/dashboard.md" 2>&1 \
    || echo "dashboard refresh failed (non-fatal)"

# ── Step 4: SHADOW e2e run (2026-05-19) ──────────────────────────────────
# Per user mandate "整条 pipeline 都参考 shadow model 的 output — 跑两遍
# e2e，但是 shadow 那一遍虽然连 alpaca 洗数据，但是并不真下单!"
# Plus: "或者你直接搞一个 shadow 的 config — 避免污染么，隔离干净".
#
# Runs full InferencePipeline with HF PatchTST (seed_44 from canonical
# 5-seed 2026-05-19) as PRIMARY. Broker = readonly-alpaca wrapper:
# reads (account / holdings / quotes / fills) hit LIVE alpaca for ground
# truth, writes (place_order / cancel / stop) get swallowed locally.
# State writes to live_state.alpaca_shadow.json + runs_alpaca_shadow.db
# (broker_name="alpaca_shadow" → automatic path isolation). Prod state
# files are NOT touched.
#
# Non-fatal: a shadow failure doesn't abort the prod cycle (prod already
# completed + committed by this point). Logs to a separate file for
# clean diff between prod and shadow outcomes, and sends one wrapper ntfy by
# default so a broken shadow path is not silent.
#
# ntfy uses "[SHADOW]RENQUANT-104" prefix on success. If shadow preflight
# fails, daily_104 owns the single non-fatal wrapper alert; suppress the
# inner runner preflight ntfy to avoid duplicate phone errors.
echo "--- Step 4: Shadow e2e run (HF PatchTST primary, no real orders) ---"
SHADOW_LOG="$LOG_DIR/${DATE}_shadow.log"
# HF PatchTST shadow is a full e2e pass: live broker reads, panel-frame
# assembly, fundamentals/earnings/insider context, then sequence inference.
# Empirical 2026-05-22 run exceeded the old 420s cap during cold start,
# producing a false shadow failure after the production pass had succeeded.
# Keep a wall-clock kill switch, but size it for the actual workload.
SHADOW_TIMEOUT_SEC="${RENQUANT_SHADOW_TIMEOUT_SEC:-1800}"
if RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1 "$PYTHON" - <<PY > "$SHADOW_LOG" 2>&1
import os
import subprocess
import sys

if os.environ.get("RQ_DAILY_RUNNER", "multirepo") == "umbrella":
    runner = [sys.executable, "-m", "live.runner"]
else:
    runner = [sys.executable, "$REPO_DIR/scripts/live_multirepo.py"]

cmd = runner + [
    "--strategy", "renquant_104",
    "--broker", "readonly-alpaca",
    "--once",
    "--strategy-config-name", "strategy_config.shadow.json",
]
try:
    raise SystemExit(subprocess.run(
        cmd,
        cwd="$REPO_DIR",
        timeout=float("$SHADOW_TIMEOUT_SEC"),
    ).returncode)
except subprocess.TimeoutExpired:
    print("SHADOW TIMEOUT after ${SHADOW_TIMEOUT_SEC}s", flush=True)
    raise SystemExit(124)
PY
then
    echo "Shadow run finished — see $SHADOW_LOG"
    # Surface the shadow ntfy line in the prod log so the operator can
    # see both decisions in one place if the daily log is what they read.
    grep "ntfy sent:" "$SHADOW_LOG" | tail -1 || echo "shadow ntfy line not found in shadow log"
else
    SHADOW_RC=$?
    if [ "$SHADOW_RC" -eq 124 ]; then
        echo "Shadow run TIMED OUT after ${SHADOW_TIMEOUT_SEC}s (non-fatal) — see $SHADOW_LOG"
        if [ "${RENQUANT_SHADOW_ALERT_NTFY:-1}" != "0" ]; then
            notify "RenQuant 104 SHADOW-TIMEOUT" "Shadow e2e exceeded ${SHADOW_TIMEOUT_SEC}s; primary already completed. See $SHADOW_LOG."
        else
            echo "Shadow timeout ntfy suppressed (RENQUANT_SHADOW_ALERT_NTFY=0)."
        fi
    else
        SHADOW_BUY_SIDE_PREFLIGHT_PATTERN="P-WF-GATE|P-REGIME-IC|P-CONFIG-FP|P-SECTOR-MAP|P-PANEL-CONTRACT|P-CALIBRATOR-HEALTH|P-CALIBRATOR-FLAT-REGION|P-FEATURE-COVER|P-WATCHLIST|P-MODEL-ARTIFACT|P-CORR-METADATA|P-BEST-ITER|P-RUN-ID|P-META-LABEL"
        if grep -Eq "$SHADOW_BUY_SIDE_PREFLIGHT_PATTERN" "$SHADOW_LOG"; then
            echo "Shadow run blocked by expected buy-side preflight gate (non-fatal, rc=$SHADOW_RC) — see $SHADOW_LOG"
            echo "Shadow preflight-block ntfy suppressed; prod path already reported the actionable gate."
        elif [ "${RENQUANT_SHADOW_ALERT_NTFY:-1}" != "0" ]; then
            echo "Shadow run FAILED (non-fatal, rc=$SHADOW_RC) — see $SHADOW_LOG"
            notify "RenQuant 104 SHADOW-FAIL" "Shadow e2e failed today (rc=$SHADOW_RC) — primary already completed. See $SHADOW_LOG."
        else
            echo "Shadow run FAILED (non-fatal, rc=$SHADOW_RC) — see $SHADOW_LOG"
            echo "Shadow failure ntfy suppressed (RENQUANT_SHADOW_ALERT_NTFY=0)."
        fi
    fi
fi
