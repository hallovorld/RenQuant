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

_kill_process_tree() {
    local root="$1"
    local child
    for child in $(pgrep -P "$root" 2>/dev/null || true); do
        _kill_process_tree "$child"
    done
    kill -TERM "$root" 2>/dev/null || true
}

run_news_sentiment_refresh() {
    local timeout_sec="${RENQUANT_DAILY_NEWS_TIMEOUT_SEC:-1200}"
    local tmp_log pid start_ts now_ts elapsed rc
    tmp_log=$(mktemp "/tmp/renquant_104_news_sentiment.XXXXXX") || return 1

    "$REPO_DIR/scripts/daily_news_sentiment_refresh.sh" > "$tmp_log" 2>&1 &
    pid=$!
    start_ts=$(date +%s)

    while kill -0 "$pid" 2>/dev/null; do
        now_ts=$(date +%s)
        elapsed=$((now_ts - start_ts))
        if [ "$elapsed" -ge "$timeout_sec" ]; then
            echo "sentiment refresh timed out after ${timeout_sec}s (non-fatal; daily continues with stale sentiment)"
            _kill_process_tree "$pid"
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                kill -KILL "$pid" 2>/dev/null || true
            fi
            wait "$pid" 2>/dev/null || true
            tail -20 "$tmp_log" || true
            rm -f "$tmp_log"
            return 124
        fi
        sleep 5
    done

    wait "$pid"
    rc=$?
    tail -3 "$tmp_log" || true
    rm -f "$tmp_log"
    return "$rc"
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
export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-orchestrator renquant-common renquant-base-data renquant-artifacts renquant-model renquant-pipeline renquant-execution renquant-strategy-104 renquant-backtesting):${PYTHONPATH:-}"
if ! PROD_STRATEGY_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json)"; then
    # RenQuant#546 (2026-07-30): fail CLOSED by default.
    #
    # This used to fall back to the umbrella copy unless one of two env vars
    # was set, and BOTH default to 0 — so the default path silently substituted
    # a DIFFERENT config and did not even log it (the ERROR line below was
    # inside the gate). Measured at the time: the umbrella copy names
    # `hf_patchtst` as the PRIMARY scorer with xgb as shadow, while the pinned
    # config — the one that actually runs — has exactly the inverse. So a
    # resolver failure promoted a 623-day-stale shadow checkpoint to primary,
    # and PatchTST's scores are intrinsically all-negative, which fails the
    # ordinary buy floor for every name: a silent sell-only book. That is the
    # 2026-07-15 incident class, reached with nobody taking an action.
    #
    # There is no umbrella-side reference that could validate the fallback:
    # the golden file carries the SAME inverted intent, so the drift guard
    # below compares one stale copy against another and reports clean forever.
    # The only authority on which model is primary is the pinned config. If it
    # cannot be resolved, the correct action is to not run.
    #
    # The umbrella runner is the one mode that legitimately has no subrepo
    # runtime, so it keeps the fallback — loudly, and only there.
    if [ "${RQ_DAILY_RUNNER:-multirepo}" != "umbrella" ]; then
        echo "ERROR: pinned renquant-strategy-104 strategy_config.json unavailable — refusing to run. The umbrella copy is NOT an equivalent config (different primary panel_scoring.kind); substituting it would run a different strategy. Restore the pinned subrepo runtime, or set RQ_DAILY_RUNNER=umbrella deliberately." | tee -a "$LOG"
        exit 1
    fi
    echo "WARN: pinned strategy_config.json unavailable; RQ_DAILY_RUNNER=umbrella so falling back to the umbrella copy. Its primary panel_scoring.kind may DIFFER from the pinned config (RenQuant#546)." | tee -a "$LOG"
    PROD_STRATEGY_CONFIG="$REPO_DIR/backtesting/renquant_104/strategy_config.json"
fi

# RenQuant#546: record WHICH scorer this run resolved, in both branches. The
# 2026-07-30 investigation could not answer "which model was primary on day X"
# from any log, which is why this line exists.
RESOLVED_SCORER_KIND="$(python3 -c 'import json,sys;print((json.load(open(sys.argv[1])).get("ranking",{}).get("panel_scoring",{}) or {}).get("kind","UNKNOWN"))' "$PROD_STRATEGY_CONFIG" 2>/dev/null || echo UNREADABLE)"
echo "strategy_config resolved: $PROD_STRATEGY_CONFIG (primary panel_scoring.kind=$RESOLVED_SCORER_KIND)" | tee -a "$LOG"

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

# ── Live-checkout guard (2026-06-25 incident, postmortem #412) ────────────────
# The umbrella checkout MUST be the stable `main` interface before pin-align. A
# stray git op — or a sub-agent operating in this shared live tree instead of its
# own worktree — can (a) leave it on a feature branch, or (b) MOVE local `main`
# itself to a feature/stale commit (the incident did exactly this). pin-align would
# then materialize runtime repos from that checkout's `subrepos.lock.json`, silently
# deploying the wrong pins (the later hard gates only prove consistency WITH the
# checkout, not that it's the stable interface). So we require BOTH: branch == main
# AND HEAD is on the origin/main lineage (= legitimately behind origin, NOT moved to
# a divergent commit — the live tree is deliberately behind origin/main). FATAL by
# default, with an explicit operator escape hatch RENQUANT_ALLOW_NONMAIN_CHECKOUT=1
# so it can never permanently halt you while still refusing to deploy a stray state.
LIVE_BRANCH=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
CHECKOUT_BAD=""
if [ "$LIVE_BRANCH" != "main" ]; then
    CHECKOUT_BAD="on '$LIVE_BRANCH' (expected main)"
elif git -C "$REPO_DIR" rev-parse origin/main >/dev/null 2>&1 \
     && ! git -C "$REPO_DIR" merge-base --is-ancestor HEAD origin/main 2>/dev/null; then
    CHECKOUT_BAD="local main @ $(git -C "$REPO_DIR" rev-parse --short HEAD) is NOT on the origin/main lineage (moved to a divergent commit)"
fi
if [ -n "$CHECKOUT_BAD" ]; then
    if [ "${RENQUANT_ALLOW_NONMAIN_CHECKOUT:-0}" = "1" ]; then
        notify "RenQuant 104 CHECKOUT-OVERRIDE ⚠" \
            "live checkout bad ($CHECKOUT_BAD) but RENQUANT_ALLOW_NONMAIN_CHECKOUT=1 — proceeding (operator override)."
        echo "OVERRIDE: bad checkout ($CHECKOUT_BAD) allowed by RENQUANT_ALLOW_NONMAIN_CHECKOUT=1 — continuing."
    else
        notify "RenQuant 104 CHECKOUT-GUARD ✗" \
            "live umbrella checkout bad: $CHECKOUT_BAD — ABORTED before pin-align so a stray state's pins can't deploy. Force a run: RENQUANT_ALLOW_NONMAIN_CHECKOUT=1. Fix: cd $REPO_DIR && git checkout main && git reset --hard origin/main (only if safe), then make doctor."
        echo "FATAL: $CHECKOUT_BAD — aborting before pin-align (override: RENQUANT_ALLOW_NONMAIN_CHECKOUT=1)."
        exit 1
    fi
else
    echo "Live-checkout guard: on main, on origin/main lineage ✓"
fi

# ── Preflight: align subrepo checkouts to the audited pins, fail-closed ────────
# Run only after duplicate/holiday exits so sync checkout is serialized and only
# happens for a real trading run. Once-daily also warns if umbrella main lags.
PREFLIGHT_CHECK_UMBRELLA=1
# shellcheck source=scripts/preflight_pin_align.sh
source "$REPO_DIR/scripts/preflight_pin_align.sh"

if [ "${RQ_DAILY_RUNNER:-multirepo}" != "umbrella" ]; then
    if ! "$PYTHON" "$REPO_DIR/scripts/runtime_qp_sanity_check.py"; then
        echo "Runtime QP sanity check failed — aborting daily run before live trade."
        notify "RenQuant 104 RUNTIME-SANITY-FAIL" "Stale or incomplete multirepo QP runtime; run make subrepo-runtime-root and paper smoke"
        exit 1
    fi
fi

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

# System health heartbeat (2026-06-25): run system_doctor — pin/runtime drift,
# lock integrity, bundle self-consistency, promote-backup hygiene. NON-FATAL:
# ntfy on RED so drift is caught the day it happens, but never block trading on
# it (the preflight pin-align above already fail-closes on a broken runtime; a
# README smudge shouldn't halt the book). Defensive: skip if not present yet.
if [ -f "$REPO_DIR/scripts/system_doctor.py" ]; then
    DOCTOR_OUT=$("$PYTHON" "$REPO_DIR/scripts/system_doctor.py" 2>&1)
    if [ "$?" -ne 0 ]; then
        DOCTOR_RED=$(echo "$DOCTOR_OUT" | grep -E "RED|✗" | head -4)
        notify "RenQuant 104 DOCTOR" "system health RED (non-fatal): $DOCTOR_RED"
    fi
    echo "$DOCTOR_OUT" | tail -3
fi

# ── Step 0c: earnings-calendar staleness rail (2026-08-30, fail SOFT) ─────────
# The pre/post-earnings buffer reads artifacts/prod/earnings-calendar.json.
# A stale calendar is indistinguishable from "no earnings soon" to
# is_earnings_blocked — the Apr-24-frozen artifact silently disabled the
# buffer for every Aug/Sep 2026 print (HPE bought 08-27 into an early-Sep
# print). com.renquant.earnings-calendar-refresh now refreshes it every
# session morning; this rail makes any regression LOUD (ntfy) without
# blocking the run.
EARN_CAL="$REPO_DIR/backtesting/renquant_104/artifacts/prod/earnings-calendar.json"
EARN_RAIL_OUT=$("$PYTHON" "$REPO_DIR/scripts/earnings_calendar_rail.py" check \
    --calendar "$EARN_CAL" --min-horizon-days 5 2>&1)
EARN_RAIL_RC=$?
echo "$EARN_RAIL_OUT"
if [ "$EARN_RAIL_RC" -ne 0 ]; then
    notify "RenQuant 104 EARNINGS-CAL ⚠" \
        "earnings buffer effectively DISABLED (rail rc=$EARN_RAIL_RC): ${EARN_RAIL_OUT:0:180} — check com.renquant.earnings-calendar-refresh"
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

# Step 2c (news sentiment refresh) MOVED to after the live trade — see Step 3b.
# P0 fix (2026-06-07): the ~30min (or hung) news refresh ran BEFORE the trade,
# so when the over-long job was SIGTERM'd by launchd the live trade never
# executed (account stuck in cash for weeks). The trade is the critical,
# irreplaceable action; it now runs first. Sentiment is one of 172 features and
# one-day staleness is negligible; the dedicated com.renquant.daily-news-sentiment
# cron is the primary refresher, this inline run is the self-sufficient backup.

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

# 2026-06-03: route the daily decision pipeline through the orchestrator-owned
# multirepo bridge. The bridge aliases lifted kernel.* modules to sibling
# subrepos (+ common/model/execution/backtesting/...) then delegates to
# live.runner.main() with the same argv. Instant rollback (§5.5):
#   RQ_DAILY_RUNNER=umbrella  → plain `-m live.runner` (the untouched baseline).
if [ "${RQ_DAILY_RUNNER:-multirepo}" = "umbrella" ]; then
    RUNNER_ARGS=(-m live.runner)
else
    RUNNER_ARGS=(-m renquant_orchestrator daily-bridge --repo-dir "$REPO_DIR")
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
    BUY_SIDE_PREFLIGHT_PATTERN="P-WF-GATE|P-REGIME-IC|P-CONFIG-FP|P-SECTOR-MAP|P-PANEL-CONTRACT|P-CALIBRATOR-HEALTH|P-CALIBRATOR-FLAT-REGION|P-FEATURE-COVER|P-WATCHLIST|P-MODEL-ARTIFACT|P-CORR-METADATA|P-BEST-ITER|P-RUN-ID|P-META-LABEL|P-FUND-FRESHNESS"
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
        # The full-run log is deleted after the sell-only rerun; keep the
        # preflight's own P-WF-GATE / P-REGIME-IC lines so the BUY-BLOCKED
        # alert can quote the verdict the run actually acted on.
        BUY_BLOCKED_PREFLIGHT_LINES=$(grep -E "✗ P-|P-WF-GATE|P-REGIME-IC" "$FULL_RUN_LOG" 2>/dev/null | tail -6 || true)
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
    # 2026-08-30: a buy-blocked book is an URGENT operator event, and the alert
    # must say WHY — which artifact, how old, what the gate stamp says, what the
    # RFC #210 license decided, that exits still run, and what unblocks buys.
    # scripts/buy_blocked_reason.py composes that (read-only on the artifact)
    # and posts through renquant_common.notify.send with Priority: urgent +
    # Tags: rotating_light,rq104. If the sender is unreachable (rc=3) or the
    # POST failed (rc=4) the wrapper falls back to curl WITH the same headers —
    # never to the bare default-priority line that hid the 2026-08-31 block.
    # The stamp is keyed by session DATE: once per session, not a 6 h window
    # (the old cooldown let a 13:55 block re-page at 06:30 and swallow the
    # rest of the day).
    BUY_BLOCKED_ALERT_STAMP="$LOG_DIR/.buy_blocked_alert_stamp"
    BUY_BLOCKED_LAST_DATE=""
    if [ -f "$BUY_BLOCKED_ALERT_STAMP" ]; then
        BUY_BLOCKED_LAST_DATE=$(cat "$BUY_BLOCKED_ALERT_STAMP" 2>/dev/null || echo "")
    fi
    if [ "$BUY_BLOCKED_LAST_DATE" != "$DATE" ]; then
        BUY_BLOCKED_TITLE="RenQuant 104 BUY-BLOCKED (sell-only fallback)"
        # stdout = the composed body (re-sendable); stderr = sender diagnostics.
        BUY_BLOCKED_BODY_FILE=$(mktemp "/tmp/renquant_104_buy_blocked.XXXXXX")
        "$PYTHON" "$REPO_DIR/scripts/buy_blocked_reason.py" \
            --strategy-config "$PROD_STRATEGY_CONFIG" \
            --strategy-dir "$REPO_DIR/backtesting/renquant_104" \
            --preflight-lines "${BUY_BLOCKED_PREFLIGHT_LINES:-}" \
            --today "$DATE" --holdings "${HOLDINGS:-}" --log-path "$LOG" \
            --send > "$BUY_BLOCKED_BODY_FILE" 2> "$BUY_BLOCKED_BODY_FILE.err"
        BUY_BLOCKED_SEND_RC=$?
        echo "BUY-BLOCKED alert ($BUY_BLOCKED_TITLE; Priority: urgent; Tags: rotating_light,rq104):"
        cat "$BUY_BLOCKED_BODY_FILE" "$BUY_BLOCKED_BODY_FILE.err"
        if [ "$BUY_BLOCKED_SEND_RC" -ne 0 ]; then
            echo "BUY-BLOCKED python sender rc=$BUY_BLOCKED_SEND_RC — falling back to curl with urgent headers."
            BUY_BLOCKED_BODY=$(cat "$BUY_BLOCKED_BODY_FILE" 2>/dev/null)
            [ -n "$BUY_BLOCKED_BODY" ] || BUY_BLOCKED_BODY="Full run blocked new buys; sell-only fallback completed. Reason helper failed (rc=$BUY_BLOCKED_SEND_RC); see $LOG"
            if [ "${RENQUANT_NO_NOTIFY:-0}" != "1" ]; then
                curl -s -H "Title: $BUY_BLOCKED_TITLE" -H "Priority: urgent" -H "Tags: rotating_light,rq104" \
                    -d "$BUY_BLOCKED_BODY" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
            fi
        fi
        rm -f "$BUY_BLOCKED_BODY_FILE" "$BUY_BLOCKED_BODY_FILE.err"
        echo "$DATE" > "$BUY_BLOCKED_ALERT_STAMP"
    else
        echo "BUY-BLOCKED ntfy suppressed: already alerted for session date $DATE."
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

# Step 3b: Refresh news sentiment (moved from Step 2c — P0 fix 2026-06-07).
# Runs AFTER the live trade so a slow/hung refresh can never block the trade.
# Time-bounded + non-fatal; the dedicated com.renquant.daily-news-sentiment cron
# is the primary refresher, this is the self-sufficient backup.
echo "--- Step 3b: Refresh news sentiment (post-trade) ---"
if [ -x "$REPO_DIR/scripts/daily_news_sentiment_refresh.sh" ]; then
    if run_news_sentiment_refresh; then
        echo "sentiment refresh finished at $(date)"
    else
        echo "sentiment refresh failed/timed out (non-fatal — trade already executed)"
    fi
else
    echo "sentiment refresh script missing — skip (non-fatal)"
fi

# ── Step 4: RETIRED 2026-08-03 (was: HF PatchTST shadow e2e, 2026-05-19) ─
# The lane this step exercised was retired: the PatchTST line was closed by
# the operator-delegated 2026-08-02 decision (RETIRE; architecture preserved,
# no successor training), and renquant-strategy-104#75 retired the
# hf_patchtst shadow lane from the pinned configs in the same arc. What
# remained here was a daily full pipeline boot of a retired scorer that ended
# at a buy-side preflight refusal every session ("blocked by expected
# buy-side preflight gate, rc=1") — cost and alert-surface noise measuring
# nothing anyone still acts on.
#
# Removed under the operator's 2026-08-03 104/105 repair directive. The
# in-process per-model comparison segments (SHADOW[...] in the prod ntfy) and
# the Step 5 blend lane are unaffected; alpaca_shadow state/log files stay on
# disk as history, orphaned by design. To resurrect a second full-funnel
# e2e lane, clone Step 5's shape (RENQUANT_READONLY_TAG isolation) rather
# than reviving this block from git history verbatim — Step 5 is the
# maintained pattern.

# ── Step 5: SHADOW-BLEND e2e run (2026-07-27, operator directive) ────────
# Option-A rail: a SECOND full-funnel readonly shadow lane, cloned from
# Step 4 verbatim, that will score with the composite blend profile
# (strategy_config.shadow_blend.json) once that profile lands in the
# pinned renquant-strategy-104 configs. Shadow like prod minus submission:
# broker = readonly-alpaca wrapper (reads hit LIVE alpaca, writes
# swallowed), sized picks visible in ntfy.
#
# Lane isolation: RENQUANT_READONLY_TAG=alpaca_shadow_blend threads through
# the live-bridge subprocess into ReadOnlyBrokerWrapper, so this lane's
# state routes to live_state.alpaca_shadow_blend.json +
# runs.alpaca_shadow_blend.db — disjoint from BOTH prod (alpaca) and the
# legacy PatchTST shadow (alpaca_shadow). ntfy title prefix becomes
# "[READONLY][ALPACA_SHADOW_BLEND]" (see live/runner.py
# _readonly_label_prefix) so blend vs legacy stay distinguishable.
#
# GATE: skips with an INFO line while strategy_config.shadow_blend.json is
# absent from the pinned strategy configs dir — this rail lands BEFORE the
# strategy profile exists and auto-activates when the profile appears.
# Non-fatal like Step 4: prod + legacy shadow already completed above.
echo "--- Step 5: Shadow-blend e2e run (composite blend profile, no real orders) ---"
if BLEND_STRATEGY_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.shadow_blend.json)"; then
    echo "shadow_blend profile found at $BLEND_STRATEGY_CONFIG"
    SHADOW_BLEND_LOG="$LOG_DIR/${DATE}_shadow_blend.log"
    SHADOW_BLEND_TIMEOUT_SEC="${RENQUANT_SHADOW_BLEND_TIMEOUT_SEC:-${RENQUANT_SHADOW_TIMEOUT_SEC:-1800}}"
    if RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1 RENQUANT_READONLY_TAG=alpaca_shadow_blend "$PYTHON" - <<PY > "$SHADOW_BLEND_LOG" 2>&1
import os
import subprocess
import sys

if os.environ.get("RQ_DAILY_RUNNER", "multirepo") == "umbrella":
    runner = [sys.executable, "-m", "live.runner"]
else:
    runner = [sys.executable, "-m", "renquant_orchestrator", "live-bridge", "--repo-dir", "$REPO_DIR"]

cmd = runner + [
    "--strategy", "renquant_104",
    "--broker", "readonly-alpaca",
    "--once",
    "--strategy-config-path", "$BLEND_STRATEGY_CONFIG",
]
try:
    raise SystemExit(subprocess.run(
        cmd,
        cwd="$REPO_DIR",
        timeout=float("$SHADOW_BLEND_TIMEOUT_SEC"),
    ).returncode)
except subprocess.TimeoutExpired:
    print("SHADOW-BLEND TIMEOUT after ${SHADOW_BLEND_TIMEOUT_SEC}s", flush=True)
    raise SystemExit(124)
PY
    then
        echo "Shadow-blend run finished — see $SHADOW_BLEND_LOG"
        # Surface the shadow-blend ntfy line in the prod log so the operator
        # can see all three decisions in one place.
        grep "ntfy sent:" "$SHADOW_BLEND_LOG" | tail -1 || echo "shadow-blend ntfy line not found in shadow-blend log"
    else
        SHADOW_BLEND_RC=$?
        if [ "$SHADOW_BLEND_RC" -eq 124 ]; then
            echo "Shadow-blend run TIMED OUT after ${SHADOW_BLEND_TIMEOUT_SEC}s (non-fatal) — see $SHADOW_BLEND_LOG"
            if [ "${RENQUANT_SHADOW_ALERT_NTFY:-1}" != "0" ]; then
                notify "RenQuant 104 SHADOW-BLEND-TIMEOUT" "Shadow-blend e2e exceeded ${SHADOW_BLEND_TIMEOUT_SEC}s; primary already completed. See $SHADOW_BLEND_LOG."
            else
                echo "Shadow-blend timeout ntfy suppressed (RENQUANT_SHADOW_ALERT_NTFY=0)."
            fi
        else
            SHADOW_BLEND_BUY_SIDE_PREFLIGHT_PATTERN="P-WF-GATE|P-REGIME-IC|P-CONFIG-FP|P-SECTOR-MAP|P-PANEL-CONTRACT|P-CALIBRATOR-HEALTH|P-CALIBRATOR-FLAT-REGION|P-FEATURE-COVER|P-WATCHLIST|P-MODEL-ARTIFACT|P-CORR-METADATA|P-BEST-ITER|P-RUN-ID|P-META-LABEL|P-FUND-FRESHNESS"
            if grep -Eq "$SHADOW_BLEND_BUY_SIDE_PREFLIGHT_PATTERN" "$SHADOW_BLEND_LOG"; then
                echo "Shadow-blend run blocked by expected buy-side preflight gate (non-fatal, rc=$SHADOW_BLEND_RC) — see $SHADOW_BLEND_LOG"
                echo "Shadow-blend preflight-block ntfy suppressed; prod path already reported the actionable gate."
            elif [ "${RENQUANT_SHADOW_ALERT_NTFY:-1}" != "0" ]; then
                echo "Shadow-blend run FAILED (non-fatal, rc=$SHADOW_BLEND_RC) — see $SHADOW_BLEND_LOG"
                notify "RenQuant 104 SHADOW-BLEND-FAIL" "Shadow-blend e2e failed today (rc=$SHADOW_BLEND_RC) — primary already completed. See $SHADOW_BLEND_LOG."
            else
                echo "Shadow-blend run FAILED (non-fatal, rc=$SHADOW_BLEND_RC) — see $SHADOW_BLEND_LOG"
                echo "Shadow-blend failure ntfy suppressed (RENQUANT_SHADOW_ALERT_NTFY=0)."
            fi
        fi
    fi
else
    echo "INFO: strategy_config.shadow_blend.json not present in pinned strategy configs ($SUBREPO_ROOT/renquant-strategy-104/configs) — Step 5 shadow-blend skipped (rail dormant until the blend profile lands)."
fi

# ── Step 5b: SHADOW-BLEND-MOM e2e run (GOAL-8 S1, 2026-08-04) ────────────
# The S1 z(prod)+z(slow momentum) lane, cloned from Step 5 per its own
# resurrect-a-second-lane instruction (RENQUANT_READONLY_TAG isolation;
# Step 5 is the maintained pattern). The Step 5 slot stays with the
# certified z(prod)+z(clf) profile; this rail gates on its OWN profile
# (strategy_config.shadow_blend_momentum.json, s104 PR reviewed against
# the frozen S1 prereg doc/research/2026-08-04-goal8-s1-zblend-prereg.md
# in renquant-orchestrator) and is dormant until that profile lands —
# the same lands-before-the-profile shape Step 5 itself shipped with.
# Lane isolation: tag alpaca_shadow_blend_mom → state
# live_state.alpaca_shadow_blend_mom.json + runs.alpaca_shadow_blend_mom.db,
# ntfy prefix "[READONLY][ALPACA_SHADOW_BLEND_MOM]" (generic alpaca_shadow*
# derivation in live/runner.py _readonly_label_prefix — no runner change).
echo "--- Step 5b: Shadow-blend-mom e2e run (S1 momentum blend profile, no real orders) ---"
if BLEND_MOM_STRATEGY_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.shadow_blend_momentum.json)"; then
    echo "shadow_blend_momentum profile found at $BLEND_MOM_STRATEGY_CONFIG"
    SHADOW_BLEND_MOM_LOG="$LOG_DIR/${DATE}_shadow_blend_mom.log"
    SHADOW_BLEND_MOM_TIMEOUT_SEC="${RENQUANT_SHADOW_BLEND_MOM_TIMEOUT_SEC:-${RENQUANT_SHADOW_TIMEOUT_SEC:-1800}}"
    if RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1 RENQUANT_READONLY_TAG=alpaca_shadow_blend_mom "$PYTHON" - <<PY > "$SHADOW_BLEND_MOM_LOG" 2>&1
import os
import subprocess
import sys

if os.environ.get("RQ_DAILY_RUNNER", "multirepo") == "umbrella":
    runner = [sys.executable, "-m", "live.runner"]
else:
    runner = [sys.executable, "-m", "renquant_orchestrator", "live-bridge", "--repo-dir", "$REPO_DIR"]

cmd = runner + [
    "--strategy", "renquant_104",
    "--broker", "readonly-alpaca",
    "--once",
    "--strategy-config-path", "$BLEND_MOM_STRATEGY_CONFIG",
]
try:
    raise SystemExit(subprocess.run(
        cmd,
        cwd="$REPO_DIR",
        timeout=float("$SHADOW_BLEND_MOM_TIMEOUT_SEC"),
    ).returncode)
except subprocess.TimeoutExpired:
    print("SHADOW-BLEND-MOM TIMEOUT after ${SHADOW_BLEND_MOM_TIMEOUT_SEC}s", flush=True)
    raise SystemExit(124)
PY
    then
        echo "Shadow-blend-mom run finished — see $SHADOW_BLEND_MOM_LOG"
        grep "ntfy sent:" "$SHADOW_BLEND_MOM_LOG" | tail -1 || echo "shadow-blend-mom ntfy line not found in shadow-blend-mom log"
    else
        SHADOW_BLEND_MOM_RC=$?
        if [ "$SHADOW_BLEND_MOM_RC" -eq 124 ]; then
            echo "Shadow-blend-mom run TIMED OUT after ${SHADOW_BLEND_MOM_TIMEOUT_SEC}s (non-fatal) — see $SHADOW_BLEND_MOM_LOG"
            if [ "${RENQUANT_SHADOW_ALERT_NTFY:-1}" != "0" ]; then
                notify "RenQuant 104 SHADOW-BLEND-MOM-TIMEOUT" "Shadow-blend-mom e2e exceeded ${SHADOW_BLEND_MOM_TIMEOUT_SEC}s; primary already completed. See $SHADOW_BLEND_MOM_LOG."
            else
                echo "Shadow-blend-mom timeout ntfy suppressed (RENQUANT_SHADOW_ALERT_NTFY=0)."
            fi
        else
            SHADOW_BLEND_MOM_BUY_SIDE_PREFLIGHT_PATTERN="P-WF-GATE|P-REGIME-IC|P-CONFIG-FP|P-SECTOR-MAP|P-PANEL-CONTRACT|P-CALIBRATOR-HEALTH|P-CALIBRATOR-FLAT-REGION|P-FEATURE-COVER|P-WATCHLIST|P-MODEL-ARTIFACT|P-CORR-METADATA|P-BEST-ITER|P-RUN-ID|P-META-LABEL|P-FUND-FRESHNESS"
            if grep -Eq "$SHADOW_BLEND_MOM_BUY_SIDE_PREFLIGHT_PATTERN" "$SHADOW_BLEND_MOM_LOG"; then
                echo "Shadow-blend-mom run blocked by expected buy-side preflight gate (non-fatal, rc=$SHADOW_BLEND_MOM_RC) — see $SHADOW_BLEND_MOM_LOG"
                echo "Shadow-blend-mom preflight-block ntfy suppressed; prod path already reported the actionable gate."
            elif [ "${RENQUANT_SHADOW_ALERT_NTFY:-1}" != "0" ]; then
                echo "Shadow-blend-mom run FAILED (non-fatal, rc=$SHADOW_BLEND_MOM_RC) — see $SHADOW_BLEND_MOM_LOG"
                notify "RenQuant 104 SHADOW-BLEND-MOM-FAIL" "Shadow-blend-mom e2e failed today (rc=$SHADOW_BLEND_MOM_RC) — primary already completed. See $SHADOW_BLEND_MOM_LOG."
            else
                echo "Shadow-blend-mom run FAILED (non-fatal, rc=$SHADOW_BLEND_MOM_RC) — see $SHADOW_BLEND_MOM_LOG"
                echo "Shadow-blend-mom failure ntfy suppressed (RENQUANT_SHADOW_ALERT_NTFY=0)."
            fi
        fi
    fi
else
    echo "INFO: strategy_config.shadow_blend_momentum.json not present in pinned strategy configs ($SUBREPO_ROOT/renquant-strategy-104/configs) — Step 5b shadow-blend-mom skipped (rail dormant until the S1 profile lands)."
fi

# ── Step 5c: SHADOW-BLEND-MOM-FAST e2e run (GOAL-9 F2, 2026-08-04) ────────
# F2 lane: z(prod) + z(FAST momentum), orch#794 AC2 — cloned from Step 5b
# (the maintained pattern) with the lane's own profile/tag/log/timeout
# names. Tag alpaca_shadow_blend_mom_fast registered AT BIRTH
# (pipeline#265); profile landed s104#89 with a bounded
# pending-first-artifact marker: until the first fast artifact publishes
# (Saturday 2026-08-08), the blend loader fail-closes on the absent
# component — that is the DESIGNED dormant daily record, non-fatal here.
echo "--- Step 5c: Shadow-blend-mom-fast e2e run (F2 FAST momentum blend profile, no real orders) ---"
if BLEND_MOM_FAST_STRATEGY_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.shadow_blend_momentum_fast.json)"; then
    echo "shadow_blend_momentum_fast profile found at $BLEND_MOM_FAST_STRATEGY_CONFIG"
    SHADOW_BLEND_MOM_FAST_LOG="$LOG_DIR/${DATE}_shadow_blend_mom_fast.log"
    SHADOW_BLEND_MOM_FAST_TIMEOUT_SEC="${RENQUANT_SHADOW_BLEND_MOM_FAST_TIMEOUT_SEC:-${RENQUANT_SHADOW_TIMEOUT_SEC:-1800}}"
    if RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1 RENQUANT_READONLY_TAG=alpaca_shadow_blend_mom_fast "$PYTHON" - <<PY > "$SHADOW_BLEND_MOM_FAST_LOG" 2>&1
import os
import subprocess
import sys

if os.environ.get("RQ_DAILY_RUNNER", "multirepo") == "umbrella":
    runner = [sys.executable, "-m", "live.runner"]
else:
    runner = [sys.executable, "-m", "renquant_orchestrator", "live-bridge", "--repo-dir", "$REPO_DIR"]

cmd = runner + [
    "--strategy", "renquant_104",
    "--broker", "readonly-alpaca",
    "--once",
    "--strategy-config-path", "$BLEND_MOM_FAST_STRATEGY_CONFIG",
]
try:
    raise SystemExit(subprocess.run(
        cmd,
        cwd="$REPO_DIR",
        timeout=float("$SHADOW_BLEND_MOM_FAST_TIMEOUT_SEC"),
    ).returncode)
except subprocess.TimeoutExpired:
    print("SHADOW-BLEND-MOM-FAST TIMEOUT after ${SHADOW_BLEND_MOM_FAST_TIMEOUT_SEC}s", flush=True)
    raise SystemExit(124)
PY
    then
        echo "Shadow-blend-mom-fast run finished — see $SHADOW_BLEND_MOM_FAST_LOG"
        grep "ntfy sent:" "$SHADOW_BLEND_MOM_FAST_LOG" | tail -1 || echo "shadow-blend-mom-fast ntfy line not found in shadow-blend-mom-fast log"
    else
        SHADOW_BLEND_MOM_FAST_RC=$?
        if [ "$SHADOW_BLEND_MOM_FAST_RC" -eq 124 ]; then
            echo "Shadow-blend-mom-fast run TIMED OUT after ${SHADOW_BLEND_MOM_FAST_TIMEOUT_SEC}s (non-fatal) — see $SHADOW_BLEND_MOM_FAST_LOG"
            if [ "${RENQUANT_SHADOW_ALERT_NTFY:-1}" != "0" ]; then
                notify "RenQuant 104 SHADOW-BLEND-MOM-FAST-TIMEOUT" "Shadow-blend-mom-fast e2e exceeded ${SHADOW_BLEND_MOM_FAST_TIMEOUT_SEC}s; primary already completed. See $SHADOW_BLEND_MOM_FAST_LOG."
            else
                echo "Shadow-blend-mom-fast timeout ntfy suppressed (RENQUANT_SHADOW_ALERT_NTFY=0)."
            fi
        else
            SHADOW_BLEND_MOM_FAST_BUY_SIDE_PREFLIGHT_PATTERN="P-WF-GATE|P-REGIME-IC|P-CONFIG-FP|P-SECTOR-MAP|P-PANEL-CONTRACT|P-CALIBRATOR-HEALTH|P-CALIBRATOR-FLAT-REGION|P-FEATURE-COVER|P-WATCHLIST|P-MODEL-ARTIFACT|P-CORR-METADATA|P-BEST-ITER|P-RUN-ID|P-META-LABEL|P-FUND-FRESHNESS"
            if grep -Eq "$SHADOW_BLEND_MOM_FAST_BUY_SIDE_PREFLIGHT_PATTERN" "$SHADOW_BLEND_MOM_FAST_LOG"; then
                echo "Shadow-blend-mom-fast run blocked by expected buy-side preflight gate (non-fatal, rc=$SHADOW_BLEND_MOM_FAST_RC) — see $SHADOW_BLEND_MOM_FAST_LOG"
                echo "Shadow-blend-mom-fast preflight-block ntfy suppressed; prod path already reported the actionable gate."
            elif [ "${RENQUANT_SHADOW_ALERT_NTFY:-1}" != "0" ]; then
                echo "Shadow-blend-mom-fast run FAILED (non-fatal, rc=$SHADOW_BLEND_MOM_FAST_RC) — see $SHADOW_BLEND_MOM_FAST_LOG"
                notify "RenQuant 104 SHADOW-BLEND-MOM-FAST-FAIL" "Shadow-blend-mom-fast e2e failed today (rc=$SHADOW_BLEND_MOM_FAST_RC) — primary already completed. See $SHADOW_BLEND_MOM_FAST_LOG."
            else
                echo "Shadow-blend-mom-fast run FAILED (non-fatal, rc=$SHADOW_BLEND_MOM_FAST_RC) — see $SHADOW_BLEND_MOM_FAST_LOG"
                echo "Shadow-blend-mom-fast failure ntfy suppressed (RENQUANT_SHADOW_ALERT_NTFY=0)."
            fi
        fi
    fi
else
    echo "INFO: strategy_config.shadow_blend_momentum_fast.json not present in pinned strategy configs ($SUBREPO_ROOT/renquant-strategy-104/configs) — Step 5c shadow-blend-mom-fast skipped (rail dormant until the F2 profile lands)."
fi

# ── Step 5d/5e: GOAL-9 F1/F3 rails (orch#794 AC2, 2026-08-04) ─────────────
# The two 3-component fleet lanes (pipeline#267 N-generalization; profiles
# s104#90; tags registered at birth pipeline#265). Same clone pattern and
# dormancy semantics as Step 5c; F3's fast leg stays fail-closed until the
# 2026-08-08 genesis batch.
echo "--- Step 5d: Shadow-blend-rb-mom e2e run (F1 rev-blend+slow profile, no real orders) ---"
if BLEND_RB_MOM_STRATEGY_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.shadow_blend_rb_mom.json)"; then
    echo "shadow_blend_rb_mom profile found at $BLEND_RB_MOM_STRATEGY_CONFIG"
    SHADOW_BLEND_RB_MOM_LOG="$LOG_DIR/${DATE}_shadow_blend_rb_mom.log"
    SHADOW_BLEND_RB_MOM_TIMEOUT_SEC="${RENQUANT_SHADOW_BLEND_RB_MOM_TIMEOUT_SEC:-${RENQUANT_SHADOW_TIMEOUT_SEC:-1800}}"
    if RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1 RENQUANT_READONLY_TAG=alpaca_shadow_blend_rb_mom "$PYTHON" - <<PY > "$SHADOW_BLEND_RB_MOM_LOG" 2>&1
import os
import subprocess
import sys

if os.environ.get("RQ_DAILY_RUNNER", "multirepo") == "umbrella":
    runner = [sys.executable, "-m", "live.runner"]
else:
    runner = [sys.executable, "-m", "renquant_orchestrator", "live-bridge", "--repo-dir", "$REPO_DIR"]

cmd = runner + [
    "--strategy", "renquant_104",
    "--broker", "readonly-alpaca",
    "--once",
    "--strategy-config-path", "$BLEND_RB_MOM_STRATEGY_CONFIG",
]
try:
    raise SystemExit(subprocess.run(
        cmd,
        cwd="$REPO_DIR",
        timeout=float("$SHADOW_BLEND_RB_MOM_TIMEOUT_SEC"),
    ).returncode)
except subprocess.TimeoutExpired:
    print("SHADOW-BLEND-RB-MOM TIMEOUT after ${SHADOW_BLEND_RB_MOM_TIMEOUT_SEC}s", flush=True)
    raise SystemExit(124)
PY
    then
        echo "Shadow-blend-rb-mom run finished — see $SHADOW_BLEND_RB_MOM_LOG"
        grep "ntfy sent:" "$SHADOW_BLEND_RB_MOM_LOG" | tail -1 || echo "shadow_blend_rb_mom ntfy line not found in shadow_blend_rb_mom log"
    else
        SHADOW_BLEND_RB_MOM_RC=$?
        if [ "$SHADOW_BLEND_RB_MOM_RC" -eq 124 ]; then
            echo "Shadow-blend-rb-mom run TIMED OUT after ${SHADOW_BLEND_RB_MOM_TIMEOUT_SEC}s (non-fatal) — see $SHADOW_BLEND_RB_MOM_LOG"
            if [ "${RENQUANT_SHADOW_ALERT_NTFY:-1}" != "0" ]; then
                notify "RenQuant 104 SHADOW-BLEND-RB-MOM-TIMEOUT" "Shadow-blend-rb-mom e2e exceeded ${SHADOW_BLEND_RB_MOM_TIMEOUT_SEC}s; primary already completed. See $SHADOW_BLEND_RB_MOM_LOG."
            else
                echo "Shadow-blend-rb-mom timeout ntfy suppressed (RENQUANT_SHADOW_ALERT_NTFY=0)."
            fi
        else
            SHADOW_BLEND_RB_MOM_BUY_SIDE_PREFLIGHT_PATTERN="P-WF-GATE|P-REGIME-IC|P-CONFIG-FP|P-SECTOR-MAP|P-PANEL-CONTRACT|P-CALIBRATOR-HEALTH|P-CALIBRATOR-FLAT-REGION|P-FEATURE-COVER|P-WATCHLIST|P-MODEL-ARTIFACT|P-CORR-METADATA|P-BEST-ITER|P-RUN-ID|P-META-LABEL|P-FUND-FRESHNESS"
            if grep -Eq "$SHADOW_BLEND_RB_MOM_BUY_SIDE_PREFLIGHT_PATTERN" "$SHADOW_BLEND_RB_MOM_LOG"; then
                echo "Shadow-blend-rb-mom run blocked by expected buy-side preflight gate (non-fatal, rc=$SHADOW_BLEND_RB_MOM_RC) — see $SHADOW_BLEND_RB_MOM_LOG"
                echo "Shadow-blend-rb-mom preflight-block ntfy suppressed; prod path already reported the actionable gate."
            elif [ "${RENQUANT_SHADOW_ALERT_NTFY:-1}" != "0" ]; then
                echo "Shadow-blend-rb-mom run FAILED (non-fatal, rc=$SHADOW_BLEND_RB_MOM_RC) — see $SHADOW_BLEND_RB_MOM_LOG"
                notify "RenQuant 104 SHADOW-BLEND-RB-MOM-FAIL" "Shadow-blend-rb-mom e2e failed today (rc=$SHADOW_BLEND_RB_MOM_RC) — primary already completed. See $SHADOW_BLEND_RB_MOM_LOG."
            else
                echo "Shadow-blend-rb-mom run FAILED (non-fatal, rc=$SHADOW_BLEND_RB_MOM_RC) — see $SHADOW_BLEND_RB_MOM_LOG"
                echo "Shadow-blend-rb-mom failure ntfy suppressed (RENQUANT_SHADOW_ALERT_NTFY=0)."
            fi
        fi
    fi
else
    echo "INFO: strategy_config.shadow_blend_rb_mom.json not present in pinned strategy configs ($SUBREPO_ROOT/renquant-strategy-104/configs) — Step 5d shadow_blend_rb_mom skipped (rail dormant until the F1 profile lands)."
fi

echo "--- Step 5e: Shadow-blend-rb-fast e2e run (F3 rev-blend+fast profile, no real orders) ---"
if BLEND_RB_FAST_STRATEGY_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.shadow_blend_rb_fast.json)"; then
    echo "shadow_blend_rb_fast profile found at $BLEND_RB_FAST_STRATEGY_CONFIG"
    SHADOW_BLEND_RB_FAST_LOG="$LOG_DIR/${DATE}_shadow_blend_rb_fast.log"
    SHADOW_BLEND_RB_FAST_TIMEOUT_SEC="${RENQUANT_SHADOW_BLEND_RB_FAST_TIMEOUT_SEC:-${RENQUANT_SHADOW_TIMEOUT_SEC:-1800}}"
    if RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1 RENQUANT_READONLY_TAG=alpaca_shadow_blend_rb_fast "$PYTHON" - <<PY > "$SHADOW_BLEND_RB_FAST_LOG" 2>&1
import os
import subprocess
import sys

if os.environ.get("RQ_DAILY_RUNNER", "multirepo") == "umbrella":
    runner = [sys.executable, "-m", "live.runner"]
else:
    runner = [sys.executable, "-m", "renquant_orchestrator", "live-bridge", "--repo-dir", "$REPO_DIR"]

cmd = runner + [
    "--strategy", "renquant_104",
    "--broker", "readonly-alpaca",
    "--once",
    "--strategy-config-path", "$BLEND_RB_FAST_STRATEGY_CONFIG",
]
try:
    raise SystemExit(subprocess.run(
        cmd,
        cwd="$REPO_DIR",
        timeout=float("$SHADOW_BLEND_RB_FAST_TIMEOUT_SEC"),
    ).returncode)
except subprocess.TimeoutExpired:
    print("SHADOW-BLEND-RB-FAST TIMEOUT after ${SHADOW_BLEND_RB_FAST_TIMEOUT_SEC}s", flush=True)
    raise SystemExit(124)
PY
    then
        echo "Shadow-blend-rb-fast run finished — see $SHADOW_BLEND_RB_FAST_LOG"
        grep "ntfy sent:" "$SHADOW_BLEND_RB_FAST_LOG" | tail -1 || echo "shadow_blend_rb_fast ntfy line not found in shadow_blend_rb_fast log"
    else
        SHADOW_BLEND_RB_FAST_RC=$?
        if [ "$SHADOW_BLEND_RB_FAST_RC" -eq 124 ]; then
            echo "Shadow-blend-rb-fast run TIMED OUT after ${SHADOW_BLEND_RB_FAST_TIMEOUT_SEC}s (non-fatal) — see $SHADOW_BLEND_RB_FAST_LOG"
            if [ "${RENQUANT_SHADOW_ALERT_NTFY:-1}" != "0" ]; then
                notify "RenQuant 104 SHADOW-BLEND-RB-FAST-TIMEOUT" "Shadow-blend-rb-fast e2e exceeded ${SHADOW_BLEND_RB_FAST_TIMEOUT_SEC}s; primary already completed. See $SHADOW_BLEND_RB_FAST_LOG."
            else
                echo "Shadow-blend-rb-fast timeout ntfy suppressed (RENQUANT_SHADOW_ALERT_NTFY=0)."
            fi
        else
            SHADOW_BLEND_RB_FAST_BUY_SIDE_PREFLIGHT_PATTERN="P-WF-GATE|P-REGIME-IC|P-CONFIG-FP|P-SECTOR-MAP|P-PANEL-CONTRACT|P-CALIBRATOR-HEALTH|P-CALIBRATOR-FLAT-REGION|P-FEATURE-COVER|P-WATCHLIST|P-MODEL-ARTIFACT|P-CORR-METADATA|P-BEST-ITER|P-RUN-ID|P-META-LABEL|P-FUND-FRESHNESS"
            if grep -Eq "$SHADOW_BLEND_RB_FAST_BUY_SIDE_PREFLIGHT_PATTERN" "$SHADOW_BLEND_RB_FAST_LOG"; then
                echo "Shadow-blend-rb-fast run blocked by expected buy-side preflight gate (non-fatal, rc=$SHADOW_BLEND_RB_FAST_RC) — see $SHADOW_BLEND_RB_FAST_LOG"
                echo "Shadow-blend-rb-fast preflight-block ntfy suppressed; prod path already reported the actionable gate."
            elif [ "${RENQUANT_SHADOW_ALERT_NTFY:-1}" != "0" ]; then
                echo "Shadow-blend-rb-fast run FAILED (non-fatal, rc=$SHADOW_BLEND_RB_FAST_RC) — see $SHADOW_BLEND_RB_FAST_LOG"
                notify "RenQuant 104 SHADOW-BLEND-RB-FAST-FAIL" "Shadow-blend-rb-fast e2e failed today (rc=$SHADOW_BLEND_RB_FAST_RC) — primary already completed. See $SHADOW_BLEND_RB_FAST_LOG."
            else
                echo "Shadow-blend-rb-fast run FAILED (non-fatal, rc=$SHADOW_BLEND_RB_FAST_RC) — see $SHADOW_BLEND_RB_FAST_LOG"
                echo "Shadow-blend-rb-fast failure ntfy suppressed (RENQUANT_SHADOW_ALERT_NTFY=0)."
            fi
        fi
    fi
else
    echo "INFO: strategy_config.shadow_blend_rb_fast.json not present in pinned strategy configs ($SUBREPO_ROOT/renquant-strategy-104/configs) — Step 5e shadow_blend_rb_fast skipped (rail dormant until the F3 profile lands)."
fi

# ── Step 5f: SHADOW-VOL-WINDOW e2e run (orch#1004 impl PR 2, 2026-08-18) ──
# The vol-window license lane: the CONFIRMED vol-switch conditional
# (orch#1003, frozen prereg orch#1001, approved design orch#1004) running
# shadow-first. Cloned from Step 5e (Step 5 is the maintained pattern) with
# the lane's own profile/tag/log/timeout names. The lane's config
# (strategy_config.shadow_vol_window.json, s104#99) pins the CERTIFIED
# solo-xgb scorer, re-arms regime_admission, and enables the pipeline#294
# vol_window_license flag — inside ON ∧ ¬BEAR the top-decile keeps buy
# admissibility, and the lane's per-session ledger row
# (logs/vol_window_license.jsonl) accrues the activation evidence the
# orchestrator readout counts (design AC3). Never submits: readonly-alpaca
# + RENQUANT_READONLY_TAG=alpaca_shadow_vol_window, state sinks disjoint
# from prod and every other lane. Dormant until the pinned s104 checkout
# carries the profile (the pin advance is a separate operator step); at
# current calm vol the lane may idle ON-less for weeks — correct behavior,
# not a defect (design §4). Kill switch: RENQUANT_VOL_WINDOW_LICENSE_DISABLE
# (lane-scoped, pipeline#294 AC4) — export it into this wrapper's env to
# force the license inactive while the session row still records it.
echo "--- Step 5f: Shadow-vol-window e2e run (vol-window license lane, no real orders) ---"
if VOL_WINDOW_STRATEGY_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.shadow_vol_window.json)"; then
    echo "shadow_vol_window profile found at $VOL_WINDOW_STRATEGY_CONFIG"
    SHADOW_VOL_WINDOW_LOG="$LOG_DIR/${DATE}_shadow_vol_window.log"
    SHADOW_VOL_WINDOW_TIMEOUT_SEC="${RENQUANT_SHADOW_VOL_WINDOW_TIMEOUT_SEC:-${RENQUANT_SHADOW_TIMEOUT_SEC:-1800}}"
    if RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1 RENQUANT_READONLY_TAG=alpaca_shadow_vol_window "$PYTHON" - <<PY > "$SHADOW_VOL_WINDOW_LOG" 2>&1
import os
import subprocess
import sys

if os.environ.get("RQ_DAILY_RUNNER", "multirepo") == "umbrella":
    runner = [sys.executable, "-m", "live.runner"]
else:
    runner = [sys.executable, "-m", "renquant_orchestrator", "live-bridge", "--repo-dir", "$REPO_DIR"]

cmd = runner + [
    "--strategy", "renquant_104",
    "--broker", "readonly-alpaca",
    "--once",
    "--strategy-config-path", "$VOL_WINDOW_STRATEGY_CONFIG",
]
try:
    raise SystemExit(subprocess.run(
        cmd,
        cwd="$REPO_DIR",
        timeout=float("$SHADOW_VOL_WINDOW_TIMEOUT_SEC"),
    ).returncode)
except subprocess.TimeoutExpired:
    print("SHADOW-VOL-WINDOW TIMEOUT after ${SHADOW_VOL_WINDOW_TIMEOUT_SEC}s", flush=True)
    raise SystemExit(124)
PY
    then
        echo "Shadow-vol-window run finished — see $SHADOW_VOL_WINDOW_LOG"
        grep "ntfy sent:" "$SHADOW_VOL_WINDOW_LOG" | tail -1 || echo "shadow_vol_window ntfy line not found in shadow_vol_window log"
    else
        SHADOW_VOL_WINDOW_RC=$?
        if [ "$SHADOW_VOL_WINDOW_RC" -eq 124 ]; then
            echo "Shadow-vol-window run TIMED OUT after ${SHADOW_VOL_WINDOW_TIMEOUT_SEC}s (non-fatal) — see $SHADOW_VOL_WINDOW_LOG"
            if [ "${RENQUANT_SHADOW_ALERT_NTFY:-1}" != "0" ]; then
                notify "RenQuant 104 SHADOW-VOL-WINDOW-TIMEOUT" "Shadow-vol-window e2e exceeded ${SHADOW_VOL_WINDOW_TIMEOUT_SEC}s; primary already completed. See $SHADOW_VOL_WINDOW_LOG."
            else
                echo "Shadow-vol-window timeout ntfy suppressed (RENQUANT_SHADOW_ALERT_NTFY=0)."
            fi
        else
            SHADOW_VOL_WINDOW_BUY_SIDE_PREFLIGHT_PATTERN="P-WF-GATE|P-REGIME-IC|P-CONFIG-FP|P-SECTOR-MAP|P-PANEL-CONTRACT|P-CALIBRATOR-HEALTH|P-CALIBRATOR-FLAT-REGION|P-FEATURE-COVER|P-WATCHLIST|P-MODEL-ARTIFACT|P-CORR-METADATA|P-BEST-ITER|P-RUN-ID|P-META-LABEL|P-FUND-FRESHNESS"
            if grep -Eq "$SHADOW_VOL_WINDOW_BUY_SIDE_PREFLIGHT_PATTERN" "$SHADOW_VOL_WINDOW_LOG"; then
                echo "Shadow-vol-window run blocked by expected buy-side preflight gate (non-fatal, rc=$SHADOW_VOL_WINDOW_RC) — see $SHADOW_VOL_WINDOW_LOG"
                echo "Shadow-vol-window preflight-block ntfy suppressed; prod path already reported the actionable gate."
            elif [ "${RENQUANT_SHADOW_ALERT_NTFY:-1}" != "0" ]; then
                echo "Shadow-vol-window run FAILED (non-fatal, rc=$SHADOW_VOL_WINDOW_RC) — see $SHADOW_VOL_WINDOW_LOG"
                notify "RenQuant 104 SHADOW-VOL-WINDOW-FAIL" "Shadow-vol-window e2e failed today (rc=$SHADOW_VOL_WINDOW_RC) — primary already completed. See $SHADOW_VOL_WINDOW_LOG."
            else
                echo "Shadow-vol-window run FAILED (non-fatal, rc=$SHADOW_VOL_WINDOW_RC) — see $SHADOW_VOL_WINDOW_LOG"
                echo "Shadow-vol-window failure ntfy suppressed (RENQUANT_SHADOW_ALERT_NTFY=0)."
            fi
        fi
    fi
else
    echo "INFO: strategy_config.shadow_vol_window.json not present in pinned strategy configs ($SUBREPO_ROOT/renquant-strategy-104/configs) — Step 5f shadow_vol_window skipped (rail dormant until the vol-window profile lands)."
fi

# ── Step 6: FLEET LANE SENTINEL (GOAL-1, orch#801) ────────────────────────
# The watcher for the five fleet e2e lanes (RC/RSs/Rf/RCS/RCf). It runs HERE,
# as the daily wrapper's last step, for a measured reason: the lanes it
# inspects are Steps 5–5e above, so daily completion IS the correct trigger.
# (Step 5f, the vol-window lane, is NOT in this watch set — its session
# accounting is the orchestrator vol-window readout's parity alarm, which
# cross-checks the lane's license ledger against its runs DB per session.)
# A clock-scheduled job would have to guess a cadence, and
# the first attempt at guessing one (15:30 PT) was derived from a MANUAL
# run's wall clock and would have paged MISSING on a still-running fleet
# (codex on orch#801).
#
# NON-FATAL by construction: every decision above is already made and
# executed; a watcher must never turn its own finding into a failed daily
# run. Its alarm channel is the wrapper's own ntfy, not this exit code.
# The wrapper logic lives in renquant-orchestrator (daily orchestration is
# its declared role); the umbrella only calls it, with the session date
# passed EXPLICITLY so a post-midnight finish still classifies its own
# session.
echo "--- Step 6: Fleet lane sentinel (GOAL-9 lanes RC/RSs/Rf/RCS/RCf) ---"
FLEET_SENTINEL="${RQ_ORCH_RUN_DIR:-/Users/renhao/git/github/renquant-orchestrator-run}/ops/renquant104/fleet_lane_sentinel_daily.sh"
if [ -x "$FLEET_SENTINEL" ] || [ -f "$FLEET_SENTINEL" ]; then
    if bash "$FLEET_SENTINEL" "$DATE"; then
        echo "Fleet lane sentinel: all lanes accounted for."
    else
        FLEET_SENTINEL_RC=$?
        echo "Fleet lane sentinel reported actionable lane state(s) (non-fatal, rc=$FLEET_SENTINEL_RC) — it has already paged; see $LOG_DIR/../rq104/fleet_lane_sentinel_${DATE}.log"
    fi
else
    # ABSENCE IS ACTIONABLE (codex on RQ#582). A stale or failed run-checkout
    # deploy can REMOVE the fleet watcher, and an INFO line about it is exactly
    # the silence this watcher exists to end: nobody else pages here, because
    # the missing component IS the pager. This is the ONE branch in Step 6 that
    # notifies — a sentinel FINDING is already paged by the wrapper itself and
    # must not be double-sent.
    echo "Fleet lane sentinel MISSING from the orchestrator run checkout ($FLEET_SENTINEL) — the fleet lanes are UNWATCHED today (non-fatal for this run; sync the run checkout to a pin carrying orch#801)."
    notify "RenQuant 104 FLEET-SENTINEL-MISSING" \
        "The fleet lane watcher is absent from $FLEET_SENTINEL — the five blend lanes ran UNWATCHED on $DATE. Sync the orchestrator run checkout to a pin carrying orch#801."
fi
