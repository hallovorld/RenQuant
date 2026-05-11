#!/usr/bin/env bash
# live_only_104.sh — Intraday sell check for renquant_104.
#
# Runs the live trader WITHOUT model retraining or LEAN data export.
# Used for the two intraday triggers:
#   • market-open run  (6:32 AM PT) — catch overnight gap-downs early
#   • pre-close run   (12:44 PM PT) — exit intraday stop breaches before close
#
# Usage:
#   bash scripts/live_only_104.sh [--sell-only]
#
# --sell-only  (default)  Skip the buy phase; process exits only.
#              Omit to run a full buy+sell cycle (not recommended for intraday).
#
# Logs to logs/live_104/{date}-{tag}.log where tag is "open" or "preclose"
# based on the current hour (before 10 AM = "open", otherwise "preclose").
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
# 2026-05-11 audit M-env: switched conda → .venv per feedback_python_env.md
VENV_DIR="/Users/renhao/git/github/RenQuant/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$REPO_DIR/logs/live_104"
NTFY_TOPIC="renquant"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
HOUR=$(date +%H)
# Determine run tag from hour: before 10 AM = open, otherwise preclose
if [ "$HOUR" -lt 10 ]; then
    TAG="open"
else
    TAG="preclose"
fi
LOG="$LOG_DIR/${DATE}-${TAG}.log"

# Parse optional --sell-only argument (default: sell-only)
SELL_ONLY_FLAG="--sell-only"
for arg in "$@"; do
    if [ "$arg" = "--no-sell-only" ]; then
        SELL_ONLY_FLAG=""
    fi
done

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
    set -a; source "$CRED_FILE"; set +a
else
    echo "ERROR: $CRED_FILE not found." | tee -a "$LOG"
    exit 1
fi

exec >> "$LOG" 2>&1
echo "=== live_104 [$TAG] started at $(date) ==="

# ── Already-ran-today guard (per tag) ────────────────────────────────────────
# launchd may fire multiple times on Mac sleep/wake. Allow only one successful
# run per TAG per day (one "open", one "preclose").
DONE_FILE="/tmp/renquant_104_${TAG}_${DATE}.done"
if [ -f "$DONE_FILE" ]; then
    echo "live_104 [$TAG] already completed today ($DATE) — skipping duplicate run."
    exit 0
fi

# ── Lock file — prevent concurrent invocations of same tag ───────────────────
# Audit fix LOCK-STALE (Round 2 deep audit, 2026-04-25): same dead-PID
# stale-lock recovery as daily_104.sh. Without this, a SIGKILL'd run
# leaves a dead-PID lockfile that silently blocks every subsequent
# 30-minute intraday tick — positions never exit on stop-loss until
# the lock is manually cleared.
LOCK_FILE="/tmp/renquant_104_${TAG}.lock"
_acquire_lock() {
    ( set -C; echo $$ > "$LOCK_FILE" ) 2>/dev/null
}
if ! _acquire_lock; then
    EXISTING_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "?")
    if [ "$EXISTING_PID" != "?" ] && [ -n "$EXISTING_PID" ] && \
            ! kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "Stale lock (PID=$EXISTING_PID is dead) — clearing and retrying."
        rm -f "$LOCK_FILE"
        if ! _acquire_lock; then
            echo "Failed to acquire lock after clearing stale — aborting."
            exit 1
        fi
    else
        echo "live_104 [$TAG] already running (PID=$EXISTING_PID) — skipping."
        exit 0
    fi
fi
trap "rm -f '$LOCK_FILE'" EXIT

# NYSE calendar guard
TODAY_DATE=$(date +%Y-%m-%d)
if ! "$PYTHON" -c "
import sys, pandas_market_calendars as mcal
cal = mcal.get_calendar('NYSE')
sched = cal.schedule('$TODAY_DATE', '$TODAY_DATE')
sys.exit(0 if len(sched) > 0 else 1)
"; then
    echo "NYSE closed today ($TODAY_DATE) — skipping run."
    exit 0
fi
echo "NYSE open — running $TAG sell check."

TRADE_LOG="$REPO_DIR/live/logs/renquant-104/$DATE.json"

# Snapshot trade log length before this run so the notification only shows
# trades placed by THIS run, not by earlier runs from the same day.
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

if "$PYTHON" -m live.runner \
    --strategy renquant_104 \
    --broker alpaca \
    --once \
    $SELL_ONLY_FLAG; then

    echo "=== live_104 [$TAG] finished at $(date) ==="
    touch "$DONE_FILE"

    # Build trade summary (THIS run's new entries only)
    SUMMARY=$("$PYTHON" -c "
import json, sys
from pathlib import Path
log_path = Path('$TRADE_LOG')
pre_count = $PRE_COUNT
if not log_path.exists():
    print('No exits this run')
    sys.exit(0)
try:
    all_trades = json.loads(log_path.read_text())
except Exception:
    print('No exits this run')
    sys.exit(0)
trades = all_trades[pre_count:]  # only entries from this run
parts = []
for t in trades:
    sig = t.get('signal', '')
    sym = t.get('symbol', '?')
    order = t.get('order', {})
    qty = order.get('qty', '?')
    if sig == 'stop_loss':
        loss = t.get('loss_pct', 0)
        parts.append(f'STOP {sym} ({loss:.1%})')
    elif sig == 'single_day_loss':
        drop = t.get('daily_drop_pct', 0)
        parts.append(f'GAP-STOP {sym} ({drop:.1%} drop)')
    elif sig == 'trailing_stop':
        parts.append(f'TRAIL-STOP {sym}')
    elif sig in ('sell', 'max_hold'):
        parts.append(f'SELL {sym} x{qty}')
print('; '.join(parts) if parts else 'No exits this run')
" 2>/dev/null || echo "No exits this run")

    # Append current holdings to notification
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
    FULL_MSG="${SUMMARY}${HOLDINGS:+ | $HOLDINGS}"
    notify "RenQuant 104 [$TAG]" "$FULL_MSG"
else
    echo "=== live_104 [$TAG] FAILED at $(date) ==="
    notify "RenQuant 104 ERROR [$TAG]" "Live trader failed — check $LOG"
    exit 1
fi
