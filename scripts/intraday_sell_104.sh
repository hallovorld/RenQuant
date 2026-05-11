#!/usr/bin/env bash
# intraday_sell_104.sh — Mid-market-hours exit-only pass with Alpaca 5-min bars.
# ─── PAPER MODE (2026-05-11) ─────────────────────────────────────────────
# 2026-05-11: per user safety mandate, --broker switched from alpaca (LIVE,
# real money via Alpaca live API) to paper (LOCAL sim broker, no external
# venue, records to live_state.paper.json + ntfy). To restore live trading:
#   sed -i "" "s/--broker paper/--broker alpaca/g" scripts/*.sh
# Or add ALPACA_PAPER_API_KEY/SECRET to .env + switch to --broker alpaca-paper
# for Alpaca's paper-trading sandbox (real API, no real money).
# ─────────────────────────────────────────────────────────────────────────
#
# Runs every ~30 min during market hours. Uses Alpaca's IEX 5-min feed to
# overlay the latest intraday close on today's daily bar, then calls
# SellOnlyPipeline → triggers stop-loss / trailing-stop / SDL / max-hold /
# model-sell signals against the fresh price. Never places buys.
#
# Complements daily_104.sh (runs once, after close). Catches gap-down days
# where EOD evaluation would have already locked in the loss.
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
# 2026-05-11 audit M-env: switched conda → .venv per feedback_python_env.md
VENV_DIR="/Users/renhao/git/github/RenQuant/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$REPO_DIR/logs/intraday_104"
NTFY_TOPIC="renquant"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
HHMM=$(date +%H%M)
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
    set -a; source "$CRED_FILE"; set +a
else
    echo "ERROR: $CRED_FILE not found." | tee -a "$LOG"
    exit 1
fi

exec >> "$LOG" 2>&1
echo "=== intraday_sell started at $(date) (HHMM=$HHMM) ==="

# NYSE calendar guard — skip on holidays/weekends
TODAY_DATE=$(date +%Y-%m-%d)
if ! "$PYTHON" -c "
import sys, pandas_market_calendars as mcal
cal = mcal.get_calendar('NYSE')
sched = cal.schedule('$TODAY_DATE', '$TODAY_DATE')
sys.exit(0 if len(sched) > 0 else 1)
"; then
    echo "NYSE closed today ($TODAY_DATE) — skipping."
    exit 0
fi

# Lock file — prevent stepping on daily_104.sh or another intraday run
LOCK_FILE="/tmp/renquant_104_intraday.lock"
if ! ( set -C; echo $$ > "$LOCK_FILE" ) 2>/dev/null; then
    EXISTING_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "?")
    echo "Another intraday_sell is active (PID=$EXISTING_PID) — skipping."
    exit 0
fi
trap "rm -f '$LOCK_FILE'" EXIT

cd "$REPO_DIR"
if "$PYTHON" -m live.runner --strategy renquant_104 --broker paper --once \
        --sell-only --intraday; then
    echo "=== intraday_sell finished at $(date) ==="
else
    echo "=== intraday_sell FAILED at $(date) ==="
    notify "RenQuant 104 intraday ERROR" "Check $LOG"
    exit 1
fi
