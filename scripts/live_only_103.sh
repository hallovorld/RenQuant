#!/usr/bin/env bash
# live_only_103.sh — Intraday sell check for renquant_103.
#
# Runs the live trader WITHOUT model retraining or LEAN data export.
# Used for the two intraday triggers:
#   • market-open run  (6:32 AM PT) — catch overnight gap-downs early
#   • pre-close run   (12:44 PM PT) — exit intraday stop breaches before close
#
# Usage:
#   bash scripts/live_only_103.sh [--sell-only]
#
# --sell-only  (default)  Skip the buy phase; process exits only.
#              Omit to run a full buy+sell cycle (not recommended for intraday).
#
# Logs to logs/live_103/{date}-{tag}.log where tag is "open" or "preclose"
# based on the current hour (before 10 AM = "open", otherwise "preclose").
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
CONDA_PREFIX="/Users/renhao/miniconda3/envs/renquant"
PYTHON="$CONDA_PREFIX/bin/python"
LOG_DIR="$REPO_DIR/logs/live_103"
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
echo "=== live_103 [$TAG] started at $(date) ==="

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

TRADE_LOG="$REPO_DIR/live/logs/renquant-103/$DATE.json"

if "$PYTHON" -m live.runner \
    --strategy renquant_103 \
    --broker alpaca \
    --once \
    $SELL_ONLY_FLAG; then

    echo "=== live_103 [$TAG] finished at $(date) ==="

    # Build trade summary (sells/stops only for intraday runs)
    SUMMARY=$("$PYTHON" -c "
import json, sys
from pathlib import Path
log_path = Path('$TRADE_LOG')
if not log_path.exists():
    print('No exits today')
    sys.exit(0)
trades = json.loads(log_path.read_text())
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
    elif sig == 'sell':
        parts.append(f'SELL {sym} x{qty}')
print('; '.join(parts) if parts else 'No exits')
" 2>/dev/null || echo "No exits")

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
    notify "RenQuant 103 [$TAG]" "$FULL_MSG"
else
    echo "=== live_103 [$TAG] FAILED at $(date) ==="
    notify "RenQuant 103 ERROR [$TAG]" "Live trader failed — check $LOG"
    exit 1
fi
