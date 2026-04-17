#!/usr/bin/env bash
# daily_103.sh — Run renquant_103 notebook (retrain models) then live trade via Alpaca.
# Designed for launchd/cron on macOS. Runs unattended.
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
CONDA_PREFIX="/Users/renhao/miniconda3/envs/renquant"
PYTHON="$CONDA_PREFIX/bin/python"
JUPYTER="$CONDA_PREFIX/bin/jupyter"
LOG_DIR="$REPO_DIR/logs/daily_103"
NTFY_TOPIC="renquant"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"

notify() {
    local title="$1" body="$2"
    # macOS banner
    if command -v terminal-notifier &>/dev/null; then
        terminal-notifier -title "$title" -message "$body" -sound Glass 2>/dev/null || true
    fi
    # iPhone push via ntfy.sh
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

exec >> "$LOG" 2>&1
echo "=== daily_103 started at $(date) ==="

# ── Time-of-day guard ─────────────────────────────────────────────────────────
# launchd fires a catch-up run whenever the Mac wakes after a missed schedule.
# A midnight or early-morning catch-up would place DAY orders that execute at
# the NEXT market open at whatever price the stock opens at — impossible to stop-
# loss correctly with stale OHLCV data.  Only allow this script to run between
# 12:00 PM (noon) and 22:00 (10 PM) PT, which covers the scheduled 1:55 PM slot
# plus a generous window for legitimate manual runs, while blocking midnight runs.
CURRENT_HOUR=$(date +%H)
if [ "$CURRENT_HOUR" -lt 12 ] || [ "$CURRENT_HOUR" -ge 22 ]; then
    echo "Outside allowed window [12:00–22:00 PT], hour=$CURRENT_HOUR — skipping (launchd catch-up guard)."
    notify "RenQuant 103 SKIP" "Daily run skipped outside window (hour=$CURRENT_HOUR)"
    exit 0
fi

# ── Lock file — prevent concurrent invocations ────────────────────────────────
LOCK_FILE="/tmp/renquant_103_daily.lock"
if ! ( set -C; echo $$ > "$LOCK_FILE" ) 2>/dev/null; then
    EXISTING_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "?")
    echo "Another daily_103 run is active (PID=$EXISTING_PID, lock=$LOCK_FILE) — skipping."
    notify "RenQuant 103 SKIP" "Duplicate daily run blocked (PID=$EXISTING_PID already running)"
    exit 0
fi
trap "rm -f '$LOCK_FILE'" EXIT

# NYSE calendar guard — skip on market holidays (launchd fires Mon–Fri but not holiday-aware)
TODAY_DATE=$(date +%Y-%m-%d)
if ! "$PYTHON" -c "
import sys, pandas_market_calendars as mcal, pandas as pd
cal = mcal.get_calendar('NYSE')
sched = cal.schedule('$TODAY_DATE', '$TODAY_DATE')
sys.exit(0 if len(sched) > 0 else 1)
"; then
    echo "NYSE closed today ($TODAY_DATE) — skipping run."
    notify "RenQuant 103" "Skipped — NYSE holiday ($TODAY_DATE)"
    exit 0
fi
echo "NYSE open today ($TODAY_DATE) — proceeding."

# Step 1: Run the 103 notebook to retrain all models
echo "--- Step 1: Running renquant_103 notebook ---"
cd "$REPO_DIR"
if "$JUPYTER" nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=3600 \
    --output /tmp/renquant_103_executed.ipynb \
    Notebooks/renquant_103.ipynb; then
    echo "Notebook finished at $(date)"
else
    echo "Notebook FAILED at $(date)"
    notify "RenQuant 103 ERROR" "Notebook failed — check $LOG"
    exit 1
fi

# Step 1b: Validate model count — alert if too few models exported
MIN_MODELS=10
MODEL_COUNT=$("$PYTHON" -c "
import json
from pathlib import Path
models_dir = Path('$REPO_DIR/backtesting/renquant_103/models')
watchlist = json.loads(Path('$REPO_DIR/backtesting/renquant_103/strategy_config.json').read_text())['watchlist']
count = sum(1 for s in watchlist if (models_dir / s / f'{s}-policy-metadata.json').exists())
print(count)
" 2>/dev/null || echo "0")
WATCHLIST_SIZE=$("$PYTHON" -c "import json; print(len(json.loads(open('$REPO_DIR/backtesting/renquant_103/strategy_config.json').read())['watchlist']))" 2>/dev/null || echo "?")
echo "Models exported: $MODEL_COUNT / $WATCHLIST_SIZE"
if [ "${MODEL_COUNT:-0}" -lt "$MIN_MODELS" ] 2>/dev/null; then
    notify "RenQuant 103 WARN" "Only $MODEL_COUNT models exported (min=$MIN_MODELS) — check OOS Sharpe floor"
else
    notify "RenQuant 103" "Models retrained: $MODEL_COUNT watchlist models ready"
fi

# Step 2: Export LEAN data for all watchlist symbols
echo "--- Step 2: Exporting LEAN watchlist data ---"
if "$PYTHON" scripts/export_lean_watchlist.py --strategy renquant_103; then
    echo "LEAN data export finished at $(date)"
else
    echo "LEAN data export FAILED at $(date)"
    notify "RenQuant 103 ERROR" "LEAN data export failed — check $LOG"
    exit 1
fi

# Step 2b: Recalibrate model score calibrations and blend weights
echo "--- Step 2b: Recalibrating model score calibrations ---"
if "$PYTHON" scripts/recalibrate_scores.py --strategy renquant_103; then
    echo "Score recalibration finished at $(date)"
else
    echo "WARNING: Score recalibration failed — continuing with existing calibration" >&2
fi

# Step 3: Run live trading (Alpaca, single pass)
echo "--- Step 3: Running live trader (alpaca) ---"
TRADE_LOG="$REPO_DIR/live/logs/renquant-103/$DATE.json"
if "$PYTHON" -m live.runner --strategy renquant_103 --broker alpaca --once; then
    echo "=== daily_103 finished at $(date) ==="
    # Build trade summary from today's trade log
    SUMMARY=$("$PYTHON" -c "
import json, sys
from pathlib import Path
log_path = Path('$TRADE_LOG')
if not log_path.exists():
    print('No trades today')
    sys.exit(0)
trades = json.loads(log_path.read_text())
parts = []
for t in trades:
    sig = t.get('signal', '')
    sym = t.get('symbol', '?')
    order = t.get('order', {})
    qty = order.get('qty', '?')
    if sig == 'buy':
        parts.append(f'BUY {sym} x{qty}')
    elif sig == 'sell':
        parts.append(f'SELL {sym} x{qty}')
    elif sig == 'stop_loss':
        loss = t.get('loss_pct', 0)
        parts.append(f'STOP {sym} ({loss:.1%})')
    elif sig == 'single_day_loss':
        drop = t.get('daily_drop_pct', 0)
        parts.append(f'GAP-STOP {sym} ({drop:.1%} drop)')
print('; '.join(parts) if parts else 'No trades today')
" 2>/dev/null || echo "No trades today")
    notify "RenQuant 103" "$SUMMARY"
else
    echo "=== daily_103 FAILED at $(date) ==="
    notify "RenQuant 103 ERROR" "Live trader failed — check $LOG"
    exit 1
fi
