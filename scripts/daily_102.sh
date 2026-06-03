#!/usr/bin/env bash
# daily_102.sh — Run renquant_102 notebook (retrain models) then live trade via Alpaca.
# Designed for launchd/cron on macOS. Runs unattended.
set -uo pipefail

if [ "${RQ_LEGACY_102_DAILY_ENABLED:-0}" != "1" ]; then
    echo "ERROR: daily_102.sh is legacy rollback-only; set RQ_LEGACY_102_DAILY_ENABLED=1 to run."
    exit 2
fi

REPO_DIR="/Users/renhao/git/github/RenQuant"
CONDA_PREFIX="/Users/renhao/miniconda3/envs/renquant"
PYTHON="$CONDA_PREFIX/bin/python"
JUPYTER="$CONDA_PREFIX/bin/jupyter"
LOG_DIR="$REPO_DIR/logs/daily_102"
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
echo "=== daily_102 started at $(date) ==="

# Step 1: Run the 102 notebook to retrain all models
echo "--- Step 1: Running renquant_102 notebook ---"
cd "$REPO_DIR"
if "$JUPYTER" nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=1800 \
    --output /tmp/renquant_102_executed.ipynb \
    Notebooks/renquant_102.ipynb; then
    echo "Notebook finished at $(date)"
else
    echo "Notebook FAILED at $(date)"
    notify "RenQuant 102 ERROR" "Notebook failed — check $LOG"
    exit 1
fi

# Step 1b: Validate model count — alert if too few models exported
MIN_MODELS=10
MODEL_COUNT=$("$PYTHON" -c "
import json
from pathlib import Path
models_dir = Path('$REPO_DIR/backtesting/renquant_102/models')
watchlist = json.loads(Path('$REPO_DIR/backtesting/renquant_102/strategy_config.json').read_text())['watchlist']
count = sum(1 for s in watchlist if (models_dir / s / f'{s}-policy-metadata.json').exists())
print(count)
" 2>/dev/null || echo "0")
echo "Models exported: $MODEL_COUNT / $(python3 -c "import json; print(len(json.loads(open('$REPO_DIR/backtesting/renquant_102/strategy_config.json').read())['watchlist']))" 2>/dev/null || echo "?")"
if [ "${MODEL_COUNT:-0}" -lt "$MIN_MODELS" ] 2>/dev/null; then
    notify "RenQuant 102 WARN" "Only $MODEL_COUNT models exported (min=$MIN_MODELS) — check OOS Sharpe floor"
else
    notify "RenQuant 102" "Models retrained: $MODEL_COUNT watchlist models ready"
fi

# Step 2: Export LEAN data for all watchlist symbols
echo "--- Step 2: Exporting LEAN watchlist data ---"
if "$PYTHON" scripts/export_lean_watchlist.py --strategy renquant_102; then
    echo "LEAN data export finished at $(date)"
else
    echo "LEAN data export FAILED at $(date)"
    notify "RenQuant 102 ERROR" "LEAN data export failed — check $LOG"
    exit 1
fi

# Step 3: Run live trading (Alpaca, single pass)
echo "--- Step 3: Running live trader (alpaca) ---"
TRADE_LOG="$REPO_DIR/live/logs/renquant-102/$DATE.json"
if "$PYTHON" -m live.runner --strategy renquant_102 --broker alpaca --once; then
    echo "=== daily_102 finished at $(date) ==="
    # Build trade summary from today's trade log
    SUMMARY=$("$PYTHON" -c "
import json, sys
from pathlib import Path
log_path = Path('$TRADE_LOG')
if not log_path.exists():
    print('No trades today')
    sys.exit(0)
trades = json.loads(log_path.read_text())
buys = [t for t in trades if t.get('signal') == 'buy']
sells = [t for t in trades if t.get('signal') == 'sell']
stops = [t for t in trades if t.get('signal') == 'stop_loss']
parts = []
for t in buys:
    sym = t.get('symbol', '?')
    score = t.get('volume_score', t.get('volume_zscore', 0))
    order = t.get('order', {})
    qty = order.get('qty', '?')
    parts.append(f'BUY {sym} x{qty}')
for t in sells:
    sym = t.get('symbol', '?')
    order = t.get('order', {})
    qty = order.get('qty', '?')
    parts.append(f'SELL {sym} x{qty}')
for t in stops:
    sym = t.get('symbol', '?')
    loss = t.get('loss_pct', 0)
    parts.append(f'STOP {sym} ({loss:.1%} loss)')
if parts:
    print('; '.join(parts))
else:
    print('No trades today')
" 2>/dev/null || echo "No trades today")
    notify "RenQuant 102" "$SUMMARY"
else
    echo "=== daily_102 FAILED at $(date) ==="
    notify "RenQuant 102 ERROR" "Live trader failed — check $LOG"
    exit 1
fi
