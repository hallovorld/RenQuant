#!/usr/bin/env bash
# daily_104.sh — Retrain renquant_104 (tournament + panel-LTR + recalibrate) then
# live trade via Alpaca. Designed for launchd/cron on macOS. Runs unattended.
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
CONDA_PREFIX="/Users/renhao/miniconda3/envs/renquant"
PYTHON="$CONDA_PREFIX/bin/python"
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

exec >> "$LOG" 2>&1
echo "=== daily_104 started at $(date) ==="

# ── Lock file — prevent concurrent invocations ────────────────────────────────
LOCK_FILE="/tmp/renquant_104_daily.lock"
if ! ( set -C; echo $$ > "$LOCK_FILE" ) 2>/dev/null; then
    EXISTING_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "?")
    echo "Another daily_104 run is active (PID=$EXISTING_PID, lock=$LOCK_FILE) — skipping."
    notify "RenQuant 104 SKIP" "Duplicate daily run blocked (PID=$EXISTING_PID already running)"
    exit 0
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
DRIFT_OUT=$("$PYTHON" "$REPO_DIR/scripts/check_config_drift.py" --strategy renquant_104 2>&1 || true)
if echo "$DRIFT_OUT" | grep -q "drift detected"; then
    # Only surface booleans + the first 2 numeric lines in ntfy to keep it short
    SHORT=$(echo "$DRIFT_OUT" | grep -E "→" | head -5 | sed 's/^  *//')
    notify "RenQuant 104 DRIFT" "strategy_config.json drifted from golden — $SHORT"
    echo "$DRIFT_OUT"
else
    echo "Config drift OK."
fi

# Step 1: Run FullTrainingPipeline (baseline tournament → panel-LTR → recalibrate)
echo "--- Step 1: Running renquant_104 FullTrainingPipeline ---"
cd "$REPO_DIR"
if "$PYTHON" scripts/train_104.py --strategy renquant_104; then
    echo "Training pipeline finished at $(date)"
else
    echo "Training pipeline FAILED at $(date)"
    notify "RenQuant 104 ERROR" "Training pipeline failed — check $LOG"
    exit 1
fi

# Step 1b: Validate model count — alert if too few models exported
MIN_MODELS=10
MODEL_COUNT=$("$PYTHON" -c "
import json
from pathlib import Path
models_dir = Path('$REPO_DIR/backtesting/renquant_104/models')
watchlist = json.loads(Path('$REPO_DIR/backtesting/renquant_104/strategy_config.json').read_text())['watchlist']
count = sum(1 for s in watchlist if (models_dir / s / f'{s}-policy-metadata.json').exists())
print(count)
" 2>/dev/null || echo "0")
WATCHLIST_SIZE=$("$PYTHON" -c "import json; print(len(json.loads(open('$REPO_DIR/backtesting/renquant_104/strategy_config.json').read())['watchlist']))" 2>/dev/null || echo "?")
echo "Models exported: $MODEL_COUNT / $WATCHLIST_SIZE"

# Pull panel + ngboost artifact metadata for the notification body so the
# alert also surfaces WHEN the panel was last retrained and how it scored.
# Falls back to "—" when a field is missing.
PANEL_INFO=$("$PYTHON" -c "
import json
from pathlib import Path
adir = Path('$REPO_DIR/backtesting/renquant_104/artifacts')
panel_path = adir / 'panel-ltr.json'
ngb_path   = adir / 'ngboost-head.json'
try:
    p = json.loads(panel_path.read_text())
except Exception:
    p = {}
ic  = p.get('oos_mean_ic')
std = p.get('oos_std_ic')
td  = p.get('trained_date') or '—'
ic_str  = f'{ic:+.4f}'  if isinstance(ic,  (int, float)) else '—'
std_str = f'{std:.4f}' if isinstance(std, (int, float)) else '—'
try:
    n = json.loads(ngb_path.read_text())
    ngb_td = n.get('trained_date') or '—'
    ngb_n  = n.get('metadata', {}).get('n_rows') or n.get('n_rows') or '—'
except Exception:
    ngb_td = '—'; ngb_n = '—'
print(f'panel@{td} IC={ic_str}±{std_str} | ngb@{ngb_td} n={ngb_n}')
" 2>/dev/null || echo "panel info unavailable")

# TTL-skipped count: how many tickers reused their prior artifact today
# (model_ttl_days gate in pp_training.py). Surfaced in ntfy body so we
# can see at a glance whether the run exercised the full tournament or
# reused cached models.
TTL_SKIPPED=$(grep -c "TTL skip" "$LOG" 2>/dev/null || echo 0)

if [ "${MODEL_COUNT:-0}" -lt "$MIN_MODELS" ] 2>/dev/null; then
    notify "RenQuant 104 WARN" "Only $MODEL_COUNT models (min=$MIN_MODELS) — $PANEL_INFO"
else
    # Only fire the "Models retrained" ntfy on the 3 days/week that
    # actually retrain (training.cadence="custom", allowed_weekdays=[1,3,6] →
    # Tue/Thu/Sun). On off-cadence days `train_104.py` short-circuits;
    # panel-ltr.json `trained_date` stays older than today, so we suppress
    # the notification to avoid daily spam. The notification still fires
    # on the 3 retrain days so the user gets the IC / model count.
    RETRAINED_TODAY=$("$PYTHON" -c "
import json, datetime
from pathlib import Path
p = Path('$REPO_DIR/backtesting/renquant_104/artifacts/panel-ltr.json')
try:
    td = json.loads(p.read_text()).get('trained_date', '')
    print('yes' if td == str(datetime.date.today()) else 'no')
except Exception:
    print('no')
" 2>/dev/null || echo "no")
    if [ "$RETRAINED_TODAY" = "yes" ]; then
        TTL_NOTE=""
        if [ "${TTL_SKIPPED:-0}" -gt 0 ] 2>/dev/null; then
            TTL_NOTE=" ($TTL_SKIPPED ticker-TTL skips)"
        fi
        notify "RenQuant 104" "Models retrained: $MODEL_COUNT watchlist models ready${TTL_NOTE} — $PANEL_INFO"
    else
        echo "Models retrained: $MODEL_COUNT models (no retrain today — suppressing ntfy)"
    fi
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
"$PYTHON" scripts/backfill_forward_returns.py --source live 2>&1 | tail -5 || echo "forward_returns backfill failed (non-fatal)"
"$PYTHON" scripts/compute_portfolio_metrics.py --source live --strategy renquant-104 2>&1 | tail -15 || echo "portfolio metrics compute failed (non-fatal)"

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

if "$PYTHON" -m live.runner --strategy renquant_104 --broker alpaca --once; then
    echo "=== daily_104 finished at $(date) ==="

    # Build trade summary from THIS run's new entries only
    SUMMARY=$("$PYTHON" -c "
import json, sys
from pathlib import Path
log_path = Path('$TRADE_LOG')
pre_count = $PRE_COUNT
if not log_path.exists():
    print('No trades this run')
    sys.exit(0)
try:
    all_trades = json.loads(log_path.read_text())
except Exception:
    print('No trades this run')
    sys.exit(0)
trades = all_trades[pre_count:]  # only entries from this run
parts = []
for t in trades:
    sig = t.get('signal', '')
    sym = t.get('symbol', '?')
    order = t.get('order', {})
    qty = order.get('qty', '?')
    if sig == 'buy':
        parts.append(f'BUY {sym} x{qty}')
    elif sig in ('sell', 'max_hold'):
        parts.append(f'SELL {sym} x{qty}')
    elif sig == 'stop_loss':
        loss = t.get('loss_pct', 0)
        parts.append(f'STOP {sym} ({loss:.1%})')
    elif sig == 'single_day_loss':
        drop = t.get('daily_drop_pct', 0)
        parts.append(f'GAP-STOP {sym} ({drop:.1%} drop)')
    elif sig == 'trailing_stop':
        parts.append(f'TRAIL-STOP {sym}')
print('; '.join(parts) if parts else 'No trades this run')
" 2>/dev/null || echo "No trades this run")
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
    notify "RenQuant 104" "$FULL_MSG"

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

# Account snapshot from Alpaca
equity = cash = None
n_positions = 0
try:
    from alpaca.trading.client import TradingClient
    client = TradingClient(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_SECRET_KEY'], paper=False)
    acct = client.get_account()
    equity = float(acct.equity)
    cash   = float(acct.cash)
    n_positions = len(client.get_all_positions())
except Exception as exc:
    pass

# Regime + HWM from live_state.json
state_path = Path('$REPO_DIR/backtesting/renquant_104/live_state.json')
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
" 2>&1 | tee -a "$LOG" || echo "audit write failed (non-fatal)"

else
    echo "=== daily_104 FAILED at $(date) ==="
    notify "RenQuant 104 ERROR" "Live trader failed — check $LOG"
    exit 1
fi
