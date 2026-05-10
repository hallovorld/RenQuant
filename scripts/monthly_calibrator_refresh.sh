#!/usr/bin/env bash
# monthly_calibrator_refresh.sh — Re-fit the global panel calibrator monthly.
#
# 2026-05-09 audit FIX-C: separates calibrator refresh (fast, low-risk)
# from model retrain (slow, high-risk, weekly). The calibrator's isotonic
# knot positions can drift as the score distribution shifts (regime
# changes, etc.) even when the underlying XGBoost model is unchanged.
# Monthly refit keeps calibrated probabilities + expected returns aligned
# with current score distribution without touching the model.
#
# Schedule: 1st of every month, 03:00 PT.
# Plist: scripts/launchd/com.renquant.monthly-calibrator-refresh.plist
#
# Steps:
#   1. Smoke test (ensure model still loads — abort if broken)
#   2. Run fit_panel_calibrator.py against the active production model
#   3. Test scorer + new calibrator produces sane (P, E[R]) on synthetic
#      input — abort if calibrator collapsed (n_unique_prob_y < floor)
#   4. ntfy summary — n knots, score → P(out) range
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
CONDA_PREFIX="/Users/renhao/miniconda3/envs/renquant"
PYTHON="$CONDA_PREFIX/bin/python"
LOG_DIR="$REPO_DIR/logs/monthly_calibrator"
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

CRED_FILE="$REPO_DIR/.env"
if [ -f "$CRED_FILE" ]; then
    set -a
    source "$CRED_FILE"
    set +a
fi

exec >> "$LOG" 2>&1
echo "=== monthly_calibrator_refresh started at $(date) ==="

LOCK_FILE="/tmp/renquant_104_monthly_cal.lock"
if ! ( set -C; echo $$ > "$LOCK_FILE" ) 2>/dev/null; then
    EXISTING=$(cat "$LOCK_FILE" 2>/dev/null || echo "?")
    if [ "$EXISTING" != "?" ] && ! kill -0 "$EXISTING" 2>/dev/null; then
        rm -f "$LOCK_FILE"; echo $$ > "$LOCK_FILE"
    else
        echo "Another monthly run is active (PID=$EXISTING) — skipping."
        exit 0
    fi
fi
trap "rm -f '$LOCK_FILE'" EXIT

export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10

cd "$REPO_DIR"

# ── Step 1: Smoke test — abort if model broken ───────────────────────────
echo "--- Step 1: Pre-flight smoke test ---"
if ! "$PYTHON" scripts/smoke_test_model.py --strategy renquant_104; then
    echo "Smoke test FAILED — aborting monthly calibrator refresh."
    notify "RenQuant 104 MONTHLY-ABORT" "Pre-flight smoke test failed; calibrator NOT refreshed. Check $LOG"
    exit 1
fi

# ── Step 2: Re-fit calibrator on current production model ────────────────
echo "--- Step 2: Re-fit global calibrator ---"
if ! "$PYTHON" scripts/fit_panel_calibrator.py --strategy renquant_104; then
    echo "Calibrator fit FAILED — prior calibrator preserved."
    notify "RenQuant 104 MONTHLY-FAIL" "Calibrator fit failed; prior calibrator unchanged. Check $LOG"
    exit 1
fi

# ── Step 3: Validate calibrator non-collapse + diversity ─────────────────
# Acceptance gate G2 invariant: n_unique_prob_y >= 10. Pre-fix, a
# calibrator collapsed to a single bucket (XGB plateau at best_iter=4)
# silently produced P(out) ≈ const → all candidates rank identical
# → ranking degenerate. This guard catches that regression.
echo "--- Step 3: Validate calibrator ---"
if ! "$PYTHON" scripts/smoke_test_model.py --strategy renquant_104; then
    echo "Post-fit smoke test FAILED — calibrator collapsed?"
    notify "RenQuant 104 MONTHLY-FAIL" "Post-fit smoke test failed; calibrator may be collapsed"
    exit 1
fi

CAL_INFO=$("$PYTHON" -c "
import json
from pathlib import Path
sd = Path('$REPO_DIR/backtesting/renquant_104')
cfg = json.loads((sd / 'strategy_config.json').read_text())
cal_rel = cfg['ranking']['panel_scoring']['global_calibration']['artifact_path']
m = json.loads((sd / cal_rel).read_text())
n_knots_p = len(m.get('probability', {}).get('x', []))
n_knots_e = len(m.get('expected_return', {}).get('x', []))
md = m.get('metadata', {})
n_uniq = md.get('n_unique_prob_y', '—')
pool_ic = md.get('pool_ic', '—')
print(f'knots: prob={n_knots_p} er={n_knots_e}  n_unique_prob_y={n_uniq}  pool_ic={pool_ic}')
" 2>/dev/null || echo "calibrator info unavailable")
echo "Calibrator state: $CAL_INFO"

# ── Step 4: Refresh dashboard so monthly cadence is visible ──────────────
"$PYTHON" "$REPO_DIR/scripts/build_dashboard.py" --broker alpaca \
    --out "$REPO_DIR/doc/dashboard.md" 2>&1 | tail -5 \
    || echo "dashboard refresh failed (non-fatal)"

echo "=== monthly_calibrator_refresh PASSED at $(date) — $CAL_INFO ==="
notify "RenQuant 104 MONTHLY-CAL ✓" "Calibrator refreshed: $CAL_INFO"
