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
VENV_DIR="$REPO_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
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

THREADS=$("$PYTHON" - <<'PY'
import os
print(os.cpu_count() or 1)
PY
)
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"

cd "$REPO_DIR"

# ── Step 1: Smoke test — abort if model broken ───────────────────────────
echo "--- Step 1: Pre-flight smoke test ---"
if ! "$PYTHON" scripts/smoke_test_model.py --strategy renquant_104; then
    echo "Smoke test FAILED — aborting monthly calibrator refresh."
    notify "RenQuant 104 MONTHLY-ABORT" "Pre-flight smoke test failed; calibrator NOT refreshed. Check $LOG"
    exit 1
fi

# ── Step 2: Re-fit calibrator on current production model ────────────────
# 2026-05-11 sim/prod isolation: explicit --out so the calibrator lands
# under artifacts/prod/ (without --out the script derives a flat-path
# orphan from the panel artifact's stem that prod runner won't read).
#
# 2026-05-17 ACCEPTANCE GATE — backup BEFORE refit + IC regression check
# AFTER. Same bug class as today's Sunday-sweep corruption (NGB
# val_IC=-0.0165 → prod silently). Pre-fix this script had no rollback
# target if the new calibrator regressed.
echo "--- Step 2: Re-fit global calibrator ---"
PROD_CAL="$REPO_DIR/backtesting/renquant_104/artifacts/prod/panel-rank-calibration.json"
ROLLBACK_CAL="$REPO_DIR/backtesting/renquant_104/artifacts/prod/panel-rank-calibration.monthly_rollback_$DATE.json"

# Snapshot prior calibrator for rollback BEFORE any destructive write.
BASELINE_POOL_IC="None"
BASELINE_N_UNIQUE=0
if [ -f "$PROD_CAL" ]; then
    # ATOMIC: write to .tmp then mv (POSIX cp is two syscalls;
    # SIGKILL mid-cp → half-written rollback). Audit P0-16.
    cp "$PROD_CAL" "$ROLLBACK_CAL.tmp" && mv "$ROLLBACK_CAL.tmp" "$ROLLBACK_CAL"
    echo "Pre-refit backup: $ROLLBACK_CAL"
    BASELINE_POOL_IC=$("$PYTHON" -c "
import json
m = json.load(open('$PROD_CAL'))
print(m.get('metadata', {}).get('pool_ic', 'None'))
" 2>/dev/null || echo "None")
    BASELINE_N_UNIQUE=$("$PYTHON" -c "
import json
m = json.load(open('$PROD_CAL'))
print(m.get('metadata', {}).get('n_unique_prob_y', 0))
" 2>/dev/null || echo "0")
    echo "Baseline: pool_ic=$BASELINE_POOL_IC  n_unique_prob_y=$BASELINE_N_UNIQUE"
else
    echo "No prior calibrator at $PROD_CAL — first-ever fit (no regression baseline)"
fi

if ! "$PYTHON" scripts/fit_panel_calibrator.py --strategy renquant_104 \
        --out "$PROD_CAL"; then
    echo "Calibrator fit FAILED — prior calibrator preserved."
    notify "RenQuant 104 MONTHLY-FAIL" "Calibrator fit failed; prior calibrator unchanged. Check $LOG"
    exit 1
fi

# ── Step 3: Validate calibrator — non-collapse + IC-regression-vs-baseline ─
# 2 hard checks:
#   H1 (existing): smoke test passes
#   H2 (new): pool_ic did not drop > 0.02 vs baseline (regression guard)
#            n_unique_prob_y >= 10 (non-collapse, was display-only pre-fix)
# Either fail → rollback to ROLLBACK_CAL + ntfy + exit non-zero.
# References:
#   - Diebold-Mariano 1995 (J. Bus. Econ. Stat.) "Comparing Predictive
#     Accuracy" — framework for forecast-accuracy testing. 0.02 IC drop
#     threshold ≈ 2σ given typical pool_ic std ~0.01; heuristic, not
#     formal DM-test (CLAUDE.md §5.12 — exploratory tune-via-A/B).
#   - n_unique_prob_y ≥ 10: internal "G2 calibrator non-collapse"
#     invariant (kernel/model_acceptance.py:DEFAULT_GATES) — calibrator
#     with fewer than 10 unique buckets degenerates to constant scores
#     → ranking collapse; was display-only pre-fix.
echo "--- Step 3: Validate calibrator ---"
if ! "$PYTHON" scripts/smoke_test_model.py --strategy renquant_104; then
    echo "Post-fit smoke test FAILED — rolling back to baseline calibrator."
    # ATOMIC rollback (audit P0-16)
    if [ -f "$ROLLBACK_CAL" ]; then
        cp "$ROLLBACK_CAL" "$PROD_CAL.tmp" && mv "$PROD_CAL.tmp" "$PROD_CAL"
    fi
    notify "RenQuant 104 MONTHLY-FAIL" "Post-fit smoke test failed; rolled back."
    exit 1
fi

# IC-regression-vs-baseline check + non-collapse gate
GATE_VERDICT=$("$PYTHON" - "$PROD_CAL" "$BASELINE_POOL_IC" "$BASELINE_N_UNIQUE" <<'PY'
import json, sys, math
prod_cal = sys.argv[1]
base_ic_str = sys.argv[2]
base_n_uniq_str = sys.argv[3]
m = json.load(open(prod_cal))
md = m.get("metadata", {}) or {}
new_ic = md.get("pool_ic")
new_n_uniq = md.get("n_unique_prob_y", 0)

fails = []
# H2a non-collapse hard guard
try:
    n_uniq = int(new_n_uniq)
    if n_uniq < 10:
        fails.append(f"n_unique_prob_y={n_uniq} < 10 (collapsed)")
except (TypeError, ValueError):
    fails.append(f"n_unique_prob_y={new_n_uniq!r} not int")

# H2b IC regression vs baseline (only if baseline existed)
if base_ic_str != "None":
    try:
        base_ic = float(base_ic_str)
        if new_ic is None or not math.isfinite(float(new_ic)):
            fails.append(f"new pool_ic={new_ic!r} not finite")
        else:
            new_ic = float(new_ic)
            drop = base_ic - new_ic
            if drop > 0.02:
                fails.append(f"pool_ic dropped {base_ic:+.4f} → {new_ic:+.4f} (Δ {-drop:+.4f} > 2pp)")
    except (TypeError, ValueError) as e:
        fails.append(f"baseline pool_ic parse: {e}")

if fails:
    print("FAIL: " + "; ".join(fails))
    sys.exit(1)
print(f"OK pool_ic={new_ic} n_unique={new_n_uniq}")
PY
)
GATE_RC=$?
if [ $GATE_RC -ne 0 ]; then
    echo "ACCEPTANCE GATE FAILED: $GATE_VERDICT"
    echo "Rolling back to baseline calibrator."
    if [ -f "$ROLLBACK_CAL" ]; then
        # ATOMIC rollback (audit P0-16)
        cp "$ROLLBACK_CAL" "$PROD_CAL.tmp" && mv "$PROD_CAL.tmp" "$PROD_CAL"
        # Smoke test the rollback too
        if "$PYTHON" scripts/smoke_test_model.py --strategy renquant_104 >/dev/null 2>&1; then
            notify "RenQuant 104 MONTHLY-REJECT" "Calibrator REJECTED ($GATE_VERDICT); rolled back to prior."
        else
            notify "RenQuant 104 MONTHLY-CRITICAL" "Calibrator rejected AND rollback failed smoke. Operator action REQUIRED."
        fi
    fi
    exit 1
fi
echo "Gate: $GATE_VERDICT"

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
