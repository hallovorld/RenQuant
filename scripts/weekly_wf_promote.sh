#!/usr/bin/env bash
# weekly_wf_promote.sh — Weekly retrain + walk-forward gate + promote.
#
# 2026-05-09 audit FIX-C: this REPLACES daily auto-promote (which had
# RQ_ALLOW_NO_WF=1 bypass and let single-cut acceptance gates ship
# bad models). Trust boundary: every production promote now passes
# WF 3-cut + §5.2 sanity battery (shuffled-label + time-shift placebo).
#
# Schedule: Saturday 04:00 PT (NYC closed weekend buffer).
# Plist: scripts/launchd/com.renquant.weekly-wf-promote.plist
#
# Steps:
#   1. Smoke test (catch immediate breakage before 90 min train)
#   2. Retrain → produces panel-ltr.staging.alpha158_fund.json
#   3. Run scripts/run_wf_gate.py — 3-cut WF + §5.2 sanity. Historical
#      WF uses a manifest, so the gate first verifies the manifest artifacts
#      match the candidate recipe before stamping wf_gate_metadata.
#   4. _check_wf_gate inside promote() refuses to swap if metadata
#      missing or .passed=False — NO RQ_ALLOW_NO_WF override here
#   5. ntfy alert with verdict + Sharpe / IC numbers
#   6. Refresh dashboard so users see the new model state
#
# Failure modes:
#   - Smoke test fail → exit 1, no train
#   - Training fail → exit 1, prior artifact preserved
#   - WF gate fail → exit 1, prior artifact preserved (gate refuses promote)
#   - Promote fail (e.g. acceptance G1-G11 fail) → prior artifact preserved
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
CONDA_PREFIX="/Users/renhao/miniconda3/envs/renquant"
PYTHON="$CONDA_PREFIX/bin/python"
LOG_DIR="$REPO_DIR/logs/weekly_wf_promote"
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
echo "=== weekly_wf_promote started at $(date) ==="

# Lock — prevent concurrent runs (a 90-min job can stack if the user
# triggers a manual rerun before the previous finishes).
LOCK_FILE="/tmp/renquant_104_weekly_wf.lock"
if ! ( set -C; echo $$ > "$LOCK_FILE" ) 2>/dev/null; then
    EXISTING=$(cat "$LOCK_FILE" 2>/dev/null || echo "?")
    if [ "$EXISTING" != "?" ] && ! kill -0 "$EXISTING" 2>/dev/null; then
        echo "Stale lock (PID $EXISTING dead) — clearing."
        rm -f "$LOCK_FILE"
        echo $$ > "$LOCK_FILE"
    else
        echo "Another weekly run is active (PID=$EXISTING) — skipping."
        notify "RenQuant 104 SKIP" "Weekly WF promote skipped — already running"
        exit 0
    fi
fi
trap "rm -f '$LOCK_FILE'" EXIT

# Saturate the M2 Pro per CLAUDE.md §5.10
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10

cd "$REPO_DIR"

# ── Step 1: Smoke test ────────────────────────────────────────────────────
echo "--- Step 1: Pre-flight smoke test ---"
if ! "$PYTHON" scripts/smoke_test_model.py --strategy renquant_104; then
    echo "Smoke test FAILED — aborting weekly promote (no train)."
    notify "RenQuant 104 WEEKLY-ABORT" "Pre-flight smoke test failed; weekly promote skipped. Check $LOG"
    exit 1
fi

# ── Step 2: BACKUP current production before destructive retrain ──────────
# Architectural caveat (audit 2026-05-09): daily_retrain_alpha158_fund.sh
# writes DIRECTLY to the active artifact path — there's no staging step.
# If WF gate then rejects, the prior trustworthy model would be gone.
# Per CLAUDE.md §5.5 (rollback rehearsal mandate), backup BEFORE retrain
# and restore on WF failure.
# 2026-05-11 sim/prod isolation: prod artifacts moved to artifacts/prod/.
# Before this fix, ACTIVE_ART pointed at the now-empty flat path, so the
# `[ -f "$ACTIVE_ART" ]` backup guard always failed silently and rollback
# never copied anything → §5.5 rehearsal invariant decoration-only.
ART_DIR="$REPO_DIR/backtesting/renquant_104/artifacts/prod"
ACTIVE_ART="$ART_DIR/panel-ltr.alpha158_fund.json"
ACTIVE_CAL="$ART_DIR/panel-rank-calibration.json"
ROLLBACK_ART="$ART_DIR/panel-ltr.alpha158_fund.weekly_rollback_$DATE.json"
ROLLBACK_CAL="$ART_DIR/panel-rank-calibration.weekly_rollback_$DATE.json"

echo "--- Step 2: Backup prior production artifacts (rollback rehearsal) ---"
if [ -f "$ACTIVE_ART" ]; then
    cp "$ACTIVE_ART" "$ROLLBACK_ART"
    echo "Backup model: $ROLLBACK_ART"
fi
if [ -f "$ACTIVE_CAL" ]; then
    cp "$ACTIVE_CAL" "$ROLLBACK_CAL"
    echo "Backup calibrator: $ROLLBACK_CAL"
fi

# ── Step 3: Retrain on the alpha158+fund 169-feat pipeline ─────────────
# Note: this also REFITS the calibrator (per daily_retrain_alpha158_fund.py
# step 4). If WF gate then fails, the calibrator on disk is the NEW one
# (matched to the rejected model), so we restore BOTH on rollback.
echo "--- Step 3: Retrain panel-LTR + calibrator (alpha158+fund+PEAD+SUE) ---"
if ! bash scripts/daily_retrain_alpha158_fund.sh; then
    echo "Training FAILED — prior production artifact still in place (no overwrite happened)."
    notify "RenQuant 104 WEEKLY-FAIL" "Training failed; production model unchanged. Check $LOG"
    exit 1
fi
echo "Training pipeline finished at $(date)"

# ── Step 4: Run WF gate (3-cut WF + §5.2 sanity battery) ──────────────────
echo "--- Step 4: Walk-forward gate (3-cut + sanity) ---"
if ! "$PYTHON" scripts/run_wf_gate.py \
    --artifact "$ACTIVE_ART" \
    --strategy-config strategy_config.sim_wl200.json \
    --strict; then
    echo "WF gate FAILED — ROLLING BACK to prior production model."
    if [ -f "$ROLLBACK_ART" ]; then
        cp "$ROLLBACK_ART" "$ACTIVE_ART"
        echo "Restored model from $ROLLBACK_ART"
    fi
    if [ -f "$ROLLBACK_CAL" ]; then
        cp "$ROLLBACK_CAL" "$ACTIVE_CAL"
        echo "Restored calibrator from $ROLLBACK_CAL"
    fi
    # Smoke-test the rolled-back state to confirm we're back to a working model
    if ! "$PYTHON" scripts/smoke_test_model.py --strategy renquant_104; then
        notify "RenQuant 104 WEEKLY-CRITICAL" \
            "Rollback FAILED smoke test — operator action REQUIRED. Production may be in unknown state."
    else
        notify "RenQuant 104 WEEKLY-FAIL" \
            "Walk-forward gate REJECTED the new model. Rolled back to prior + calibrator. Check $LOG."
    fi
    exit 1
fi

# ── Step 5: Inspect gate metadata + summarize ─────────────────────────────
GATE_SUMMARY=$("$PYTHON" -c "
import json
m = json.load(open('$ACTIVE_ART'))
gate = m.get('wf_gate_metadata') or m.get('metadata', {}).get('wf_gate_metadata') or {}
sharpe = gate.get('wf_3cut_sharpe_mean')
apy    = gate.get('wf_3cut_apy_mean')
shuf   = gate.get('sanity_shuffled_ic')
plac   = gate.get('sanity_placebo_ic')
parts = []
if sharpe is not None: parts.append(f'WF Sharpe {sharpe:+.2f}')
if apy is not None:    parts.append(f'APY {apy:+.2f}%')
if shuf is not None:   parts.append(f'shuf_IC {shuf:+.4f}')
if plac is not None:   parts.append(f'placebo_IC {plac:+.4f}')
print('  '.join(parts) if parts else '(no metadata)')
" 2>/dev/null || echo "(metadata parse failed)")
echo "Gate metadata: $GATE_SUMMARY"

# ── Step 6: Refresh dashboard ─────────────────────────────────────────────
"$PYTHON" "$REPO_DIR/scripts/build_dashboard.py" --broker alpaca \
    --out "$REPO_DIR/doc/dashboard.md" 2>&1 | tail -5 \
    || echo "dashboard refresh failed (non-fatal)"

echo "=== weekly_wf_promote PASSED at $(date) — $GATE_SUMMARY ==="
notify "RenQuant 104 WEEKLY-PROMOTE ✓" \
    "Walk-forward gate passed. New model promoted to production. $GATE_SUMMARY"
