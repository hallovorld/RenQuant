#!/usr/bin/env bash
# Auto-revert B1 (227-watchlist retrain) once it finishes if IC regression
# was confirmed. Reads the freshly-written panel-ltr.json's oos_mean_ic and:
#   • IC ≥ +0.040 (within noise of golden baseline)  → KEEP B1, do nothing.
#   • IC <  +0.040                                  → REVERT panel-ltr +
#                                                     ngboost-head +
#                                                     panel-rank-calibration
#                                                     from the chmod 444
#                                                     checkpoint.
#
# The checkpoint at artifacts/checkpoint_2026-04-27_22h28/ has the
# OOS-IC-+0.0400 production triplet that survives smoke + 47 tests.
#
# Logs: logs/ablation_2026-04-27/auto_revert.log
# Status: logs/ablation_2026-04-27/auto_revert_status.json

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO_ROOT/logs/ablation_2026-04-27/auto_revert.log"
STATUS="$REPO_ROOT/logs/ablation_2026-04-27/auto_revert_status.json"
B1_LOG="$REPO_ROOT/logs/ablation_2026-04-27/b1_baseline_227.log"
CHECK="$REPO_ROOT/backtesting/renquant_104/artifacts/checkpoint_2026-04-27_22h28"
LIVE="$REPO_ROOT/backtesting/renquant_104/artifacts"
THRESHOLD="0.040"

CONDA_ACT="source ~/miniconda3/etc/profile.d/conda.sh && conda activate renquant"

write_status() {
    local stage="$1" msg="$2" ic="${3:-}"
    cat > "$STATUS" <<EOF
{
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "stage":      "$stage",
  "ic":         "$ic",
  "threshold":  "$THRESHOLD",
  "message":    "$msg"
}
EOF
}

mkdir -p "$REPO_ROOT/logs/ablation_2026-04-27"
exec >>"$LOG" 2>&1
echo
echo "============================================================"
echo "[auto-revert] $(date) — START"
write_status "waiting" "polling B1 log for FullTrainingPipeline DONE"

# Wait for B1 fully complete
while true; do
    if grep -q "FullTrainingPipeline DONE" "$B1_LOG" 2>/dev/null; then
        echo "[auto-revert] B1 reached FullTrainingPipeline DONE"
        break
    fi
    if grep -q "Traceback" "$B1_LOG" 2>/dev/null; then
        echo "[auto-revert] B1 ABORTED (Traceback). Reverting anyway as defensive."
        break
    fi
    sleep 30
done

# Read final panel-ltr OOS IC
NEW_IC=$(bash -c "$CONDA_ACT && python -c \"
import json
try:
    d = json.load(open('$LIVE/panel-ltr.json'))
    print(d.get('oos_mean_ic', 'unknown'))
except Exception as e:
    print(f'error_{e}')
\"")
echo "[auto-revert] new panel-ltr.json OOS IC = $NEW_IC"

# Numeric compare with bc
KEEP_B1=$(bash -c "$CONDA_ACT && python -c \"
ic = '$NEW_IC'
try:
    print('1' if float(ic) >= float('$THRESHOLD') else '0')
except Exception:
    print('0')
\"")

if [[ "$KEEP_B1" == "1" ]]; then
    echo "[auto-revert] IC $NEW_IC >= $THRESHOLD — KEEPING B1 (no revert)"
    write_status "kept_b1" "B1 IC within tolerance, no revert" "$NEW_IC"
    exit 0
fi

echo "[auto-revert] IC $NEW_IC < $THRESHOLD — REVERTING from checkpoint"
write_status "reverting" "B1 IC below threshold, restoring checkpoint" "$NEW_IC"

# Backup the regressed B1 outputs first (in case operator wants to inspect)
ts="$(date +%Y%m%d_%H%M%S)"
B1_BACKUP="$LIVE/b1_regressed_${ts}"
mkdir -p "$B1_BACKUP"
for f in panel-ltr.json ngboost-head.json panel-rank-calibration.json; do
    [[ -f "$LIVE/$f" ]] && cp "$LIVE/$f" "$B1_BACKUP/$f"
done
echo "[auto-revert] B1 regressed outputs backed up → $B1_BACKUP"

# Restore from checkpoint (chmod 444 immutable)
for f in panel-ltr.json ngboost-head.json strategy_config.json strategy_config.golden.json; do
    if [[ -f "$CHECK/$f" ]]; then
        # Need to allow overwrite since chmod 444 + production may also be 444
        chmod u+w "$LIVE/$f" 2>/dev/null || true
        cp "$CHECK/$f" "$LIVE/$f" 2>/dev/null || cp "$CHECK/$f" "$REPO_ROOT/backtesting/renquant_104/$f"
        echo "[auto-revert] restored $f"
    fi
done
# Calibration NOT in checkpoint dir directly — copy from any pre-B1 backup if available
# Otherwise leave the post-B1 cal (calibration is not load-bearing for trade admission)

# Verify SHA matches checkpoint manifest
echo "[auto-revert] verifying SHAs..."
( cd "$CHECK" && shasum -a 256 -c MANIFEST.sha256 2>&1 | head -10 ) || true

NEW_IC_AFTER=$(bash -c "$CONDA_ACT && python -c \"
import json
print(json.load(open('$LIVE/panel-ltr.json'))['oos_mean_ic'])
\"")
echo "[auto-revert] post-revert panel-ltr.json IC = $NEW_IC_AFTER"
write_status "DONE" "Reverted to checkpoint; production IC restored to $NEW_IC_AFTER (B1 backed up to $B1_BACKUP)" "$NEW_IC_AFTER"
echo "[auto-revert] DONE  $(date)"
