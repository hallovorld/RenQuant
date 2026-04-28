#!/usr/bin/env bash
# Auto-revert B1 (227-watchlist retrain) once it finishes if IC regression
# was confirmed.
#
# 2026-04-28 audit (principle 5.5 — rollback rehearsal): the previous
# version of this script silently mis-copied strategy_config.json into
# artifacts/ instead of the production location, leaving the watchlist=227
# config in production while the model was reverted to 103. Result was
# the 06:32 ntfy fingerprint mismatch. This rewrite:
#   • per-file explicit destination paths (artifacts/ for models,
#     strategy dir for configs) — no LIVE/$f shorthand
#   • post-cp SHA verification on every restored file against MANIFEST
#   • cp errors no longer suppressed; any failure => FAILED status + exit 1
#
# Behaviour:
#   • IC ≥ +0.040 (within noise of golden baseline)  → KEEP B1, do nothing.
#   • IC <  +0.040                                  → REVERT panel-ltr +
#                                                     ngboost-head +
#                                                     strategy_config(s)
#                                                     from the immutable
#                                                     checkpoint.
#
# Logs: logs/ablation_2026-04-27/auto_revert.log
# Status: logs/ablation_2026-04-27/auto_revert_status.json

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO_ROOT/logs/ablation_2026-04-27/auto_revert.log"
STATUS="$REPO_ROOT/logs/ablation_2026-04-27/auto_revert_status.json"
B1_LOG="$REPO_ROOT/logs/ablation_2026-04-27/b1_baseline_227.log"
CHECK="$REPO_ROOT/backtesting/renquant_104/artifacts/checkpoint_2026-04-27_22h28"
STRAT_DIR="$REPO_ROOT/backtesting/renquant_104"
ART_DIR="$STRAT_DIR/artifacts"
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

# Per-file restore: explicit destination path, no $LIVE shorthand.
# Returns 0 on success (cp ok + post-cp SHA matches checkpoint), 1 otherwise.
# Always logs what it did. Loud failures only — no `2>/dev/null` swallowing.
restore_file() {
    local fname="$1" dest_dir="$2"
    local src="$CHECK/$fname"
    local dst="$dest_dir/$fname"

    if [[ ! -f "$src" ]]; then
        echo "[auto-revert] FAIL: source missing $src"
        return 1
    fi
    # Allow overwrite even if file is 444 (immutable) at destination.
    if [[ -e "$dst" ]]; then
        chmod u+w "$dst" || { echo "[auto-revert] FAIL: chmod $dst"; return 1; }
    fi
    if ! cp "$src" "$dst"; then
        echo "[auto-revert] FAIL: cp $src -> $dst"
        return 1
    fi
    # Verify the post-cp file's SHA matches MANIFEST.
    local expected actual
    expected="$(grep "  $fname$" "$CHECK/MANIFEST.sha256" 2>/dev/null | awk '{print $1}')"
    if [[ -z "$expected" ]]; then
        echo "[auto-revert] WARN: $fname not in MANIFEST.sha256 — copied but not SHA-verified"
        return 0
    fi
    actual="$(shasum -a 256 "$dst" | awk '{print $1}')"
    if [[ "$expected" != "$actual" ]]; then
        echo "[auto-revert] FAIL: SHA mismatch for $dst"
        echo "  expected: $expected"
        echo "  actual:   $actual"
        return 1
    fi
    echo "[auto-revert] OK: restored $fname -> $dst (SHA verified)"
    return 0
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
    d = json.load(open('$ART_DIR/panel-ltr.json'))
    print(d.get('oos_mean_ic', 'unknown'))
except Exception as e:
    print(f'error_{e}')
\"")
echo "[auto-revert] new panel-ltr.json OOS IC = $NEW_IC"

# Numeric compare
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

# Backup the regressed B1 outputs first
ts="$(date +%Y%m%d_%H%M%S)"
B1_BACKUP="$ART_DIR/b1_regressed_${ts}"
mkdir -p "$B1_BACKUP"
for f in panel-ltr.json ngboost-head.json panel-rank-calibration.json; do
    [[ -f "$ART_DIR/$f" ]] && cp "$ART_DIR/$f" "$B1_BACKUP/$f"
done
echo "[auto-revert] B1 regressed outputs backed up → $B1_BACKUP"

# Restore from checkpoint with explicit per-file destinations.
# Models go to artifacts/, configs go to strategy dir (one level up).
fail=0
restore_file panel-ltr.json    "$ART_DIR"   || fail=1
restore_file ngboost-head.json "$ART_DIR"   || fail=1
restore_file strategy_config.json        "$STRAT_DIR" || fail=1
restore_file strategy_config.golden.json "$STRAT_DIR" || fail=1

if [[ "$fail" -ne 0 ]]; then
    msg="One or more files failed to restore — production state may be INCONSISTENT. Check log."
    echo "[auto-revert] FAILED: $msg"
    write_status "FAILED" "$msg" "$NEW_IC"
    exit 1
fi

# Calibration NOT in checkpoint — leave the post-B1 cal as-is.
# (Calibration is not load-bearing for trade admission; gates use Gate-B.)

NEW_IC_AFTER=$(bash -c "$CONDA_ACT && python -c \"
import json
print(json.load(open('$ART_DIR/panel-ltr.json'))['oos_mean_ic'])
\"")
echo "[auto-revert] post-revert panel-ltr.json IC = $NEW_IC_AFTER"

# Sanity: live config watchlist must match model's training watchlist.
WL_CHECK=$(bash -c "$CONDA_ACT && python -c \"
import json, hashlib
import sys
sys.path.insert(0, '$STRAT_DIR')
from kernel.config_consistency import fingerprint_config

cfg = json.load(open('$STRAT_DIR/strategy_config.json'))
mdl = json.load(open('$ART_DIR/panel-ltr.json'))
live_fp   = fingerprint_config(cfg)
stored_fp = mdl.get('config_fingerprint')
print('match' if (stored_fp is None or stored_fp == live_fp) else f'MISMATCH live={live_fp} stored={stored_fp}')
\"")
echo "[auto-revert] config-consistency post-revert: $WL_CHECK"
if [[ "$WL_CHECK" != match* ]]; then
    msg="Post-revert config/model fingerprint MISMATCH: $WL_CHECK"
    echo "[auto-revert] FAILED: $msg"
    write_status "FAILED" "$msg" "$NEW_IC_AFTER"
    exit 1
fi

write_status "DONE" "Reverted to checkpoint; production IC $NEW_IC_AFTER, fingerprint matches (B1 backed up to $B1_BACKUP)" "$NEW_IC_AFTER"
echo "[auto-revert] DONE  $(date)"
