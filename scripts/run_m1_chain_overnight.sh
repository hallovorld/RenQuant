#!/usr/bin/env bash
# Autonomous chain: wait for B1 → run M1a (panel @ 20d) → M1b (panel @ 60d)
# → fit conformal Gate B → write status doc.
#
# Designed to run unattended overnight. All artifacts go to side paths so
# nothing touches production unless the operator explicitly promotes.
#
# Usage (from repo root):
#     bash scripts/run_m1_chain_overnight.sh
#
# Status:    logs/ablation_2026-04-27/m1_chain_status.json (live progress)
# Logs:      logs/ablation_2026-04-27/m1a_20d.log + m1b_60d.log + conformal_fit.log

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs/ablation_2026-04-27"
STATUS="$LOG_DIR/m1_chain_status.json"
B1_LOG="$LOG_DIR/b1_baseline_227.log"
mkdir -p "$LOG_DIR"

CONDA_ACT="source ~/miniconda3/etc/profile.d/conda.sh && conda activate renquant"

write_status() {
    local stage="$1" msg="$2"
    cat > "$STATUS" <<EOF
{
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "stage":      "$stage",
  "message":    "$msg"
}
EOF
}

write_status "init" "Waiting for B1 FullTrainingPipeline DONE marker"

# ── Phase 1: wait for B1 panel + NGBoost + calibration to finish ────────────
echo "[chain] $(date) — waiting for B1 to finish (FullTrainingPipeline DONE marker)..."
while true; do
    if grep -q "FullTrainingPipeline DONE" "$B1_LOG" 2>/dev/null; then
        echo "[chain] $(date) — B1 done; new panel-ltr + ngboost + cal artifacts written"
        write_status "b1_done" "Starting M1a panel @ 20d"
        break
    fi
    if grep -q "Traceback" "$B1_LOG" 2>/dev/null; then
        echo "[chain] $(date) — B1 ABORTED (Traceback in log). Stopping chain."
        write_status "b1_failed" "B1 hit Traceback — chain aborted, see log"
        exit 2
    fi
    sleep 60
done

cd "$REPO_ROOT"

# ── Phase 2: M1a — panel @ 20d ──────────────────────────────────────────────
write_status "m1a_running" "Training panel @ 20d horizon"
echo "[chain] $(date) — starting M1a (panel @ 20d, --skip-baseline since per-ticker models already fresh)"
bash -c "$CONDA_ACT && python scripts/train_104.py \
    --skip-baseline --skip-recalibrate --skip-acceptance --force \
    --strategy-config-name strategy_config.20d.json" \
    > "$LOG_DIR/m1a_20d.log" 2>&1
M1A_RC=$?
if [[ $M1A_RC -ne 0 ]]; then
    echo "[chain] $(date) — M1a FAILED (rc=$M1A_RC)"
    write_status "m1a_failed" "M1a panel @ 20d returned $M1A_RC — stopping chain"
    exit 3
fi
echo "[chain] $(date) — M1a done"

# ── Phase 3: M1b — panel @ 60d ──────────────────────────────────────────────
write_status "m1b_running" "Training panel @ 60d horizon"
echo "[chain] $(date) — starting M1b (panel @ 60d)"
bash -c "$CONDA_ACT && python scripts/train_104.py \
    --skip-baseline --skip-recalibrate --skip-acceptance --force \
    --strategy-config-name strategy_config.60d.json" \
    > "$LOG_DIR/m1b_60d.log" 2>&1
M1B_RC=$?
if [[ $M1B_RC -ne 0 ]]; then
    echo "[chain] $(date) — M1b FAILED (rc=$M1B_RC)"
    write_status "m1b_failed" "M1b panel @ 60d returned $M1B_RC — stopping chain"
    exit 4
fi
echo "[chain] $(date) — M1b done"

# ── Phase 4: M3 conformal fit ───────────────────────────────────────────────
write_status "conformal_fit_running" "Fitting per-regime Gate B thresholds"
echo "[chain] $(date) — fitting conformal Gate B"
bash -c "$CONDA_ACT && python scripts/fit_conformal_gate_b.py" \
    > "$LOG_DIR/conformal_fit.log" 2>&1
CONF_RC=$?
if [[ $CONF_RC -ne 0 ]]; then
    echo "[chain] $(date) — conformal fit FAILED (rc=$CONF_RC) — non-fatal, continuing"
    write_status "conformal_failed_non_fatal" "Gate B fit returned $CONF_RC; continuing"
else
    echo "[chain] $(date) — conformal fit done"
fi

# ── Phase 5: backup all new artifacts (broker-tagged) ───────────────────────
write_status "backup_running" "Snapshotting new artifacts"
bash "$REPO_ROOT/scripts/backup_state.sh" alpaca >> "$LOG_DIR/m1_chain.log" 2>&1 || true

# ── Phase 6: extract IC results to status ───────────────────────────────────
PANEL_IC=$(bash -c "$CONDA_ACT && python -c \"
import json
for h, p in [('10d', 'panel-ltr.json'), ('20d', 'panel-ltr.20d.json'), ('60d', 'panel-ltr.60d.json')]:
    try:
        d = json.load(open(f'backtesting/renquant_104/artifacts/{p}'))
        ic = d.get('oos_mean_ic')
        rows = d.get('panel_shape', {}).get('rows', '?')
        print(f'{h}: IC={ic:+.5f} rows={rows}' if ic else f'{h}: missing')
    except Exception as e:
        print(f'{h}: error {e}')
\"")

write_status "DONE" "M-series chain complete: $(echo "$PANEL_IC" | tr '\n' '|')"
echo "[chain] $(date) — ALL DONE"
echo "$PANEL_IC"
