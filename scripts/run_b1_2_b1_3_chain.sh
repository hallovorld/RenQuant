#!/usr/bin/env bash
# Autonomous chain: wait for F3 → run B1.2 (filtered 10d) → B1.3 (tuned 60d)
# → write final summary doc with all M1 + F3 + B1.2 + B1.3 results.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs/ablation_2026-04-27"
STATUS="$LOG_DIR/b1_chain_status.json"
F3_LOG="$LOG_DIR/b1_1_retune.log"
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

echo "[chain-b1.2-b1.3] $(date) — start, waiting for F3"
write_status "waiting_f3" "polling F3 retune log for FullTrainingPipeline DONE"

# Phase 1 — wait for F3
while true; do
    if grep -q "FullTrainingPipeline DONE" "$F3_LOG" 2>/dev/null; then
        echo "[chain] F3 done"
        break
    fi
    if grep -q "Traceback" "$F3_LOG" 2>/dev/null; then
        echo "[chain] F3 ABORTED — continuing chain anyway"
        break
    fi
    sleep 60
done

cd "$REPO_ROOT"

# Phase 2 — B1.2: filtered 10d watchlist
write_status "b1_2_running" "Training filtered ~75-ticker 10d panel"
echo "[chain] $(date) — starting B1.2 (filtered 75-ticker 10d)"
bash -c "$CONDA_ACT && python scripts/train_104.py \
    --skip-baseline --skip-recalibrate --skip-acceptance --force \
    --strategy-config-name strategy_config.b1_2_filtered_10d.json" \
    > "$LOG_DIR/b1_2_filtered_10d.log" 2>&1
B1_2_RC=$?
if [[ $B1_2_RC -ne 0 ]]; then
    echo "[chain] B1.2 FAILED (rc=$B1_2_RC) — continuing"
    write_status "b1_2_failed" "B1.2 returned $B1_2_RC; continuing to B1.3"
fi
echo "[chain] $(date) — B1.2 finished"

# Phase 3 — B1.3: 60d with aggressive hypers
write_status "b1_3_running" "Training 60d panel with aggressive hypers"
echo "[chain] $(date) — starting B1.3 (60d num_boost=800 max_depth=5)"
bash -c "$CONDA_ACT && python scripts/train_104.py \
    --skip-baseline --skip-recalibrate --skip-acceptance --force \
    --strategy-config-name strategy_config.b1_3_60d_tuned.json" \
    > "$LOG_DIR/b1_3_60d_tuned.log" 2>&1
B1_3_RC=$?
if [[ $B1_3_RC -ne 0 ]]; then
    echo "[chain] B1.3 FAILED (rc=$B1_3_RC)"
    write_status "b1_3_failed" "B1.3 returned $B1_3_RC; finalising"
fi
echo "[chain] $(date) — B1.3 finished"

# Phase 4 — final summary
write_status "summarising" "Building final IC summary"
echo "[chain] $(date) — generating final IC summary"

bash -c "$CONDA_ACT && python -c \"
import json
print('═'*72)
print('  FINAL IC SUMMARY — all 2026-04-28 overnight experiments')
print('═'*72)
print()
print(f'{\\\"Experiment\\\":<35s} {\\\"Watchlist\\\":>9s} {\\\"Lookahead\\\":>9s} {\\\"OOS IC\\\":>9s} {\\\"Δ vs prod\\\":>10s}')
print('-'*72)
artifacts = [
    ('panel-ltr.json',                                      'production',  '103',  10),
    ('panel-ltr.20d.json',                                   'M1a 20d',     '227',  20),
    ('panel-ltr.60d.json',                                   'M1b 60d',     '227',  60),
    ('b1_regressed_20260428_020304/panel-ltr.json',          'B1 regressed','227',  10),
    ('panel-ltr.b1_1_retune.json',                          'F3 retune',    '227',  10),
    ('panel-ltr.b1_2_filtered_10d.json',                     'B1.2 filtered','75',  10),
    ('panel-ltr.b1_3_60d_tuned.json',                        'B1.3 60d-tuned','227', 60),
]
production_ic = 0.0400
for fn, label, wl, h in artifacts:
    try:
        d = json.load(open(f'backtesting/renquant_104/artifacts/{fn}'))
        ic = d.get('oos_mean_ic', float('nan'))
        delta = (ic - production_ic) / production_ic * 100
        print(f'{label:<35s} {wl:>9s} {h:>9d} {ic:>+9.5f} {delta:>+9.1f}%')
    except FileNotFoundError:
        print(f'{label:<35s} {wl:>9s} {h:>9d}        ?            ?')
\""

write_status "DONE" "All experiments complete; see logs/ablation_2026-04-27/ for details"
echo "[chain-b1.2-b1.3] $(date) — ALL DONE"
