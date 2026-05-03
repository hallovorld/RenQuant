#!/usr/bin/env bash
# Auto-dispatch a B2 hold-out simulation on the best Stage 3 batch artifact.
#
# Run this AFTER Stage 3 (run_stage3_greedy.py) finishes. It:
#   1. Reads scripts/stage3_progress.json
#   2. Finds the highest-mean_ic accepted batch
#   3. Builds a B2 hold-out config pointing to that batch's panel-ltr artifact
#   4. Dispatches scripts/holdout_backtest.py
#   5. Reports APY / Sharpe / Calmar / MaxDD
#
# Usage:
#   bash scripts/run_b2_on_stage3_winner.sh
#   bash scripts/run_b2_on_stage3_winner.sh --train-end 2024-12-31 --sim-start 2025-01-02
#
# Wallclock: ~30-40 min for the sim (uses one CPU bound, won't conflict with
# anything else).
set -euo pipefail

# CLAUDE.md §5.10: saturate hardware
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export VECLIB_MAXIMUM_THREADS=10
export NUMEXPR_NUM_THREADS=10

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROGRESS="$REPO_ROOT/scripts/stage3_progress.json"

if [ ! -f "$PROGRESS" ]; then
    echo "ERROR: $PROGRESS not found — did Stage 3 run?" >&2
    exit 1
fi

# Find best accepted batch via Python (no jq dep)
read -r BEST_IDX BEST_IC <<EOF
$(python3 -c "
import json
prog = json.load(open('$PROGRESS'))
best = max(
    (b for b in prog['batches'] if b.get('accepted') and b.get('new_ic') is not None),
    key=lambda b: b['new_ic'],
    default=None,
)
if best is None:
    raise SystemExit('No accepted batches found in progress.json')
print(f'{best[\"batch_idx\"]:03d} {best[\"new_ic\"]}')
")
EOF

LABEL="stage3_batch_${BEST_IDX}"
ARTIFACT="$REPO_ROOT/backtesting/renquant_104/artifacts/panel-ltr.${LABEL}.json"
SIDE_CONFIG="$REPO_ROOT/backtesting/renquant_104/strategy_config.${LABEL}.json"

if [ ! -f "$ARTIFACT" ]; then
    echo "ERROR: artifact $ARTIFACT not found" >&2
    exit 2
fi
if [ ! -f "$SIDE_CONFIG" ]; then
    echo "ERROR: side config $SIDE_CONFIG not found" >&2
    exit 3
fi

echo "Best Stage 3 batch: $LABEL  CPCV mean_ic=$BEST_IC"
echo "Artifact: $ARTIFACT"
echo "Side config: $SIDE_CONFIG"
echo

# Default B2 window: train through 2024 calendar end, sim 2025-01-02 → 2026-04-30
TRAIN_END="${TRAIN_END:-2024-12-31}"
SIM_START="${SIM_START:-2025-01-02}"
SIM_END="${SIM_END:-2026-04-30}"

# Override defaults via flags
while [ "${1:-}" != "" ]; do
    case "$1" in
        --train-end) TRAIN_END="$2"; shift 2 ;;
        --sim-start) SIM_START="$2"; shift 2 ;;
        --sim-end)   SIM_END="$2";   shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 4 ;;
    esac
done

LOG_PATH="/tmp/b2_${LABEL}.log"
echo "Dispatching B2 hold-out sim:"
echo "  train-end:  $TRAIN_END"
echo "  sim window: $SIM_START → $SIM_END"
echo "  artifact:   panel-ltr.${LABEL}.json"
echo "  log:        $LOG_PATH"
echo

cd "$REPO_ROOT"

# --skip-train reuses the existing Stage 3 batch's panel-ltr + ngboost-head
# artifacts (no need to retrain). Sim only.
python scripts/holdout_backtest.py \
    --strategy-config-name "strategy_config.${LABEL}.json" \
    --train-end "$TRAIN_END" \
    --sim-start "$SIM_START" \
    --sim-end "$SIM_END" \
    --skip-train \
    > "$LOG_PATH" 2>&1

# Extract summary
echo
echo "=== B2 result ==="
grep -E "apy_holdout|sharpe_holdout|sortino_holdout|calmar_holdout|max_dd_holdout|win_rate_holdout|buys / sells|wall seconds" "$LOG_PATH" | tail -10
