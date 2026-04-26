#!/usr/bin/env bash
# Sim validation matrix — parallel launch.
#
# Strategy: run each variant in parallel as a separate process so all
# cores get utilised. Each writes to logs/sim_validations/{date}-{tag}.{md,json}
# atomically. Results survive process restart (idempotent — same flags
# produce same tag → same file path → re-run overwrites).
#
# Avoid duplicate runs: skip if the JSON output already exists for today.
#
# Usage:
#   bash scripts/run_validation_matrix.sh [start_date] [end_date]
#
# Default window: 2024-01-01 → today (27-mo OOS).

set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
PYTHON="/Users/renhao/miniconda3/envs/renquant/bin/python"
DATE=$(date +%Y-%m-%d)
START="${1:-2024-01-01}"
END="${2:-}"
OUT_DIR="$REPO_DIR/logs/sim_validations"
LOG_DIR="$REPO_DIR/logs/sim_validations/run_logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

cd "$REPO_DIR"

# Each entry: tag → CLI args
declare -a EXPERIMENTS=(
    # Variant 1 — baseline (no flag flips, golden v4.1 reference)
    "baseline|--baseline"

    # Buy logic stages: enable gates one at a time
    "gate-b0.20|--gate-b 0.20"
    "gate-b0.15|--gate-b 0.15"
    "gate-a-p85|--gate-a-pct 85"
    "gate-a-p75|--gate-a-pct 75"
    "gate-c-g3|--gate-c-gamma 3.0"

    # Combined buy gates
    "gate-b0.20+a-p85|--gate-b 0.20 --gate-a-pct 85"
    "gate-b0.20+a-p85+c-g3|--gate-b 0.20 --gate-a-pct 85 --gate-c-gamma 3.0"

    # Portfolio QP variants
    "qp|--qp-solver"
    "qp+decay0.5|--qp-solver --qp-signal-decay 0.5"
    "qp+robust0.5|--qp-solver --qp-robust-kappa 0.5"
    "qp+cvar1|--qp-solver --qp-cvar-lambda 1.0"

    # Full stack
    "full-stack|--gate-b 0.20 --gate-a-pct 85 --gate-c-gamma 3.0 --qp-solver --qp-signal-decay 0.5 --qp-robust-kappa 0.5 --qp-cvar-lambda 1.0"
)

PARALLEL_LIMIT=4   # 4 concurrent sims; ~30 min each, ~1.5 hr to clear queue

launch_one() {
    local tag="$1"
    local args="$2"
    local out_json="$OUT_DIR/$DATE-$tag.json"
    local out_log="$LOG_DIR/$DATE-$tag.log"

    if [ -f "$out_json" ]; then
        echo "  SKIP  $tag (already complete: $out_json)"
        return 0
    fi
    echo "  LAUNCH $tag → $out_log"
    end_args=""
    [ -n "$END" ] && end_args="--end $END"
    nohup "$PYTHON" scripts/validate_buy_logic.py \
        --start "$START" $end_args $args \
        > "$out_log" 2>&1 &
    return 0
}

echo "=== validation matrix start $(date) ==="
echo "    window: $START → ${END:-today}"
echo "    experiments: ${#EXPERIMENTS[@]}"
echo "    parallel:    $PARALLEL_LIMIT"
echo

for exp in "${EXPERIMENTS[@]}"; do
    tag="${exp%%|*}"
    args="${exp#*|}"
    # Limit concurrency
    while [ "$(jobs -p | wc -l | tr -d ' ')" -ge "$PARALLEL_LIMIT" ]; do
        sleep 5
    done
    launch_one "$tag" "$args"
done

# Wait for all background jobs
wait
echo "=== validation matrix DONE $(date) ==="
echo "Reports in $OUT_DIR/$DATE-*.md"
