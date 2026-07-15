#!/usr/bin/env bash
# RETIRED (2026-07-14, PR #471 r5): completed BB DOE sweep.
# run_sim_104.py now requires strict-pinned configs or an experiment manifest;
# --strategy-config-name with local sweep configs no longer works.
# To re-run: register experiment manifests in experiments/manifests/ and
# use --experiment-manifest instead.
echo "ERROR: this script is RETIRED — see header for migration instructions" >&2
exit 1
# --- original below for reference ---
# Orchestrate the 27-run BB sweep in 3 batches of 9.
# Batch 1 must already be launched before this script starts; this script
# waits for batch 1 to finish, then launches batch 2, then batch 3.
set -uo pipefail

REPO="/Users/renhao/git/github/RenQuant"
cd "$REPO"

wait_for_batch() {
    local label="$1"
    while pgrep -fl "run_sim_104.*sim_BB_" > /dev/null 2>&1; do
        sleep 30
    done
    echo "[$(date '+%H:%M:%S')] batch $label complete"
}

launch_batch() {
    local label="$1"
    shift
    local ts=$(date +%H%M%S)
    for i in "$@"; do
        nohup .venv/bin/python scripts/run_sim_104.py \
            --start 2024-04-01 --end 2026-03-26 \
            --strategy-config-name "strategy_config.sim_BB_${i}.json" \
            --no-persist --no-compare \
            > "data/logs/wf_sim_BB_${i}_${ts}.log" 2>&1 &
    done
    echo "[$(date '+%H:%M:%S')] launched batch $label (ts=${ts}) — 9 sims"
}

echo "[$(date '+%H:%M:%S')] orchestrator started — waiting for batch 1"

wait_for_batch "1"
launch_batch "2" 09 10 11 12 13 14 15 16 17
wait_for_batch "2"
launch_batch "3" 18 19 20 21 22 23 24 25 26
wait_for_batch "3"

echo "[$(date '+%H:%M:%S')] all 27 BB sims complete"
.venv/bin/python scripts/_doe_fit_response_surface.py > data/logs/bb_analysis.log 2>&1
echo "[$(date '+%H:%M:%S')] response surface fit → data/logs/bb_analysis.log"

# ── Auto-chain into Track B (meta-label pipeline) ────────────────────
echo "[$(date '+%H:%M:%S')] building BB-optimum config from response-surface output …"
.venv/bin/python scripts/_doe_build_optimum_config.py >> data/logs/bb_analysis.log 2>&1
echo "[$(date '+%H:%M:%S')] launching Track B meta-label pipeline (chronological split) …"
bash scripts/_meta_label_pipeline.sh strategy_config.sim_BB_optimum.json \
    > data/logs/meta_label_pipeline.log 2>&1 &
echo "[$(date '+%H:%M:%S')] Track B running in background (PID=$!); see meta_label_pipeline.log"
