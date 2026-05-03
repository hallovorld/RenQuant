#!/usr/bin/env bash
# Top-down complement to Stage 3: train ONE model on the maximum candidate
# universe (wl103 + all Stage-2-admitted) to get the "ceiling IC" in
# a single retrain (~30 min) instead of 18 sequential batches (~10 h).
#
# Together with Stage 3 greedy results this answers two questions:
#   * Top-down IC ≥ Stage 3 best  → greedy was too conservative, ship full set
#   * Top-down IC < Stage 3 best  → greedy correctly screened out toxic batches
#   * Top-down IC < baseline      → expansion fundamentally fails (full set toxic)
#
# Run this AFTER Stage 3 finishes:
#   bash scripts/run_topdown_wl_max.sh
set -euo pipefail

# CLAUDE.md §5.10: saturate hardware
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export VECLIB_MAXIMUM_THREADS=10
export NUMEXPR_NUM_THREADS=10

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

STAGE2="$REPO_ROOT/scripts/screen_stage2_results.json"
GOLDEN="$REPO_ROOT/backtesting/renquant_104/strategy_config.golden.json"
LABEL="topdown_wl_max"
SIDE_CONFIG="$REPO_ROOT/backtesting/renquant_104/strategy_config.${LABEL}.json"

# Build side config: golden settings + max wl (103 + 178 admitted = 281)
python3 << EOF
import json
golden = json.load(open("$GOLDEN"))
s2 = json.load(open("$STAGE2"))
admitted = [r["ticker"] for r in s2["admitted"]]
baseline_wl = list(golden["watchlist"])
new_wl = baseline_wl + [t for t in admitted if t not in set(baseline_wl)]
print(f"baseline wl: {len(baseline_wl)}, admitted: {len(admitted)}, combined: {len(new_wl)}")

cfg = json.loads(json.dumps(golden))
cfg["watchlist"] = new_wl
cfg["_audit_label"] = "$LABEL"
cfg["panel_ltr"]["min_best_iter"] = 1   # bypass strict guard (eval_ic_floor=0.02 still active)

# Side artifact paths — DO NOT clobber prod
for k in ["panel-ltr", "ngboost-head", "panel-rank-calibration"]:
    p = f"artifacts/{k}.$LABEL.json"
    if k == "panel-ltr":
        cfg["panel_ltr"]["artifact_path"] = p
        cfg["ranking"]["panel_scoring"]["artifact_path"] = p
    elif k == "ngboost-head":
        cfg["panel_ltr"]["ngboost"]["artifact_path"] = p
        cfg["ranking"]["panel_scoring"]["ngboost"]["artifact_path"] = p
    else:
        cfg["ranking"]["panel_scoring"]["global_calibration"]["artifact_path"] = p

json.dump(cfg, open("$SIDE_CONFIG", "w"), indent=2)
print(f"wrote $SIDE_CONFIG")
EOF

LOG_PATH="/tmp/topdown_wl_max.log"
echo
echo "Dispatching top-down training..."
echo "  config: $SIDE_CONFIG"
echo "  log:    $LOG_PATH"
echo

source ~/miniconda3/etc/profile.d/conda.sh
conda activate renquant
python scripts/train_104.py \
    --strategy-config-name "strategy_config.${LABEL}.json" \
    --skip-baseline --skip-recalibrate --force \
    > "$LOG_PATH" 2>&1

echo
echo "=== Top-down result ==="
sqlite3 data/runs.db "SELECT run_id, oos_mean_ic, train_ic, n_tickers, n_features FROM training_runs WHERE artifact_type='panel-ltr' ORDER BY run_date DESC LIMIT 1;"
echo
echo "Compare with Stage 3 best (in scripts/stage3_progress.json) and"
echo "production baseline (+0.034) to interpret."
