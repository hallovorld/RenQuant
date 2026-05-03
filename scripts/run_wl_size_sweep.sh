#!/usr/bin/env bash
# WL size sweep — fills the IC-vs-wl-size curve between Stage 3 best (wl=173)
# and top-down ceiling (wl=281).
#
# For each target wl size N:
#   1. Build config with wl = production_wl_103 + first (N-103) tickers from
#      scripts/screen_stage2_results.json admitted list (alphabetical, deterministic).
#   2. Dispatch train_104 with 10-core saturation env vars set.
#   3. Read mean_ic from data/runs.db.
#
# Default sizes: 183, 203, 223, 243, 263  (5 points × ~30 min ≈ 2.5 h)
#
# Usage:
#   bash scripts/run_wl_size_sweep.sh
#   WL_SIZES="185 215 245" bash scripts/run_wl_size_sweep.sh
set -euo pipefail

# CLAUDE.md §5.10
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export VECLIB_MAXIMUM_THREADS=10
export NUMEXPR_NUM_THREADS=10

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

WL_SIZES="${WL_SIZES:-183 203 223 243 263}"
PER_RUN_TIMEOUT_SEC="${PER_RUN_TIMEOUT_SEC:-3000}"

echo "WL sweep targets: $WL_SIZES"
echo "Per-run timeout: ${PER_RUN_TIMEOUT_SEC}s"
echo

source ~/miniconda3/etc/profile.d/conda.sh
conda activate renquant

for wl_size in $WL_SIZES; do
    label="wl_sweep_${wl_size}"
    side_config="$REPO_ROOT/backtesting/renquant_104/strategy_config.${label}.json"
    log_path="/tmp/${label}.log"

    # Build side config: golden + first (wl_size - 103) tickers
    python3 << EOF
import json, sys
golden = json.load(open("$REPO_ROOT/backtesting/renquant_104/strategy_config.golden.json"))
s2 = json.load(open("$REPO_ROOT/scripts/screen_stage2_results.json"))
admitted = [r["ticker"] for r in s2["admitted"]]
baseline_wl = list(golden["watchlist"])
n_to_add = $wl_size - len(baseline_wl)
if n_to_add > len(admitted):
    print(f"ERROR: requested wl=${wl_size} (need {n_to_add} new) but only {len(admitted)} admitted candidates", file=sys.stderr)
    sys.exit(1)
new_wl = baseline_wl + [t for t in admitted[:n_to_add] if t not in set(baseline_wl)]
# In case some admitted overlap baseline (shouldn't), keep adding until target
i = n_to_add
while len(new_wl) < $wl_size and i < len(admitted):
    if admitted[i] not in set(baseline_wl):
        new_wl.append(admitted[i])
    i += 1
print(f"composing wl=${wl_size}: baseline={len(baseline_wl)} + {len(new_wl)-len(baseline_wl)} new = {len(new_wl)}")

cfg = json.loads(json.dumps(golden))
cfg["watchlist"] = new_wl
cfg["_audit_label"] = "$label"
cfg["panel_ltr"]["min_best_iter"] = 1
for k in ["panel-ltr", "ngboost-head", "panel-rank-calibration"]:
    p = f"artifacts/{k}.$label.json"
    if k == "panel-ltr":
        cfg["panel_ltr"]["artifact_path"] = p
        cfg["ranking"]["panel_scoring"]["artifact_path"] = p
    elif k == "ngboost-head":
        cfg["panel_ltr"]["ngboost"]["artifact_path"] = p
        cfg["ranking"]["panel_scoring"]["ngboost"]["artifact_path"] = p
    else:
        cfg["ranking"]["panel_scoring"]["global_calibration"]["artifact_path"] = p
json.dump(cfg, open("$side_config", "w"), indent=2)
print(f"wrote $side_config")
EOF

    echo
    echo "============================================================"
    echo "Training wl=${wl_size}  →  $log_path"
    echo "============================================================"
    t0=$(date +%s)
    # gtimeout from coreutils (brew install coreutils); fall back to plain
    # python invocation if gtimeout unavailable. Timeout is a safety net,
    # not the primary control — DB read after gives the real result.
    if command -v gtimeout &>/dev/null; then
        gtimeout "$PER_RUN_TIMEOUT_SEC" python scripts/train_104.py \
            --strategy-config-name "strategy_config.${label}.json" \
            --skip-baseline --skip-recalibrate --force \
            > "$log_path" 2>&1 || true
    else
        python scripts/train_104.py \
            --strategy-config-name "strategy_config.${label}.json" \
            --skip-baseline --skip-recalibrate --force \
            > "$log_path" 2>&1 || true
    fi
    elapsed=$(( $(date +%s) - t0 ))

    # Read result from DB regardless of subprocess exit (timeout-resilient)
    result=$(sqlite3 data/runs.db "SELECT oos_mean_ic, train_ic, n_tickers FROM training_runs WHERE artifact_type='panel-ltr' ORDER BY run_date DESC LIMIT 1;")
    echo "wl=${wl_size}  result: $result  elapsed=${elapsed}s"
done

echo
echo "All done. Summary (by wl size, latest mean_ic from DB):"
for wl_size in $WL_SIZES; do
    label="wl_sweep_${wl_size}"
    sqlite3 data/runs.db "SELECT printf('wl=%d  mean_ic=%+.4f  train_ic=%+.4f  n_tickers=%d', n_tickers, oos_mean_ic, train_ic, n_tickers) FROM training_runs WHERE artifact_path LIKE '%${label}%' AND artifact_type='panel-ltr' ORDER BY run_date DESC LIMIT 1;"
done
