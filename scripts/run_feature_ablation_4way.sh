#!/usr/bin/env bash
# Feature ablation 4-way at wl=183 (sweep peak).
#
# Two new factors landed 2026-05-03: idio_vol_z (Ang 2006 IVOL puzzle) and
# mom_1m_reversal_z (Jegadeesh 1990 1-month reversal). 27 production features
# also include 8 raw technicals (adx, cci, bbp, williams_r, trend, trend_long,
# rel_mom_20d, rel_mom_60d) that have low literature support and may be
# dragging on IC. Four arms isolate the lift from each side independently.
#
#   A (drop8)   : drop 8 weak raw technicals       → 19 features
#   B (add2)    : keep 27 + new IVOL + reversal    → 29 features
#   C (ultra)   : drop 8 + add 2                   → 21 features
#   D (control) : current 27 (drop the new 2)      → 27 features  (baseline)
#
# Wallclock: 4 retrains × ~50min ≈ 3h 20min.
# Promotion gate: best arm wins ONLY after §5.2 sanity triple
# (A/A + shuffled-label + time-shift placebo) clears.
set -euo pipefail

# CLAUDE.md §5.10
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export VECLIB_MAXIMUM_THREADS=10
export NUMEXPR_NUM_THREADS=10

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

GOLDEN="$REPO_ROOT/backtesting/renquant_104/strategy_config.golden.json"
S2="$REPO_ROOT/scripts/screen_stage2_results.json"
WL_SIZE=183

WEAK_TECHNICALS='["adx","cci","bbp","williams_r","trend","trend_long","rel_mom_20d","rel_mom_60d"]'
NEW_FACTORS='["idio_vol_z","mom_1m_reversal_z"]'

build_config() {
    local label="$1"
    local extra_drops="$2"
    local out_path="$REPO_ROOT/backtesting/renquant_104/strategy_config.${label}.json"
    python3 - <<PYEOF
import json
golden = json.load(open("$GOLDEN"))
s2 = json.load(open("$S2"))
admitted = [r["ticker"] for r in s2["admitted"]]
baseline_wl = list(golden["watchlist"])
n_to_add = $WL_SIZE - len(baseline_wl)
new_wl = baseline_wl + [t for t in admitted[:n_to_add] if t not in set(baseline_wl)]
i = n_to_add
while len(new_wl) < $WL_SIZE and i < len(admitted):
    if admitted[i] not in set(baseline_wl):
        new_wl.append(admitted[i])
    i += 1

cfg = json.loads(json.dumps(golden))
cfg["watchlist"] = new_wl
cfg["_audit_label"] = "$label"
cfg["panel_ltr"]["min_best_iter"] = 1

extra_drops = $extra_drops
existing_drops = list(cfg["panel_ltr"].get("drop_cols", []))
cfg["panel_ltr"]["drop_cols"] = sorted(set(existing_drops + extra_drops))

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

json.dump(cfg, open("$out_path", "w"), indent=2)
print(f"wrote $out_path  drops={len(cfg['panel_ltr']['drop_cols'])}  wl={len(new_wl)}")
PYEOF
}

# Build 4 configs (idempotent — overwrites if rerun)
build_config "ablation_A_drop8"   "$WEAK_TECHNICALS + $NEW_FACTORS"
build_config "ablation_B_add2"    "[]"
build_config "ablation_C_ultra"   "$WEAK_TECHNICALS"
build_config "ablation_D_control" "$NEW_FACTORS"

if [ "${BUILD_ONLY:-0}" = "1" ]; then
    echo
    echo "BUILD_ONLY=1 — exiting before training dispatch."
    exit 0
fi

# Safety: refuse to dispatch while another train_104 is running (would
# starve the sweep + fight for cores). Set ALLOW_PARALLEL=1 to override.
if [ "${ALLOW_PARALLEL:-0}" != "1" ]; then
    if pgrep -f "train_104.py" > /dev/null; then
        echo "ERROR: train_104.py already running. Wait for sweep to finish, or set ALLOW_PARALLEL=1." >&2
        ps -ef | grep "[t]rain_104.py" >&2
        exit 5
    fi
fi

echo
echo "Configs built. Dispatching 4 retrains sequentially..."
echo

source ~/miniconda3/etc/profile.d/conda.sh
conda activate renquant

for arm in ablation_A_drop8 ablation_B_add2 ablation_C_ultra ablation_D_control; do
    log="/tmp/${arm}.log"
    echo
    echo "============================================================"
    echo "Training $arm  →  $log"
    echo "============================================================"
    t0=$(date +%s)
    python scripts/train_104.py \
        --strategy-config-name "strategy_config.${arm}.json" \
        --skip-baseline --skip-recalibrate --force \
        > "$log" 2>&1 || true
    elapsed=$(( $(date +%s) - t0 ))
    result=$(sqlite3 data/runs.db "SELECT printf('oos_ic=%+.4f train_ic=%+.4f n_features=%d', oos_mean_ic, train_ic, n_features) FROM training_runs WHERE artifact_path LIKE '%${arm}%' AND artifact_type='panel-ltr' ORDER BY run_date DESC LIMIT 1;")
    echo "$arm  $result  elapsed=${elapsed}s"
done

echo
echo "=== Ablation summary ==="
for arm in ablation_A_drop8 ablation_B_add2 ablation_C_ultra ablation_D_control; do
    sqlite3 data/runs.db "SELECT printf('${arm}  oos_ic=%+.4f  train_ic=%+.4f  n_features=%d', oos_mean_ic, train_ic, n_features) FROM training_runs WHERE artifact_path LIKE '%${arm}%' AND artifact_type='panel-ltr' ORDER BY run_date DESC LIMIT 1;"
done
