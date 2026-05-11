#!/usr/bin/env bash
# Walk-forward validation — train meta-label on rolling 6-mo windows,
# evaluate on the next 6 months. 3 disjoint windows total.
#
# Window plan (all within 2024-04 → 2026-03 OOS):
#   W1: train 2024-04→2024-10   test 2024-10→2025-04
#   W2: train 2024-10→2025-04   test 2025-04→2025-10
#   W3: train 2025-04→2025-10   test 2025-10→2026-03
#
# For each window, runs the full P4.5 chain (snapshot → label → train →
# 3-way OOS). Re-uses _meta_label_pipeline.sh logic by exporting
# TRAIN_START/TRAIN_END/TEST_START/TEST_END env vars per-window.
#
# Output:
#   data/logs/wf_meta_W{1,2,3}/   — per-window artifacts
#   data/logs/wf_meta_summary.json — aggregated 3×3=9 OOS results
set -uo pipefail

REPO="/Users/renhao/git/github/RenQuant"
cd "$REPO"
source .venv/bin/activate

# BB_14 is the empirical APY-best stop-loss config from Track A
OPTIMUM_CFG="${1:-strategy_config.sim_BB_14.json}"

run_window() {
    local W="$1"; local TRS="$2"; local TRE="$3"; local TES="$4"; local TEE="$5"
    local WDIR="$REPO/data/logs/wf_meta_$W"
    mkdir -p "$WDIR"
    echo
    echo "[$(date '+%H:%M:%S')] ==== Window $W: train [$TRS,$TRE]  test [$TES,$TEE] ===="

    # Distinct artifact paths per window so they don't overwrite each other
    SNAP_CFG="strategy_config.sim_wf_${W}_snapshot.json"
    DEPLOY_BB_CFG="strategy_config.sim_wf_${W}_deploy_bbopt.json"
    DEPLOY_META_CFG="strategy_config.sim_wf_${W}_deploy_meta.json"
    ARTIFACT_PATH="backtesting/renquant_104/artifacts/meta-label-exit-${W}.json"
    SNAPSHOT_OUT="data/wf_meta_${W}_snapshots.parquet"
    LABEL_OUT="data/wf_meta_${W}_labels.parquet"

    # ── Step 1: snapshot-collection config ────────────────────────────
    python <<PY
import json
src = json.load(open("backtesting/renquant_104/$OPTIMUM_CFG"))
src["_side_config_label"] = "sim_wf_${W}_snapshot"
src["meta_label_training"] = {"enabled": True, "output_path": "$SNAPSHOT_OUT"}
src.setdefault("ranking", {})["meta_label"] = {"enabled": False}
json.dump(src, open("backtesting/renquant_104/$SNAP_CFG", "w"), indent=2)
PY

    # ── Step 2: snapshot sim (training window) ────────────────────────
    echo "[$(date '+%H:%M:%S')]   Step 2: snapshot sim ($TRS → $TRE)…"
    python scripts/run_sim_104.py \
        --start "$TRS" --end "$TRE" \
        --strategy-config-name "$SNAP_CFG" \
        --no-persist --no-compare \
        > "$WDIR/snapshot.log" 2>&1

    # ── Step 3: triple-barrier labels ─────────────────────────────────
    echo "[$(date '+%H:%M:%S')]   Step 3: labels…"
    python scripts/_meta_label_generate.py \
        --snapshots "$SNAPSHOT_OUT" \
        --out "$LABEL_OUT" \
        --pt-mult 10 --sl-mult 10 --fwd-window 20 \
        > "$WDIR/generate.log" 2>&1

    # ── Step 4: train per-window classifier ───────────────────────────
    echo "[$(date '+%H:%M:%S')]   Step 4: train…"
    python scripts/_meta_label_train.py \
        --labels "$LABEL_OUT" \
        --out "$ARTIFACT_PATH" \
        --n-splits 5 --label-horizon-days 20 --pct-embargo 0.02 \
        > "$WDIR/train.log" 2>&1

    # ── Step 5: build deploy configs ──────────────────────────────────
    python <<PY
import json
src = json.load(open("backtesting/renquant_104/$OPTIMUM_CFG"))
src["_side_config_label"] = "sim_wf_${W}_deploy_bbopt"
src.setdefault("ranking", {})["meta_label"] = {"enabled": False}
json.dump(src, open("backtesting/renquant_104/$DEPLOY_BB_CFG", "w"), indent=2)
src = json.load(open("backtesting/renquant_104/$OPTIMUM_CFG"))
src["_side_config_label"] = "sim_wf_${W}_deploy_meta"
src.setdefault("ranking", {})["meta_label"] = {
    "enabled": True, "threshold": 0.5,
    "artifact_path": "$ARTIFACT_PATH",
}
json.dump(src, open("backtesting/renquant_104/$DEPLOY_META_CFG", "w"), indent=2)
PY

    # ── Step 6: 3-way OOS sims (parallel within window) ───────────────
    echo "[$(date '+%H:%M:%S')]   Step 6: 3-way OOS ($TES → $TEE) parallel…"
    for cfg_short in baseline ${DEPLOY_BB_CFG%.json} ${DEPLOY_META_CFG%.json}; do
        cfg_file="strategy_config.${cfg_short}.json"
        # baseline special case
        if [ "$cfg_short" = "baseline" ]; then
            cfg_file="strategy_config.sim_baseline.json"
        fi
        label=$(basename "$cfg_file" .json | sed 's/strategy_config\.//')
        nohup python scripts/run_sim_104.py \
            --start "$TES" --end "$TEE" \
            --strategy-config-name "$cfg_file" \
            --no-persist --no-compare \
            > "$WDIR/oos_${label}.log" 2>&1 &
    done
    wait
    echo "[$(date '+%H:%M:%S')]   Step 6: window $W complete."

    # ── Step 7: parse & print window-level results ────────────────────
    echo "[$(date '+%H:%M:%S')]   $W results:"
    for f in "$WDIR"/oos_*.log; do
        label=$(basename "$f" .log | sed 's/oos_//')
        line=$(grep "Risk: Sharpe" "$f" | head -1)
        apy=$(grep "APY:" "$f" | head -1 | grep -oE "APY: [+-]?[0-9.]+%" || echo "")
        echo "    $label  $line  $apy"
    done
}

run_window "W1" "2024-04-01" "2024-10-01" "2024-10-01" "2025-04-01"
run_window "W2" "2024-10-01" "2025-04-01" "2025-04-01" "2025-10-01"
run_window "W3" "2025-04-01" "2025-10-01" "2025-10-01" "2026-03-26"

echo
echo "[$(date '+%H:%M:%S')] ==== Walk-forward 3-window validation complete ===="
echo
echo "Aggregating into summary JSON…"
python scripts/_meta_label_wf_aggregate.py > data/logs/wf_meta_summary.log 2>&1
tail -40 data/logs/wf_meta_summary.log
