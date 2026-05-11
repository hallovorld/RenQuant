#!/usr/bin/env bash
# End-to-end Track B orchestrator — runs the chronological-split
# meta-label experiment after Track A (BB sweep) produces BB-optimum.
#
# Invariants enforced:
#   * Train + test windows are NON-OVERLAPPING (CLAUDE.md §5.14.4 — no
#     look-ahead leakage between meta-label train and OOS test).
#   * The snapshot-collection sim runs the SAME stop-loss config as the
#     final deployment sim (§5.13.10 — no distribution shift between
#     train and deploy).
#
# Usage::
#
#     bash scripts/_meta_label_pipeline.sh \
#         strategy_config.sim_BB_optimum.json   # post-DOE config
#
# Output:
#     data/position_day_snapshots.parquet
#     data/position_day_labels.parquet
#     backtesting/renquant_104/artifacts/meta-label-exit.json
#     data/logs/meta_label_*.log
#     3-way comparison printed at end
set -euo pipefail

REPO="/Users/renhao/git/github/RenQuant"
cd "$REPO"
source .venv/bin/activate

# Configurable via env vars, defaults match the doc
TRAIN_START="${TRAIN_START:-2024-04-01}"
TRAIN_END="${TRAIN_END:-2025-04-01}"
TEST_START="${TEST_START:-2025-04-01}"
TEST_END="${TEST_END:-2026-03-26}"
OPTIMUM_CFG="${1:-strategy_config.sim_BB_24.json}"   # default to BB center if no arg

STAGE_DIR="backtesting/renquant_104"
LOGS="data/logs"
mkdir -p "$LOGS"

echo "[$(date '+%H:%M:%S')] === Track B chronological-split meta-label pipeline ==="
echo "  optimum config: $OPTIMUM_CFG"
echo "  train window:   $TRAIN_START → $TRAIN_END"
echo "  test  window:   $TEST_START → $TEST_END"
echo

# ── Step 1: Build the SNAPSHOT-COLLECTION config ──────────────────────
# Clone BB-optimum + enable snapshot logger + disable veto (no model yet)
SNAP_CFG="strategy_config.sim_metalabel_snapshot.json"
python <<PY
import json
src  = json.load(open("$STAGE_DIR/$OPTIMUM_CFG"))
src["_side_config_label"] = "sim_metalabel_snapshot"
src["meta_label_training"] = {
    "enabled":     True,
    "output_path": "data/position_day_snapshots.parquet",
}
# Disable any veto for the SNAPSHOT run — we're COLLECTING data
src.setdefault("ranking", {})["meta_label"] = {"enabled": False}
json.dump(src, open("$STAGE_DIR/$SNAP_CFG", "w"), indent=2)
print("OK: built $SNAP_CFG")
PY

# ── Step 2: Snapshot-collection sim on the TRAINING window ────────────
echo "[$(date '+%H:%M:%S')] Step 2: snapshot sim ($TRAIN_START → $TRAIN_END) …"
python scripts/run_sim_104.py \
    --start "$TRAIN_START" --end "$TRAIN_END" \
    --strategy-config-name "$SNAP_CFG" \
    --no-persist --no-compare \
    > "$LOGS/meta_label_snapshot_sim.log" 2>&1
echo "[$(date '+%H:%M:%S')]   done. tail:"
tail -3 "$LOGS/meta_label_snapshot_sim.log"

# ── Step 3: Triple-barrier label generator ────────────────────────────
echo "[$(date '+%H:%M:%S')] Step 3: triple-barrier labeling …"
python scripts/_meta_label_generate.py \
    --snapshots data/position_day_snapshots.parquet \
    --out       data/position_day_labels.parquet \
    --pt-mult 10 --sl-mult 10 --fwd-window 20 \
    > "$LOGS/meta_label_generate.log" 2>&1
tail -3 "$LOGS/meta_label_generate.log"

# ── Step 4: Train XGBoost meta-label classifier ──────────────────────
echo "[$(date '+%H:%M:%S')] Step 4: train meta-label XGBoost …"
python scripts/_meta_label_train.py \
    --labels data/position_day_labels.parquet \
    --out "$STAGE_DIR/artifacts/meta-label-exit.json" \
    --n-splits 5 \
    --label-horizon-days 20 \
    --pct-embargo 0.02 \
    > "$LOGS/meta_label_train.log" 2>&1
tail -8 "$LOGS/meta_label_train.log"

# ── Step 5: Build DEPLOY configs (3-way: baseline / BB-opt / BB-opt+meta) ─
DEPLOY_OPT_CFG="strategy_config.sim_metalabel_deploy_bbopt.json"
DEPLOY_META_CFG="strategy_config.sim_metalabel_deploy_meta.json"
python <<PY
import json
# BB-opt only (no meta) — for OOS comparison
src = json.load(open("$STAGE_DIR/$OPTIMUM_CFG"))
src["_side_config_label"] = "sim_metalabel_deploy_bbopt"
src.setdefault("ranking", {})["meta_label"] = {"enabled": False}
json.dump(src, open("$STAGE_DIR/$DEPLOY_OPT_CFG", "w"), indent=2)

# BB-opt + meta-label veto enabled
src = json.load(open("$STAGE_DIR/$OPTIMUM_CFG"))
src["_side_config_label"] = "sim_metalabel_deploy_meta"
src.setdefault("ranking", {})["meta_label"] = {
    "enabled": True,
    "threshold": 0.5,
    "artifact_path": "backtesting/renquant_104/artifacts/meta-label-exit.json",
}
json.dump(src, open("$STAGE_DIR/$DEPLOY_META_CFG", "w"), indent=2)
print("OK: built deploy configs")
PY

# ── Step 6: 3-way OOS sims in parallel ───────────────────────────────
echo "[$(date '+%H:%M:%S')] Step 6: 3-way OOS sims ($TEST_START → $TEST_END) in parallel …"
TS=$(date +%H%M%S)
for cfg in strategy_config.sim_baseline.json $DEPLOY_OPT_CFG $DEPLOY_META_CFG; do
    label=$(echo "$cfg" | sed 's/strategy_config\.//;s/\.json//')
    nohup python scripts/run_sim_104.py \
        --start "$TEST_START" --end "$TEST_END" \
        --strategy-config-name "$cfg" \
        --no-persist --no-compare \
        > "$LOGS/meta_label_oos_${label}_${TS}.log" 2>&1 &
done
wait
echo "[$(date '+%H:%M:%S')]   done."

# ── Step 7: Parse + tabulate ─────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] === FINAL 3-way OOS comparison ($TEST_START → $TEST_END) ==="
for f in "$LOGS"/meta_label_oos_*_${TS}.log; do
    label=$(basename "$f" | sed "s/meta_label_oos_//;s/_${TS}\.log//")
    summary=$(grep -E "Risk: Sharpe|^Final value" "$f" | head -2)
    echo "  --- $label ---"
    echo "$summary"
done

echo "[$(date '+%H:%M:%S')] === Track B complete ==="
