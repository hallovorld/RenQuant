#!/usr/bin/env bash
# monthly_meta_label_retrain.sh — Re-train the meta-label exit classifier
# on a rolling 12-month window of fresh snapshot data.
#
# Why monthly: meta-label artifact stays valid only while the
# panel-LTR score distribution + path-rule trigger patterns remain
# similar to training. Cadence rationale (CLAUDE.md §5.13.6):
#   * 12-mo training window produces ~146 events (P4.5 baseline)
#   * Monthly = ~12 new event-bars/month → ~8% new info per tick →
#     meaningful refresh without daily-thrash retraining.
#
# Schedule: 1st of every month, 03:30 PT (after monthly calibrator).
# Plist: scripts/launchd/com.renquant.monthly-meta-label-retrain.plist
#
# Steps:
#   1. Range-find training window: today − 60d (lookahead safety) − 12mo
#   2. Run snapshot-collection sim on that window (writes parquet)
#   3. Apply triple-barrier labels (López de Prado AFML ch.3)
#   4. Train XGBoost with PurgedKFold CV (AFML ch.7)
#   5. Atomic swap artifact (backup old, deploy new)
#   6. Health checks: AUC ≥ 0.52, n_events ≥ 100, balance in [0.3, 0.7]
#   7. ntfy summary
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
VENV_DIR="/Users/renhao/git/github/RenQuant/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$REPO_DIR/logs/monthly_meta_label"
ART_DIR="$REPO_DIR/backtesting/renquant_104/artifacts"
NTFY_TOPIC="renquant"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"

notify() {
    local title="$1" body="$2"
    curl -s -H "Title: $title" -d "$body" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
}

cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"

# ── Compute training window: [today − 60d − 365d, today − 60d] ──────
# 60d = lookahead_days safety buffer (fwd_60d_excess label horizon).
TRAIN_END=$(date -v-60d +%Y-%m-%d 2>/dev/null || date -d "today - 60 days" +%Y-%m-%d)
TRAIN_START=$(date -v-60d -v-365d +%Y-%m-%d 2>/dev/null || date -d "today - 60 days - 365 days" +%Y-%m-%d)
echo "[$(date '+%H:%M:%S')] Monthly meta-label retrain — training window $TRAIN_START → $TRAIN_END" | tee -a "$LOG"

# ── Step 2: snapshot sim on prior 12 months ─────────────────────────
SNAP_CFG="strategy_config.sim_monthly_retrain_snapshot.json"
SNAP_OUT="data/monthly_meta_label_snapshots_${DATE}.parquet"
LABEL_OUT="data/monthly_meta_label_labels_${DATE}.parquet"
NEW_ARTIFACT="$ART_DIR/meta-label-exit.candidate-${DATE}.json"
PROD_ARTIFACT="$ART_DIR/meta-label-exit.json"

$PYTHON <<PY 2>&1 | tee -a "$LOG"
import json
src = json.load(open("backtesting/renquant_104/strategy_config.json"))
src["_side_config_label"] = "sim_monthly_retrain_snapshot"
src["meta_label_training"] = {"enabled": True, "output_path": "$SNAP_OUT"}
src.setdefault("ranking", {})["meta_label"] = {"enabled": False}
json.dump(src, open("backtesting/renquant_104/$SNAP_CFG", "w"), indent=2)
print(f"Built snapshot config: $SNAP_CFG")
PY

echo "[$(date '+%H:%M:%S')] Step 2: snapshot sim …" | tee -a "$LOG"
$PYTHON scripts/run_sim_104.py \
    --start "$TRAIN_START" --end "$TRAIN_END" \
    --strategy-config-name "$SNAP_CFG" \
    --no-persist --no-compare >> "$LOG" 2>&1

if [ ! -f "$SNAP_OUT" ]; then
    notify "META-LABEL RETRAIN ✗" "snapshot parquet missing — check $LOG"
    exit 1
fi

# ── Step 3: triple-barrier labels ───────────────────────────────────
echo "[$(date '+%H:%M:%S')] Step 3: label …" | tee -a "$LOG"
$PYTHON scripts/_meta_label_generate.py \
    --snapshots "$SNAP_OUT" \
    --out       "$LABEL_OUT" \
    --pt-mult 10 --sl-mult 10 --fwd-window 20 >> "$LOG" 2>&1

# ── Step 4: train classifier ────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Step 4: train …" | tee -a "$LOG"
$PYTHON scripts/_meta_label_train.py \
    --labels "$LABEL_OUT" \
    --out    "$NEW_ARTIFACT" \
    --n-splits 5 --label-horizon-days 20 --pct-embargo 0.02 \
    >> "$LOG" 2>&1

if [ ! -f "$NEW_ARTIFACT" ]; then
    notify "META-LABEL RETRAIN ✗" "training failed — check $LOG"
    exit 1
fi

# ── Step 5: health gates BEFORE swap ────────────────────────────────
HEALTH=$($PYTHON <<PY
import json
art = json.load(open("$NEW_ARTIFACT"))
cv = art.get("cv_metrics", {})
td = art.get("training_data_summary", {})
auc = cv.get("auc_mean", 0.0)
n_events = td.get("n_events", 0)
balance = td.get("class_balance", 0.5)
n_features = td.get("feature_count", 0)

problems = []
if auc < 0.52:
    problems.append(f"AUC_LOW:{auc:.3f}")
if n_events < 100:
    problems.append(f"NEVENTS_LOW:{n_events}")
if balance < 0.30 or balance > 0.70:
    problems.append(f"BAL_OFF:{balance:.2f}")
if n_features < 25:
    problems.append(f"FEAT_LOW:{n_features}")

if problems:
    print("FAIL:" + ",".join(problems))
else:
    print(f"OK:auc={auc:.3f} n={n_events} bal={balance:.2f} feats={n_features}")
PY
)
echo "[$(date '+%H:%M:%S')] Health: $HEALTH" | tee -a "$LOG"

if [[ "$HEALTH" == FAIL:* ]]; then
    notify "META-LABEL RETRAIN ✗" "health gate FAIL: $HEALTH (keeping prior artifact)"
    rm -f "$NEW_ARTIFACT"
    exit 1
fi

# ── Step 6: atomic swap (backup → swap) ──────────────────────────────
BACKUP="$PROD_ARTIFACT.backup-$(date +%Y-%m-%d_%H%M%S)"
if [ -f "$PROD_ARTIFACT" ]; then
    cp "$PROD_ARTIFACT" "$BACKUP"
fi
mv "$NEW_ARTIFACT" "$PROD_ARTIFACT"
echo "[$(date '+%H:%M:%S')] Atomic swap complete. Backup: $BACKUP" | tee -a "$LOG"

# ── Step 7: ntfy + cleanup old backups (keep last 6) ────────────────
notify "META-LABEL RETRAIN ✓" "$DATE  $HEALTH  artifact swapped"
ls -t "$ART_DIR"/meta-label-exit.json.backup-* 2>/dev/null \
    | tail -n +7 | xargs -r rm -f

echo "[$(date '+%H:%M:%S')] DONE" | tee -a "$LOG"
