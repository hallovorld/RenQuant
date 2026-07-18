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
# Steps (redesign: doc/design/2026-07-18-metalabel-monthly-retrain-redesign.md):
#   0. CONSUMER GATE (§2.1): if ranking.meta_label.enabled is false/absent
#      in the PINNED strategy config, exit 0 — skipped by design.
#   0.5 Corpus asserts (§2.2/§2.3): WF corpus coverage for the training
#      window + scorer-family parity against the pinned config. Fail closed.
#   1. Range-find training window: today − 60d (lookahead safety) − 12mo
#   2. Run snapshot-collection sim on that window (writes parquet) on the
#      WALK-FORWARD path (§2.2): per-bar point-in-time vintages from the
#      calibrator-bound v2 corpus manifest; never the legacy static load.
#   3. Apply triple-barrier labels (López de Prado AFML ch.3)
#   4. Train XGBoost with PurgedKFold CV (AFML ch.7)
#   5. Atomic swap artifact (backup old, deploy new)
#   6. Health checks: AUC ≥ 0.52, n_events ≥ 100, balance in [0.3, 0.7]
#   7. ntfy summary
set -uo pipefail

# RQ_META_LABEL_REPO_DIR: test-only sandbox override (same convention as
# RQ_WEEKLY_PROMOTE_REPO_DIR / RQ_MANUAL_PROMOTE_REPO_DIR); production
# launchd invokes with the env unset → live umbrella path.
REPO_DIR="${RQ_META_LABEL_REPO_DIR:-/Users/renhao/git/github/RenQuant}"
VENV_DIR="$REPO_DIR/.venv"
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
GITHUB_DIR="$(dirname "$REPO_DIR")"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh"
renquant_load_subrepo_env "$REPO_DIR"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR")"
export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"
export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-backtesting renquant-pipeline renquant-model renquant-common renquant-base-data renquant-artifacts):${PYTHONPATH:-}"
META_LABEL_STRICT=0
META_LABEL_SIM_STRICT=0
if renquant_strict_enabled RQ_META_LABEL_STRICT; then
    META_LABEL_STRICT=1
fi
if renquant_strict_enabled RQ_META_LABEL_SIM_STRICT; then
    META_LABEL_SIM_STRICT=1
fi
if ! PROD_STRATEGY_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json)"; then
    notify "META-LABEL RETRAIN ✗" "pinned renquant-strategy-104 strategy_config.json unavailable; monthly job fails closed"
    exit 1
fi

# ── Step 0: CONSUMER GATE (redesign §2.1) ───────────────────────────
# The only wired consumer (MetaLabelVetoTask) is dark:
# ranking.meta_label.enabled=false in the pinned config AND the artifact
# was retired on 2026-05-11. Retraining an artifact nothing reads is
# inert scaffolding — skip BY DESIGN, with zero training compute, zero
# artifact churn, and no alarm. Re-arming the consumer is a config
# change via its own design PR, not an ops change; this gate then
# passes without touching this script. Only a JSON boolean true arms the
# consumer; malformed values must fail closed rather than relying on
# Python truthiness (for example, the string "false" is truthy).
if ! CONSUMER_ENABLED="$("$PYTHON" - "$PROD_STRATEGY_CONFIG" <<'PY'
import json
import sys

cfg = json.load(open(sys.argv[1]))
enabled = cfg.get("ranking", {}).get("meta_label", {}).get("enabled", False)
if type(enabled) is not bool:
    raise ValueError("ranking.meta_label.enabled must be a JSON boolean")
print("true" if enabled is True else "false")
PY
)"; then
    notify "META-LABEL RETRAIN ✗" "pinned strategy config invalid or unreadable for consumer gate; monthly job fails closed"
    exit 1
fi
if [ "$CONSUMER_ENABLED" != "true" ]; then
    echo "meta-label consumer dark — retrain skipped by design (see doc/design/2026-07-18-metalabel-monthly-retrain-redesign.md)" | tee -a "$LOG"
    exit 0
fi

if "$PYTHON" - <<'PY' >/dev/null 2>&1
import renquant_backtesting.wf_gate.sim_driver  # noqa: F401
PY
then
    :
else
    notify "META-LABEL RETRAIN ✗" "renquant_backtesting.wf_gate.sim_driver unavailable; monthly job fails closed"
    exit 1
fi

if "$PYTHON" - <<'PY' >/dev/null 2>&1
import renquant_model_common.meta_label_exit  # noqa: F401
PY
then
    :
else
    notify "META-LABEL RETRAIN ✗" "renquant_model_common.meta_label_exit unavailable; monthly job fails closed"
    exit 1
fi

# ── Compute training window: [today − 60d − 365d, today − 60d] ──────
# 60d = lookahead_days safety buffer (fwd_60d_excess label horizon).
TRAIN_END=$(date -v-60d +%Y-%m-%d 2>/dev/null || date -d "today - 60 days" +%Y-%m-%d)
TRAIN_START=$(date -v-60d -v-365d +%Y-%m-%d 2>/dev/null || date -d "today - 60 days - 365 days" +%Y-%m-%d)
echo "[$(date '+%H:%M:%S')] Monthly meta-label retrain — training window $TRAIN_START → $TRAIN_END" | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] Multirepo fail-closed: enabled (strict_model=$META_LABEL_STRICT strict_sim=$META_LABEL_SIM_STRICT)" | tee -a "$LOG"

# ── Step 0.5: WF corpus coverage + scorer-family parity (§2.2/§2.3) ──
# The calibrator-bound v2 corpus manifest (39 point-in-time vintages;
# the plain walkforward_manifest.json twin has no calibrator bindings
# and hard-raises under global calibration at the first scored bar).
WF_MANIFEST="$REPO_DIR/backtesting/renquant_104/artifacts/sim/walkforward_manifest_v2_20260602.json"
if ! WF_ASSERT="$("$PYTHON" - "$WF_MANIFEST" "$PROD_STRATEGY_CONFIG" "$TRAIN_END" "$REPO_DIR/backtesting/renquant_104" <<'PY'
import datetime as dt
import json
import os
import sys

manifest_path, config_path, train_end_raw, strategy_root = sys.argv[1:5]


def fail(msg: str) -> None:
    print(msg)
    sys.exit(1)


try:
    manifest = json.load(open(manifest_path))
except (OSError, ValueError) as exc:
    fail(f"wf corpus manifest unreadable ({manifest_path}): {exc}")

rows = manifest.get("retrains") or []
if not rows:
    fail(f"wf corpus manifest has empty retrains[] ({manifest_path})")


def _date(value) -> dt.date:
    return dt.date.fromisoformat(str(value)[:10])


def feature_cutoff(row) -> dt.date:
    # Loader parity (WalkForwardModelLoader._feature_cutoff_date):
    # effective_train_cutoff_date when stamped, else cutoff_date.
    return _date(row.get("effective_train_cutoff_date") or row["cutoff_date"])


train_end = _date(train_end_raw)

# §2.3 math: eligibility is feature_cutoff + 60 business days < bar, so
# the newest vintage able to serve the LAST window bar must have
# feature_cutoff < TRAIN_END − 60bd. Allow 35 calendar days of EXTRA
# staleness beyond that structural embargo (one 21-day cadence step +
# margin); anything staler fails closed. NOTE for maintainers: this
# python block runs inside a shell command substitution — keep it free
# of apostrophes (bash 3.2 heredoc-in-substitution parsing).
threshold = train_end
remaining = 60
while remaining > 0:
    threshold -= dt.timedelta(days=1)
    if threshold.weekday() < 5:
        remaining -= 1
threshold -= dt.timedelta(days=35)

newest = max(feature_cutoff(row) for row in rows)
if newest < threshold:
    fail(
        f"wf corpus stale for window (newest cutoff {newest.isoformat()}; "
        f"need >= {threshold.isoformat()} for TRAIN_END {train_end.isoformat()})"
    )

# §2.2 scorer-family parity: the v2 manifest rows carry no kind field —
# read the vintage artifact metadata kind per row and map it to the
# pinned scorer family via an explicit allowlist. This is also the
# check that surfaces training/serving family skew (e.g. a pinned
# hf_patchtst config against the panel_ltr_xgboost corpus).
FAMILY_BY_ARTIFACT_KIND = {"panel_ltr_xgboost": "xgb"}

try:
    pinned_cfg = json.load(open(config_path))
except (OSError, ValueError) as exc:
    fail(f"pinned strategy config unreadable for scorer-family parity: {exc}")
pinned_family = (pinned_cfg.get("ranking", {}).get("panel_scoring") or {}).get("kind")
if not pinned_family:
    fail(
        "wf corpus scorer-family parity unresolvable: pinned config missing "
        "ranking.panel_scoring.kind"
    )

for row in rows:
    artifact_uri = row.get("artifact_uri") or ""
    artifact_path = os.path.join(strategy_root, artifact_uri)
    vintage = row.get("cutoff_date")
    try:
        kind = json.load(open(artifact_path)).get("kind")
    except (OSError, ValueError) as exc:
        fail(f"wf corpus vintage artifact unreadable ({artifact_uri}): {exc}")
    family = FAMILY_BY_ARTIFACT_KIND.get(kind)
    if family is None:
        fail(
            f"wf corpus scorer-family unmapped (vintage {vintage} "
            f"kind={kind!r} not in allowlist {sorted(FAMILY_BY_ARTIFACT_KIND)})"
        )
    if family != pinned_family:
        fail(
            f"wf corpus scorer-family mismatch (vintage {vintage} "
            f"kind={kind!r} -> family={family!r}; pinned family={pinned_family!r})"
        )

print(
    f"newest_cutoff={newest.isoformat()} threshold={threshold.isoformat()} "
    f"family={pinned_family} vintages={len(rows)}"
)
PY
)"; then
    echo "[$(date '+%H:%M:%S')] WF corpus assert FAIL: $WF_ASSERT" | tee -a "$LOG"
    notify "META-LABEL RETRAIN ✗" "$WF_ASSERT"
    exit 1
fi
echo "[$(date '+%H:%M:%S')] WF corpus asserts OK: $WF_ASSERT" | tee -a "$LOG"

# ── Step 2: snapshot sim on prior 12 months ─────────────────────────
SNAP_CFG="strategy_config.sim_monthly_retrain_snapshot.json"
SNAP_OUT="data/monthly_meta_label_snapshots_${DATE}.parquet"
LABEL_OUT="data/monthly_meta_label_labels_${DATE}.parquet"
NEW_ARTIFACT="$ART_DIR/meta-label-exit.candidate-${DATE}.json"
PROD_ARTIFACT="$ART_DIR/meta-label-exit.json"

$PYTHON <<PY 2>&1 | tee -a "$LOG"
import json
src = json.load(open("$PROD_STRATEGY_CONFIG"))
src["_side_config_label"] = "sim_monthly_retrain_snapshot"
src["_source_strategy_config"] = "$PROD_STRATEGY_CONFIG"
src["meta_label_training"] = {"enabled": True, "output_path": "$SNAP_OUT"}
src.setdefault("ranking", {})["meta_label"] = {"enabled": False}
# §2.2: leakage-correct walk-forward path. WHOLESALE replacement — the
# prod walkforward pointer is a dead reference (trap 2a) and must NEVER
# be inherited; always the explicit calibrator-bound v2 manifest.
# fail_on_no_model=true: a sim bar with no ELIGIBLE vintage is a hard
# failure, never a silent fallback to the legacy static load.
src["walkforward"] = {
    "enabled": True,
    "manifest_path": "$WF_MANIFEST",
    "fail_on_no_model": True,
}
json.dump(src, open("backtesting/renquant_104/$SNAP_CFG", "w"), indent=2)
print(f"Built snapshot config: $SNAP_CFG (walkforward manifest: $WF_MANIFEST)")
PY

echo "[$(date '+%H:%M:%S')] Step 2: snapshot sim …" | tee -a "$LOG"
if ! $PYTHON -m renquant_backtesting.wf_gate.sim_driver \
    --repo-root "$REPO_DIR" \
    --start "$TRAIN_START" --end "$TRAIN_END" \
    --strategy-config-name "$SNAP_CFG" \
    --no-persist --no-compare >> "$LOG" 2>&1
then
    notify "META-LABEL RETRAIN ✗" "snapshot sim failed — check $LOG"
    exit 1
fi

if [ ! -f "$SNAP_OUT" ]; then
    notify "META-LABEL RETRAIN ✗" "snapshot parquet missing — check $LOG"
    exit 1
fi

# ── Step 3: triple-barrier labels ───────────────────────────────────
echo "[$(date '+%H:%M:%S')] Step 3: label …" | tee -a "$LOG"
if ! $PYTHON -m renquant_model_common.meta_label_exit generate-labels \
    --snapshots "$REPO_DIR/$SNAP_OUT" \
    --out       "$REPO_DIR/$LABEL_OUT" \
    --data-dir  "$REPO_DIR/data" \
    --pt-mult 10 --sl-mult 10 --fwd-window 20 \
    --json >> "$LOG" 2>&1
then
    notify "META-LABEL RETRAIN ✗" "label generation failed — check $LOG"
    exit 1
fi

# ── Step 4: train classifier ────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Step 4: train …" | tee -a "$LOG"
if ! $PYTHON -m renquant_model_common.meta_label_exit train \
    --labels "$REPO_DIR/$LABEL_OUT" \
    --out    "$NEW_ARTIFACT" \
    --n-splits 5 --label-horizon-days 20 --pct-embargo 0.02 \
    --json >> "$LOG" 2>&1
then
    notify "META-LABEL RETRAIN ✗" "training failed — check $LOG"
    exit 1
fi

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
