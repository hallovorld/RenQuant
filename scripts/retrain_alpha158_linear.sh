#!/usr/bin/env bash
# retrain_alpha158_linear.sh — Refit the alpha158 + LinearRegression panel scorer
# on most-recent data. Output: artifacts/panel-ltr.alpha158_linear.json +
# panel-rank-calibration.alpha158_linear.json.
#
# Usage:
#   bash scripts/retrain_alpha158_linear.sh                  # full rebuild
#   bash scripts/retrain_alpha158_linear.sh --skip-features  # use existing
#                                                            # parquet
#
# Designed to run daily AFTER scripts/retrain_panel.sh (the production XGB
# sweep) so both panel-LTR variants stay fresh. Wired via launchd plist
# `com.renquant.retrain-alpha158-linear.plist` once production-ready.
#
# Per CLAUDE.md §5.5 — production-touching change. Verify rollback path
# before scheduling: a corrupted alpha158 artifact is automatically
# rejected by `LoadScorerTask`'s strict_config_consistency=True check
# (artifact must have kind="panel_linear" + matching feature_cols).

set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
VENV_DIR="$REPO_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$REPO_DIR/logs/retrain_alpha158_linear"
NTFY_TOPIC="renquant"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"

# ── Lock — prevent concurrent invocations ──────────────────────────────────
LOCK_FILE="/tmp/renquant_alpha158_linear_retrain.lock"
if ! (set -C; echo $$ > "$LOCK_FILE") 2>/dev/null; then
    EXISTING_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "?")
    echo "Another retrain_alpha158_linear run is active (PID=$EXISTING_PID) — skipping."
    exit 0
fi
trap "rm -f '$LOCK_FILE'" EXIT

# ── CPU saturation per CLAUDE.md §5.10 ─────────────────────────────────────
THREADS=$("$PYTHON" - <<'PY'
import os
print(os.cpu_count() or 1)
PY
)
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"

cd "$REPO_DIR"

exec >> "$LOG" 2>&1
echo "=== retrain_alpha158_linear started at $(date) ==="

SKIP_FEATURES=0
for arg in "$@"; do
    case "$arg" in
        --skip-features) SKIP_FEATURES=1 ;;
    esac
done

ARTIFACT="$REPO_DIR/backtesting/renquant_104/artifacts/panel-ltr.alpha158_linear.json"
PREV_ARTIFACT="$REPO_DIR/backtesting/renquant_104/artifacts/panel-ltr.alpha158_linear.previous.json"
CALIB="$REPO_DIR/backtesting/renquant_104/artifacts/panel-rank-calibration.alpha158_linear.json"

run_multirepo() {
    local args=(--repo-dir "$REPO_DIR" --scorer-out "$ARTIFACT" --calibrator-out "$CALIB")
    if [ "$SKIP_FEATURES" -eq 1 ]; then
        args+=(--skip-features)
    fi
    "$PYTHON" -m renquant_orchestrator.retrain_alpha158_linear "${args[@]}"
}

export RENQUANT_REPO_ROOT="$REPO_DIR"
GITHUB_DIR="$(dirname "$REPO_DIR")"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR")"
export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-orchestrator renquant-common renquant-base-data renquant-artifacts renquant-model renquant-pipeline renquant-execution renquant-strategy-104 renquant-backtesting):${PYTHONPATH:-}"

RUNNER="${RQ_ALPHA158_LINEAR_RUNNER:-multirepo}"
if [ "$RUNNER" = "multirepo" ]; then
    if "$PYTHON" - <<'PY' >/dev/null 2>&1
import renquant_orchestrator.retrain_alpha158_linear  # noqa: F401
PY
    then
        "$PYTHON" - <<'PY' >&2
import renquant_orchestrator.retrain_alpha158_linear as m
print(f"renquant_orchestrator.retrain_alpha158_linear={m.__file__}")
PY
        if [ -f "$ARTIFACT" ]; then
            cp "$ARTIFACT" "$PREV_ARTIFACT"
            echo "Backed up current artifact -> $(basename "$PREV_ARTIFACT")"
        fi
        run_multirepo "$@"
        RUN_RC=$?
        if [ "$RUN_RC" -eq 0 ]; then
            echo "=== retrain_alpha158_linear COMPLETE at $(date) ==="
            ARTIFACT_AGE=$(date -r "$ARTIFACT" "+%Y-%m-%d %H:%M:%S")
            curl -s -H "Title: RENQUANT-104 alpha158 retrain OK" \
                 -d "Artifact refreshed: $ARTIFACT_AGE" \
                 "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
            exit 0
        fi
        echo "FATAL: renquant_orchestrator.retrain_alpha158_linear failed (rc=$RUN_RC)"
        if [ -f "$PREV_ARTIFACT" ]; then
            cp "$PREV_ARTIFACT" "$ARTIFACT"
            echo "Rolled back to previous artifact"
        fi
        curl -s -H "Title: RENQUANT-104 retrain FAIL" -d "alpha158_linear multirepo retrain failed (rc=$RUN_RC)" \
             "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
        exit "$RUN_RC"
    elif [ "${RQ_ALPHA158_LINEAR_STRICT:-0}" = "1" ]; then
        echo "ERROR: renquant_orchestrator.retrain_alpha158_linear unavailable and RQ_ALPHA158_LINEAR_STRICT=1"
        exit 1
    else
        echo "WARN: renquant_orchestrator.retrain_alpha158_linear unavailable; falling back to umbrella retrain."
    fi
elif [ "$RUNNER" != "umbrella" ]; then
    echo "ERROR: unknown RQ_ALPHA158_LINEAR_RUNNER=$RUNNER (expected multirepo or umbrella)"
    exit 2
fi

# ── Phase 1: rebuild alpha158 dataset (unless skip) ────────────────────────
if [ "$SKIP_FEATURES" -eq 0 ]; then
    echo "--- Phase 1/3: rebuild alpha158 dataset ---"
    "$PYTHON" scripts/build_alpha158_qlib.py
    if [ $? -ne 0 ]; then
        echo "FATAL: alpha158 dataset build failed"
        curl -s -H "Title: RENQUANT-104 retrain FAIL" -d "alpha158 dataset build failed" \
             "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
        exit 1
    fi
else
    echo "--- Phase 1/3: SKIPPED (--skip-features) ---"
fi

# ── Phase 2: refit alpha158_linear scorer ──────────────────────────────────
echo "--- Phase 2/3: refit PanelLinearScorer (sklearn LinearRegression on z-scored fwd_5d_excess) ---"

# Backup current artifact for rollback
if [ -f "$ARTIFACT" ]; then
    cp "$ARTIFACT" "$PREV_ARTIFACT"
    echo "Backed up current artifact -> $(basename "$PREV_ARTIFACT")"
fi

"$PYTHON" scripts/train_panel_linear.py \
    --label fwd_5d_excess \
    --estimator ols \
    --output "$ARTIFACT"
TRAIN_RC=$?

if [ "$TRAIN_RC" -ne 0 ]; then
    echo "FATAL: train_panel_linear.py failed (rc=$TRAIN_RC)"
    if [ -f "$PREV_ARTIFACT" ]; then
        cp "$PREV_ARTIFACT" "$ARTIFACT"
        echo "Rolled back to previous artifact"
    fi
    curl -s -H "Title: RENQUANT-104 retrain FAIL" -d "alpha158_linear refit failed (rc=$TRAIN_RC)" \
         "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
    exit "$TRAIN_RC"
fi

# ── Phase 3: refit calibrator ──────────────────────────────────────────────
echo "--- Phase 3/3: refit alpha158_linear calibrator ---"
"$PYTHON" scripts/fit_alpha158_linear_calibrator.py --out "$CALIB"
CALIB_RC=$?

if [ "$CALIB_RC" -ne 0 ]; then
    echo "WARN: calibrator refit failed (rc=$CALIB_RC); existing calibrator will be used"
    curl -s -H "Title: RENQUANT-104 retrain WARN" -d "alpha158_linear calibrator refit failed; using stale" \
         "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
fi

echo "=== retrain_alpha158_linear COMPLETE at $(date) ==="
ARTIFACT_AGE=$(date -r "$ARTIFACT" "+%Y-%m-%d %H:%M:%S")
curl -s -H "Title: RENQUANT-104 alpha158 retrain OK" \
     -d "Artifact refreshed: $ARTIFACT_AGE" \
     "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
