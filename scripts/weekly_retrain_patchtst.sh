#!/usr/bin/env bash
# weekly_retrain_patchtst.sh — scheduled PatchTST walk-forward retrain.
#
# Multi-repo + pipeline mode: bash only handles lock, log redirect, root + the
# subrepo PYTHONPATH, then DELEGATES the actual work to the orchestrator-owned
# pipeline `renquant_orchestrator.build_patchtst_wf_manifest` (a Task/Job/Pipeline
# that runs per-cutoff hf_trainer + calibrator subprocesses and emits the
# PatchTST WF manifest). No training logic lives in this wrapper.
#
# Mirrors scripts/daily_retrain_alpha158_fund.sh's delegation. PatchTST is one of
# the two production models (GBDT alpha158_fund is the other); both retrain on
# schedule and feed the daily signal.
#
# Usage: bash scripts/weekly_retrain_patchtst.sh [pipeline args]
set -euo pipefail

REPO_DIR="${RENQUANT_REPO_ROOT:-/Users/renhao/git/github/RenQuant}"
PYTHON="$REPO_DIR/.venv/bin/python"
LOG_DIR="$REPO_DIR/logs/weekly_retrain_patchtst"
mkdir -p "$LOG_DIR"
DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"
exec > >(tee -a "$LOG") 2>&1
echo "═══ weekly_retrain_patchtst started $(date -u +'%Y-%m-%dT%H:%M:%SZ') ═══"

LOCK_FILE="${RQ_PATCHTST_LOCK_FILE:-/tmp/renquant_retrain_patchtst.lock}"
if ! (set -C; echo $$ > "$LOCK_FILE") 2>/dev/null; then
    EXISTING_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "?")
    if ! kill -0 "$EXISTING_PID" 2>/dev/null; then
        rm -f "$LOCK_FILE"
        echo $$ > "$LOCK_FILE"
    else
        echo "Another weekly_retrain_patchtst is active (PID=$EXISTING_PID) — exiting."
        exit 0
    fi
fi
trap 'rm -f "$LOCK_FILE"' EXIT

# torch on macOS can segfault on MPS under headless launchd + at high OMP on .pt
# LOAD (not training). Default CPU/OMP=1 for the launchd schedule (safe); override
# RQ_PATCHTST_DEVICE=mps + RQ_PATCHTST_OMP for fast interactive/manual runs.
export OMP_NUM_THREADS="${RQ_PATCHTST_OMP:-1}" MKL_NUM_THREADS="${RQ_PATCHTST_OMP:-1}" OPENBLAS_NUM_THREADS="${RQ_PATCHTST_OMP:-1}"

# Multi-repo delegation: correct root + subrepo PYTHONPATH (incl orchestrator).
export RENQUANT_REPO_ROOT="$REPO_DIR"
GITHUB_DIR="$(dirname "$REPO_DIR")"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh"
renquant_load_subrepo_env "$REPO_DIR"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR")"
export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"
export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-orchestrator renquant-common renquant-base-data renquant-artifacts renquant-model renquant-pipeline renquant-execution renquant-strategy-104 renquant-backtesting):${PYTHONPATH:-}"

if ! "$PYTHON" -c "import renquant_orchestrator.build_patchtst_wf_manifest" >/dev/null 2>&1; then
    echo "ERROR: renquant_orchestrator.build_patchtst_wf_manifest unavailable (subrepo pin/env)."
    exit 1
fi

SRC_MANIFEST="${RQ_PATCHTST_SOURCE_MANIFEST:-$REPO_DIR/backtesting/renquant_104/artifacts/sim/walkforward_manifest_v2_20260602.json}"
OUT_DIR="$REPO_DIR/backtesting/renquant_104/artifacts/walkforward_patchtst"
OUT_MANIFEST="$REPO_DIR/backtesting/renquant_104/artifacts/walkforward_patchtst_manifest.json"

# Two modes (mirrors the GBDT side: 1 model + gate weekly; WF manifest built once):
#   WEEKLY (default): train ONLY the latest cutoff — 1 model, ~15min on MPS.
#   FULL (RQ_PATCHTST_FULL_MANIFEST=1): one-time SPARSE validation manifest,
#     ~6 cutoffs at cadence 180 (~90min). NOT the dense 39-cut/12h build.
cd "$REPO_DIR"
if [ "${RQ_PATCHTST_FULL_MANIFEST:-0}" = "1" ]; then
    EFFECTIVE_SRC="$SRC_MANIFEST"
    CADENCE="${RQ_PATCHTST_CADENCE:-180}"
    echo "Mode: FULL validation manifest (sparse, cadence=${CADENCE}d)"
else
    LATEST_CUT="$("$PYTHON" -c "import json;r=json.load(open('$SRC_MANIFEST')).get('retrains',[]);print(sorted(x['cutoff_date'] for x in r if x.get('cutoff_date'))[-1].split('T')[0])")"
    EFFECTIVE_SRC="$(mktemp "${TMPDIR:-/tmp}/patchtst_src.XXXXXX")"
    "$PYTHON" -c "import json;json.dump({'retrains':[{'cutoff_date':'$LATEST_CUT'}]},open('$EFFECTIVE_SRC','w'))"
    CADENCE=0
    echo "Mode: WEEKLY — train latest cutoff only ($LATEST_CUT)"
    trap 'rm -f "$LOCK_FILE" "$EFFECTIVE_SRC"' EXIT
fi
"$PYTHON" -m renquant_orchestrator.build_patchtst_wf_manifest \
    --source-manifest "$EFFECTIVE_SRC" \
    --output-dir "$OUT_DIR" \
    --output-manifest "$OUT_MANIFEST" \
    --cadence-days "$CADENCE" \
    --seed "${RQ_PATCHTST_SEED:-44}" \
    --epochs "${RQ_PATCHTST_EPOCHS:-5}" \
    --device "${RQ_PATCHTST_DEVICE:-cpu}" \
    "$@"
rc=$?
echo "WF corpus build rc=$rc"

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATED SERVED-PIN PROMOTE (design 2026-06-30-shadow-scorer-freshness.md §3.1/§3.4)
#
# A successful WF build (above) writes ONLY the walk-forward corpus — it does NOT
# advance the SERVED shadow pin (RFC §1.3: "even a retrain does NOT unfreeze the
# served model"). Without this step the model ages in place while the retrain
# "succeeds" — the repo's recurring "merged is not deployed" failure. So, on a
# clean build, chain the validated promote: it FAILS CLOSED (keeps the old pin)
# unless every recipe source is on-SLA AND the effective cutoffs advance AND the
# candidate passes the §3.4 load/parity/non-degenerate/resource/sanity gates.
#
# NON-FATAL: the promote never fails this job. The WF corpus is already built; a
# not-fresh refusal (exit 10) is EXPECTED on a stale panel, and a gate failure
# (exit 20) means the safe state (old pin retained) was kept. Disable with
# RQ_PATCHTST_PROMOTE=0; dry-run with RQ_PATCHTST_PROMOTE=dry.
# ─────────────────────────────────────────────────────────────────────────────
PROMOTE_MODE="${RQ_PATCHTST_PROMOTE:-1}"
if [ "$rc" -eq 0 ] && [ "$PROMOTE_MODE" != "0" ]; then
    # Resolve the config carrying the served hf_patchtst pin. Default: the pinned
    # renquant-strategy-104 shadow config; the promote refuses harmlessly (exit 2)
    # if that config is not kind=hf_patchtst, so override RQ_PATCHTST_SERVED_CONFIG
    # to the authoritative config for this deploy.
    PROMOTE_CONFIG="${RQ_PATCHTST_SERVED_CONFIG:-}"
    if [ -z "$PROMOTE_CONFIG" ]; then
        PROMOTE_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.shadow.json 2>/dev/null)" \
            || PROMOTE_CONFIG="$REPO_DIR/backtesting/renquant_104/strategy_config.shadow.json"
    fi
    PROMOTE_APPLY="--apply"
    [ "$PROMOTE_MODE" = "dry" ] && PROMOTE_APPLY="--check"
    echo "─── validated served-pin promote ($PROMOTE_APPLY) against $PROMOTE_CONFIG ───"
    set +e
    "$PYTHON" "$REPO_DIR/scripts/promote_shadow_patchtst.py" \
        --served-config "$PROMOTE_CONFIG" \
        --wf-manifest "$OUT_MANIFEST" \
        $PROMOTE_APPLY
    prc=$?
    set -e
    case "$prc" in
        0)  echo "promote: OK (pin advanced or clean dry-run)";;
        10) echo "promote: refused — NOT FRESH (expected on a stale panel; old pin kept)";;
        20) echo "promote: refused — VALIDATION GATE FAILED (old pin kept; investigate)";;
        *)  echo "promote: precondition/usage rc=$prc (old pin kept)";;
    esac
else
    echo "promote: skipped (rc=$rc, RQ_PATCHTST_PROMOTE=$PROMOTE_MODE)"
fi

echo "═══ weekly_retrain_patchtst finished rc=$rc $(date -u +'%Y-%m-%dT%H:%M:%SZ') ═══"
exit "$rc"
