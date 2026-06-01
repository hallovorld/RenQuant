#!/usr/bin/env bash
# event_sec_schema_change.sh — Regen SEC fundamentals + retrain.
#
# 2026-05-09 audit FIX-C: explicit event for the BUG #5 / data-staleness
# class. When scripts/fetch_sec_fundamentals.py is modified (e.g.
# 4d → 252d periods change), the parquet on disk is still the OLD
# version. The model trains on stale features until the parquet is
# regenerated and a new retrain runs.
#
# This script:
#   1. Re-runs the SEC fundamentals refresh pipeline to regenerate parquet
#      outputs (~60 min, hits SEC EDGAR; rate-limited, be patient)
#   2. Triggers a full weekly_wf_promote run on the fresh data
#
# Usage::
#
#     bash scripts/event_sec_schema_change.sh
#     bash scripts/event_sec_schema_change.sh --skip-fetch    # use existing parquet
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
GITHUB_DIR="$(cd "$REPO_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh"
renquant_load_subrepo_env "$REPO_DIR"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR")"
VENV_DIR="$REPO_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
SKIP_FETCH=0
for arg in "$@"; do
    case "$arg" in
        --skip-fetch) SKIP_FETCH=1 ;;
    esac
done

run_sec_refresh() {
    if [ "${RQ_EVENT_SEC_REFRESH_RUNNER:-multirepo}" = "legacy" ]; then
        "$PYTHON" scripts/fetch_sec_fundamentals.py
        return $?
    fi

    local base_data_src
    base_data_src="$(renquant_subrepo_src "$SUBREPO_ROOT" renquant-base-data)"
    if PYTHONPATH="$base_data_src:${PYTHONPATH:-}" "$PYTHON" - <<'PY'
import renquant_base_data.sec_fundamentals  # noqa: F401
PY
    then
        PYTHONPATH="$base_data_src:${PYTHONPATH:-}" "$PYTHON" -m renquant_base_data.sec_fundamentals \
            --data-dir "$REPO_DIR/data" \
            --mode both
        return $?
    fi

    if [ "${RQ_EVENT_SEC_REFRESH_STRICT:-0}" = "1" ]; then
        echo "ERROR: renquant_base_data.sec_fundamentals unavailable and RQ_EVENT_SEC_REFRESH_STRICT=1"
        return 2
    fi

    echo "WARN: renquant_base_data.sec_fundamentals unavailable; falling back to umbrella SEC fetch."
    "$PYTHON" scripts/fetch_sec_fundamentals.py
}

echo "=== event_sec_schema_change started at $(date) ==="
if [ "$SKIP_FETCH" -eq 0 ]; then
    echo "Step 1: Regenerate SEC fundamentals parquet outputs"
    echo "  This hits SEC EDGAR API (rate-limited, ~60 min)."
    read -p "  Continue with fetch? (y/N) " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Aborted."
        exit 1
    fi
    cd "$REPO_DIR"
    if ! run_sec_refresh; then
        echo "SEC fundamentals refresh FAILED — preserving prior parquet."
        exit 1
    fi
    echo "SEC parquet outputs regenerated at $(date)."
else
    echo "Step 1: --skip-fetch given, using existing parquet."
fi

echo "Step 2: Run weekly_wf_promote with fresh fundamentals data."
bash "$REPO_DIR/scripts/weekly_wf_promote.sh"
