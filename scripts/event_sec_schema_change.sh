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
#   1. Re-runs fetch_sec_fundamentals.py to regenerate the parquet
#      (~60 min, hits SEC EDGAR — RATE-LIMITED, BE PATIENT)
#   2. Triggers a full weekly_wf_promote run on the fresh data
#
# Usage::
#
#     bash scripts/event_sec_schema_change.sh
#     bash scripts/event_sec_schema_change.sh --skip-fetch    # use existing parquet
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
SKIP_FETCH=0
for arg in "$@"; do
    case "$arg" in
        --skip-fetch) SKIP_FETCH=1 ;;
    esac
done

echo "=== event_sec_schema_change started at $(date) ==="
if [ "$SKIP_FETCH" -eq 0 ]; then
    echo "Step 1: Regenerate sec_fundamentals_daily.parquet"
    echo "  This hits SEC EDGAR API (rate-limited, ~60 min)."
    read -p "  Continue with fetch? (y/N) " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Aborted."
        exit 1
    fi
    cd "$REPO_DIR"
    if ! "/Users/renhao/miniconda3/envs/renquant/bin/python" \
            scripts/fetch_sec_fundamentals.py; then
        echo "fetch_sec_fundamentals.py FAILED — preserving prior parquet."
        exit 1
    fi
    echo "Parquet regenerated at $(date)."
else
    echo "Step 1: --skip-fetch given, using existing parquet."
fi

echo "Step 2: Run weekly_wf_promote with fresh fundamentals data."
bash "$REPO_DIR/scripts/weekly_wf_promote.sh"
