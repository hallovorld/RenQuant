#!/usr/bin/env bash
# event_watchlist_change.sh — Trigger after watchlist (universe) is changed.
#
# 2026-05-09 audit FIX-C: explicit event-triggered cadence. Watchlist
# changes (e.g. wl103 → wl162 quality-first selection) require a full
# panel rebuild + retrain because:
#   - new tickers' alpha158 features must be added to training panel
#   - dropped tickers' rows must be removed (no leakage on stale labels)
#   - cross-sectional features (rank, z-score) recompute with new universe
#
# DOES NOT auto-promote — runs through the WF gate so even watchlist
# changes can't ship a worse model.
#
# Usage::
#
#     bash scripts/event_watchlist_change.sh
#
# Pre-requisites:
#   - strategy_config.json watchlist already updated to new universe
#   - data/wl<N>_quality_first.json (or equivalent) committed if used
#
# Post-conditions:
#   - panel-ltr.alpha158_fund.json refit on new universe (if WF passes)
#   - calibrator refit (if WF passes)
#   - dashboard refreshed
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
echo "=== event_watchlist_change started at $(date) ==="
echo "This will:"
echo "  1. Smoke test current model"
echo "  2. Rebuild panel with new watchlist"
echo "  3. Retrain (~75 min)"
echo "  4. Run WF gate (~15 min)"
echo "  5. Promote if WF passes (else preserve prior)"
read -p "Continue? (y/N) " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted."
    exit 1
fi

# Delegate to weekly script — same operation, just user-triggered.
# (The weekly cron path runs the SAME 5 steps; an off-cycle watchlist
# change uses the same trust boundary.)
bash "$REPO_DIR/scripts/weekly_wf_promote.sh"
