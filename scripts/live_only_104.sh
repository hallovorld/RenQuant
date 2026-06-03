#!/usr/bin/env bash
# live_only_104.sh — compatibility entrypoint for the old open/preclose
# RenQuant 104 trigger. The active intraday sell-only workflow lives in
# intraday_sell_104.sh; keep this file only so disabled/stale launchd plists
# and manual muscle memory do not call live.runner directly.
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"

for arg in "$@"; do
    if [ "$arg" = "--no-sell-only" ]; then
        echo "ERROR: live_only_104.sh is sell-only compatibility glue; use scripts/daily_104.sh for a full buy/sell cycle."
        exit 2
    fi
done

exec bash "$REPO_DIR/scripts/intraday_sell_104.sh"
