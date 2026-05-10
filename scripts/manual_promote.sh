#!/usr/bin/env bash
# manual_promote.sh — Emergency operator promote (with explicit confirmation).
#
# 2026-05-09 audit FIX-C: only path that can promote without going through
# weekly_wf_promote.sh's WF gate. Required for emergencies (e.g. urgent
# bug fix needs to ship same-day, weekly cron is 6 days away). Uses
# RQ_ALLOW_NO_WF=1 BUT requires three explicit y/N confirmations:
#   1. Confirm artifact path
#   2. Confirm reason (must be one of: emergency_bugfix /
#      regulatory / hotfix / disaster_recovery)
#   3. Confirm rollback rehearsed (per CLAUDE.md §5.5)
#
# After every emergency promote, the operator MUST also run
# weekly_wf_promote.sh within 24h to validate the rushed model passes
# the proper trust boundary.
#
# Usage::
#
#     bash scripts/manual_promote.sh
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
PYTHON="/Users/renhao/miniconda3/envs/renquant/bin/python"

echo "=== EMERGENCY MANUAL PROMOTE ==="
echo "This bypasses the weekly walk-forward gate. Use ONLY for emergencies."
echo

# 1. Confirm artifact path
DEFAULT_STAGING="$REPO_DIR/backtesting/renquant_104/artifacts/panel-ltr.staging.json"
read -p "Staging artifact path [$DEFAULT_STAGING]: " STAGING
STAGING="${STAGING:-$DEFAULT_STAGING}"
if [ ! -f "$STAGING" ]; then
    echo "ERROR: staging file not found: $STAGING"
    exit 1
fi
echo "  → using $STAGING"
echo

# 2. Confirm reason
echo "Allowed reasons: emergency_bugfix / regulatory / hotfix / disaster_recovery"
read -p "Reason: " REASON
case "$REASON" in
    emergency_bugfix|regulatory|hotfix|disaster_recovery) ;;
    *)
        echo "ERROR: reason '$REASON' not allowed — see comment block in script"
        exit 1
        ;;
esac
echo

# 3. Confirm rollback rehearsed (CLAUDE.md §5.5)
read -p "Have you rehearsed the rollback path? (y/N) " ROLLBACK
if [ "$ROLLBACK" != "y" ] && [ "$ROLLBACK" != "Y" ]; then
    echo "ERROR: rollback rehearsal MANDATORY per CLAUDE.md §5.5. Aborted."
    exit 1
fi
echo

echo "Proceeding with emergency promote ($REASON) at $(date)"
echo "Per CLAUDE.md §5.5 you MUST run weekly_wf_promote.sh within 24h."

cd "$REPO_DIR"
RQ_ALLOW_NO_WF=1 "$PYTHON" -c "
from pathlib import Path
import sys
sys.path.insert(0, 'backtesting/renquant_104')
from kernel.model_acceptance import promote
staging = Path('$STAGING')
active = staging.parent / 'panel-ltr.alpha158_fund.json'
print(f'Promoting {staging} → {active}  reason=$REASON')
promote(staging, active)
print('Promote complete.')
"
echo "=== EMERGENCY PROMOTE complete at $(date) ==="
echo "REMINDER: run scripts/weekly_wf_promote.sh within 24h to validate."

# Refresh dashboard so live status reflects new model
"$PYTHON" "$REPO_DIR/scripts/build_dashboard.py" --broker alpaca \
    --out "$REPO_DIR/doc/dashboard.md" 2>&1 | tail -3 \
    || echo "dashboard refresh failed (non-fatal)"
