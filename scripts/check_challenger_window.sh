#!/usr/bin/env bash
#
# Phase 4b daily check — has the active challenger's shadow window closed?
# If yes, fire scripts/finalize_challenger.py (which writes the report and
# pushes ntfy). Otherwise, exit silently.
#
# Wire into launchd via daily_104.sh or its own plist for early-evening run.
#
# Usage:
#   ./scripts/check_challenger_window.sh                 # uses RENQUANT_104 default
#   ./scripts/check_challenger_window.sh renquant_104    # explicit strategy
#
# Exit codes:
#   0 — no challenger active OR window not yet closed (no-op success)
#   2 — challenger active, window closed, finalize_challenger.py succeeded
#   3 — finalize_challenger.py failed (operator should investigate)
#
# Why bash + jq instead of Python:
# This script needs to run as a fast cron poll. Reading
# strategy_config.json + checking a date math is trivial; spinning up a
# Python interpreter for it is wasteful when launchd fires it every day.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STRATEGY="${1:-renquant_104}"
STRATEGY_DIR="${REPO_ROOT}/backtesting/${STRATEGY}"
CONFIG="${STRATEGY_DIR}/strategy_config.json"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[check-challenger] strategy config missing: ${CONFIG}" >&2
  exit 0   # not an error per se — strategy may not exist yet
fi

# Read challenger.enabled and shadow_period_days. Fall back to disabled
# when the keys aren't present (default state in Phase 4a).
ENABLED="$(jq -r '.acceptance.challenger.enabled // false' "${CONFIG}" 2>/dev/null || echo false)"
NAME="$(jq -r '.acceptance.challenger.name // ""' "${CONFIG}" 2>/dev/null || echo '')"
WINDOW_DAYS="$(jq -r '.acceptance.challenger.shadow_period_days // 0' "${CONFIG}" 2>/dev/null || echo 0)"

if [[ "${ENABLED}" != "true" ]]; then
  exit 0    # challenger off → no-op
fi
if [[ -z "${NAME}" ]]; then
  echo "[check-challenger] enabled=true but no name set; skipping" >&2
  exit 0
fi
if [[ "${WINDOW_DAYS}" -le 0 ]]; then
  echo "[check-challenger] shadow_period_days=${WINDOW_DAYS}; skipping" >&2
  exit 0
fi

# Resolve runs.db path (same logic as kernel.persistence._db_path)
DB_PATH_RAW="$(jq -r '.persistence.db_path // "data/runs.db"' "${CONFIG}")"
if [[ "${DB_PATH_RAW}" = /* ]]; then
  DB_PATH="${DB_PATH_RAW}"
else
  DB_PATH="${REPO_ROOT}/${DB_PATH_RAW}"
fi

if [[ ! -f "${DB_PATH}" ]]; then
  exit 0   # DB doesn't exist yet → challenger never ran
fi

# Find earliest decision_date for this challenger
EARLIEST="$(sqlite3 "${DB_PATH}" \
  "SELECT MIN(decision_date) FROM challenger_decisions WHERE challenger_name='${NAME}'" \
  2>/dev/null || true)"
if [[ -z "${EARLIEST}" || "${EARLIEST}" = "" ]]; then
  exit 0   # no decisions yet (challenger enabled but live runner hasn't logged)
fi

# Compute window-end = earliest + window_days
# Use Python for date math — bash date doesn't handle ISO + day-add cleanly on macOS
WINDOW_END="$(/Users/renhao/miniconda3/envs/renquant/bin/python -c "
import sys, datetime as dt
earliest = '${EARLIEST}'.split(' ')[0].split('T')[0]
y,m,d = earliest.split('-')
end = dt.date(int(y), int(m), int(d)) + dt.timedelta(days=${WINDOW_DAYS})
print(end.isoformat())
")"

TODAY="$(date -u +%Y-%m-%d)"
if [[ "${TODAY}" < "${WINDOW_END}" ]]; then
  echo "[check-challenger] window still open (${EARLIEST} → ${WINDOW_END}, today=${TODAY})"
  exit 0
fi

echo "[check-challenger] window CLOSED — running finalize_challenger.py"
if /Users/renhao/miniconda3/envs/renquant/bin/python "${REPO_ROOT}/scripts/finalize_challenger.py" \
  --strategy "${STRATEGY}" \
  --challenger-name "${NAME}" \
  --start-date "${EARLIEST%% *}" \
  --end-date "${TODAY}"; then
  exit 2   # success — operator now has report + ntfy
else
  echo "[check-challenger] finalize_challenger.py FAILED" >&2
  exit 3
fi
