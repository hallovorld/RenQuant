#!/usr/bin/env bash
# daily_earnings_surprise_refresh.sh — incremental PEAD/SUE refresh after prints.
#
# 2026-08-30 fix (data audit): data/earnings_surprise/*.parquet was
# refreshed ONLY by weekly-fundamental-refresh (Sat 04:00), so after a
# print (NVDA/CRWD 08-26, PANW Aug) the PEAD/SUE features stayed
# median-imputed for up to a week while the panel already saw the price
# gap — and the 5-day per-ticker ER model flips on the gap day. This job
# runs every session morning (Mon-Fri 06:00 PT, before the 06:05 dawn
# preflight) and refreshes ONLY the tickers with a print in the last 7
# days, so PEAD/SUE are populated for the NEXT session after any print.
#
# INCREMENTAL selection: scripts/earnings_calendar_rail.py select-recent
# reads the (now daily-refreshed) prod earnings calendar. The underlying
# fetch is cache-gated (refresh_after_days=30): a just-printed ticker's
# cache tops out at the PREVIOUS quarter (~90d old) → refetched; once
# the new print is cached, subsequent runs short-circuit. The weekly
# FULL refresh (weekly_fundamental_refresh.sh step 3) stays as-is.
#
# FAIL BEHAVIOUR:
#   * calendar stale/missing → ntfy ⚠ and FALL BACK to the full
#     watchlist (cache-gated, so still cheap) — the surprise refresh
#     must not silently skip because its selector's input rotted;
#   * pinned config unavailable / module unavailable / fetch crash →
#     ntfy ✗, exit non-zero (same contract as weekly_fundamental_refresh).
#
# Vendor: yfinance .earnings_dates (free, no API key).
# Plist: scripts/launchd/com.renquant.daily-earnings-surprise.plist
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
VENV_DIR="/Users/renhao/git/github/RenQuant/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$REPO_DIR/logs/daily_earnings_surprise"
NTFY_TOPIC="renquant"
LOOKBACK_DAYS=7
MIN_HORIZON_DAYS=5
CAL_PATH="$REPO_DIR/backtesting/renquant_104/artifacts/prod/earnings-calendar.json"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"

notify() {
    local title="$1" body="$2"
    if [ "${RENQUANT_NO_NOTIFY:-0}" = "1" ]; then return 0; fi
    curl -s -H "Title: $title" -d "$body" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
}

fail() {
    local msg="$1"
    echo "[$(date '+%H:%M:%S')] FAIL: $msg" | tee -a "$LOG"
    notify "EARNINGS SURPRISE ✗" "$DATE  $msg"
    exit 1
}

cd "$REPO_DIR" || fail "cannot cd to $REPO_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
GITHUB_DIR="$(dirname "$REPO_DIR")"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh"
renquant_load_subrepo_env "$REPO_DIR"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR")"
export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"
export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-base-data renquant-common):${PYTHONPATH:-}"

echo "[$(date '+%H:%M:%S')] Daily earnings-surprise refresh — $DATE" | tee -a "$LOG"

if ! STRATEGY_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json)"; then
    fail "pinned renquant-strategy-104 strategy_config.json unavailable"
fi

if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import renquant_base_data.earnings_surprise_refresh  # noqa: F401
PY
then
    fail "renquant_base_data.earnings_surprise_refresh unavailable on PYTHONPATH"
fi

# ── Incremental selection: tickers with a print in the last N days ──────────
SYMBOL_ARGS=()
"$PYTHON" "$REPO_DIR/scripts/earnings_calendar_rail.py" check \
    --calendar "$CAL_PATH" --min-horizon-days "$MIN_HORIZON_DAYS" >> "$LOG" 2>&1
RAIL_RC=$?
if [ "$RAIL_RC" -eq 0 ]; then
    RECENT=$("$PYTHON" "$REPO_DIR/scripts/earnings_calendar_rail.py" select-recent \
        --calendar "$CAL_PATH" --lookback-days "$LOOKBACK_DAYS" 2>>"$LOG")
    if [ -z "$RECENT" ]; then
        echo "[$(date '+%H:%M:%S')] no prints in the last ${LOOKBACK_DAYS}d — nothing to refresh ✓" | tee -a "$LOG"
        exit 0
    fi
    # shellcheck disable=SC2206
    SYMBOL_ARGS=(--symbols $RECENT)
    echo "recent prints (${LOOKBACK_DAYS}d): $RECENT" | tr '\n' ' ' | tee -a "$LOG"; echo | tee -a "$LOG"
else
    # The selector's input rotted — refresh must not silently skip.
    echo "[$(date '+%H:%M:%S')] calendar rail rc=$RAIL_RC — falling back to FULL watchlist refresh" | tee -a "$LOG"
    notify "EARNINGS SURPRISE ⚠" "$DATE  earnings calendar stale/missing (rail rc=$RAIL_RC) — fell back to full-watchlist refresh; check com.renquant.earnings-calendar-refresh"
fi

"$PYTHON" -m renquant_base_data.earnings_surprise_refresh \
    --strategy-config "$STRATEGY_CONFIG" \
    --data-dir "$REPO_DIR/data" \
    --total-budget-sec 600 \
    "${SYMBOL_ARGS[@]}" \
    --json >> "$LOG" 2>&1
RC=$?
if [ "$RC" -ne 0 ]; then
    fail "earnings_surprise_refresh exited rc=$RC — see $LOG"
fi

echo "[$(date '+%H:%M:%S')] DONE ✓" | tee -a "$LOG"
tail -3 "$LOG"
