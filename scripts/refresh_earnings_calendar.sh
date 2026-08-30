#!/usr/bin/env bash
# refresh_earnings_calendar.sh — scheduled earnings-calendar producer.
#
# 2026-08-30 fix (data audit): scripts/fetch_earnings_calendar.py had NEVER
# been scheduled — the prod artifact froze at its last manual run
# (2026-04-24 mtime, last date 2026-07-24), and since the 2026-05-10
# sim/prod isolation refactor (238359b) the script also wrote to
# artifacts/ while every consumer reads artifacts/prod/. Net effect: the
# pre/post-earnings buffer (±3d buy, -2d..+5d sell) silently could not
# fire for any Aug/Sep print (HPE bought 2026-08-27 into an early-Sep
# print). This wrapper + com.renquant.earnings-calendar-refresh make the
# producer a scheduled job:
#
#   * Mon-Fri 05:40 PT — every session morning, so the day's decisions
#     see at least the next 14 days of prints (we fetch 90d lookahead);
#     finishes before the 06:00 earnings-surprise refresh and the 06:05
#     dawn preflight.
#   * Sat 04:40 PT — weekly anchor, clear of weekly-fundamental-refresh
#     (04:00) and daily-analyst-ratings (04:20).
#
# Vendor: yfinance (free, no API key; no FMP/Finnhub quota). ~142
# tickers x 2 endpoints with a 0.3 s throttle ≈ a few minutes.
#
# FAIL LOUD: a fetch crash, a missing pinned config, or a
# stale-after-fetch calendar (last date < today+5d, exit 2 from the
# fetcher) each ntfy ✗ and exit non-zero. The CONSUMER side stays fail
# SOFT: daily_104.sh Step 0c + adapters/runner_artifacts.py alarm on a
# stale calendar but never block the run.
#
# Plist: scripts/launchd/com.renquant.earnings-calendar-refresh.plist
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
VENV_DIR="/Users/renhao/git/github/RenQuant/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$REPO_DIR/logs/earnings_calendar_refresh"
NTFY_TOPIC="renquant"
MIN_HORIZON_DAYS=5
LOOKAHEAD_DAYS=90
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
    notify "EARNINGS CALENDAR ✗" "$DATE  $msg"
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

echo "[$(date '+%H:%M:%S')] Earnings-calendar refresh — $DATE" | tee -a "$LOG"

# Watchlist source = the PINNED strategy config (the one the live run
# trades). Fail closed like weekly_fundamental_refresh: the umbrella
# copy is NOT an equivalent config (RenQuant#546).
if ! STRATEGY_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json)"; then
    fail "pinned renquant-strategy-104 strategy_config.json unavailable — refusing to fetch for a substitute watchlist"
fi
echo "watchlist config: $STRATEGY_CONFIG" | tee -a "$LOG"

"$PYTHON" "$REPO_DIR/scripts/fetch_earnings_calendar.py" \
    --strategy renquant_104 \
    --config "$STRATEGY_CONFIG" \
    --lookahead "$LOOKAHEAD_DAYS" \
    --min-horizon-days "$MIN_HORIZON_DAYS" >> "$LOG" 2>&1
RC=$?

CAL_PATH="$REPO_DIR/backtesting/renquant_104/artifacts/prod/earnings-calendar.json"
if [ "$RC" -eq 2 ]; then
    fail "stale-after-fetch: calendar last date still < today+${MIN_HORIZON_DAYS}d after a full fetch (vendor returned no usable forward dates) — see $LOG"
elif [ "$RC" -ne 0 ]; then
    fail "fetch_earnings_calendar.py exited rc=$RC — see $LOG"
fi

# Belt-and-braces: verify the CONSUMED artifact now passes the rail.
if ! "$PYTHON" "$REPO_DIR/scripts/earnings_calendar_rail.py" check \
        --calendar "$CAL_PATH" --min-horizon-days "$MIN_HORIZON_DAYS" >> "$LOG" 2>&1; then
    fail "post-write rail check failed on $CAL_PATH — see $LOG"
fi

echo "[$(date '+%H:%M:%S')] DONE ✓" | tee -a "$LOG"
tail -3 "$LOG"
