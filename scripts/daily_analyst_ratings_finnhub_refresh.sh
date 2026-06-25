#!/usr/bin/env bash
# daily_analyst_ratings_finnhub_refresh.sh — daily FULL-coverage Finnhub pull.
#
# WHY FINNHUB + DAILY FULL (2026-06-25):
# Finnhub's free `/stock/recommendation` gives monthly analyst recommendation
# distributions with FULL US-stock coverage — unlike FMP free's ~30% plan-lock
# (HTTP 402). Verified 136/145 (the 9 missing are ETFs/indices = no analysts).
# Free tier is 60 calls/min, so the whole ~145-name watchlist fits in one daily
# pass (~2.5 min at ~1s throttle) — no rotation needed (MAX_PULL=0 = all). The
# free window is only ~4 months, so DAILY accumulation (dedup by (ticker,period))
# grows the multi-month series the 3-month revision signal needs.
#
# FAIL-CLOSED (silent-degradation lesson): a quota hit / bad key / schema break
# must ntfy ✗ and exit non-zero, NOT pass as "no coverage". The refresh CLI
# distinguishes with_data / no_coverage[ETF] / quota_error[429] / fetch_error and
# we gate on BOTH:
#   * --fail-on-error    → exit non-zero on ANY quota/fetch error;
#   * --min-coverage-pct → exit non-zero on a SYSTEMIC coverage collapse over the
#                          COVERABLE set (ETFs excluded — they have no analysts).
#
# INERT UNTIL DEPLOYED: calls `renquant_base_data.finnhub_analyst_ratings_refresh`
# (base-data #25). Until that merges + the base-data pin is bumped, the import
# fails — the wrapper reports ✗ "module unavailable" rather than crashing.
#
# Schedule: daily 04:25 PT (offline hours). Plist:
# scripts/launchd/com.renquant.daily-analyst-ratings-finnhub.plist
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
VENV_DIR="/Users/renhao/git/github/RenQuant/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$REPO_DIR/logs/daily_analyst_ratings_finnhub"
NTFY_TOPIC="renquant"
OUTPUT="$REPO_DIR/data/analyst_ratings_finnhub.parquet"
MAX_PULL=0             # 0 = whole watchlist daily (full coverage; 60/min fits ~145)
SLEEP_SEC=1.1          # throttle for the free 60/min cap
MIN_COVERAGE_PCT=90    # systemic-collapse floor over the COVERABLE (non-ETF) set
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"

notify() {
    local title="$1" body="$2"
    curl -s -H "Title: $title" -d "$body" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
}

fail() {
    local msg="$1"
    echo "[$(date '+%H:%M:%S')] FAIL: $msg" | tee -a "$LOG"
    notify "FINNHUB ANALYST ✗" "$DATE  $msg"
    exit 1
}

cd "$REPO_DIR" || fail "cannot cd to $REPO_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
# FINNHUB_API_KEY lives in .env (gitignored, never committed)
if [[ -f "$REPO_DIR/.env" ]]; then
    set -a; # shellcheck disable=SC1091
    source "$REPO_DIR/.env"; set +a
fi
GITHUB_DIR="$(dirname "$REPO_DIR")"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh"
renquant_load_subrepo_env "$REPO_DIR"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR")"
export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"
export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-base-data renquant-common):${PYTHONPATH:-}"
echo "[$(date '+%H:%M:%S')] Daily Finnhub analyst refresh — $DATE" | tee -a "$LOG"

[[ -n "${FINNHUB_API_KEY:-}" ]] || fail "FINNHUB_API_KEY not set (.env)"

if ! STRATEGY_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json)"; then
    fail "pinned renquant-strategy-104 strategy_config.json unavailable"
fi

# Inert-until-deployed guard: the base-data Finnhub module must be importable.
if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import renquant_base_data.finnhub_analyst_ratings_refresh  # noqa: F401
PY
then
    fail "renquant_base_data.finnhub_analyst_ratings_refresh unavailable (base-data #25 not merged / pin not bumped)"
fi

echo "[$(date '+%H:%M:%S')] Pulling full watchlist (throttle ${SLEEP_SEC}s, floor ${MIN_COVERAGE_PCT}%) …" | tee -a "$LOG"
SUMMARY=$("$PYTHON" -m renquant_base_data.finnhub_analyst_ratings_refresh \
    --watchlist "$STRATEGY_CONFIG" \
    --output "$OUTPUT" \
    --max-pull "$MAX_PULL" \
    --sleep-sec "$SLEEP_SEC" \
    --min-coverage-pct "$MIN_COVERAGE_PCT" \
    --fail-on-error \
    2>>"$LOG")
RC=$?
echo "[$(date '+%H:%M:%S')] summary: ${SUMMARY:-<none>}" | tee -a "$LOG"

[[ $RC -eq 0 && -n "$SUMMARY" ]] || fail "refresh exited $RC (coverage gate or run error) — ${SUMMARY:-no output}"

READOUT=$("$PYTHON" - "$SUMMARY" <<'PY'
import json, sys
try:
    s = json.loads(sys.argv[1])
except Exception as e:
    print(f"PARSE_ERR|{e}"); raise SystemExit(0)
ok = "OK" if s.get("with_data", 0) > 0 else "ZERO_DATA"
print("{}|with_data={} cov(coverable)={}% etf_no_cov={} quota_err={} fetch_err={} store={}names rows={}".format(
    ok, s.get("with_data"), s.get("coverage_pct"), s.get("no_coverage"),
    s.get("quota_error"), s.get("fetch_error"), s.get("tickers_in_store"), s.get("total_rows")))
PY
)
STATE="${READOUT%%|*}"; BODY="${READOUT#*|}"
case "$STATE" in
    OK)        notify "FINNHUB ANALYST ✓" "$DATE  $BODY" ;;
    ZERO_DATA) fail "0 names returned recommendations — $BODY" ;;
    *)         fail "could not parse refresh summary — $READOUT" ;;
esac
echo "[$(date '+%H:%M:%S')] DONE — $BODY" | tee -a "$LOG"
