#!/usr/bin/env bash
# daily_analyst_ratings_finnhub_refresh.sh — daily broad-coverage Finnhub pull.
#
# WHY FINNHUB, DAILY (2026-06-25):
# Finnhub's free `/stock/recommendation` gives monthly analyst recommendation
# distributions with broad US-stock coverage — unlike FMP free's ~30% plan-lock
# (HTTP 402). Coverage is broad but NOT assumed full: live probe 136/145 returned
# data, 9 empty (no_coverage). An empty response is AMBIGUOUS (ETF/index,
# delisted/unsupported, vendor-empty, or a real stock with no current recs) — the
# fetcher can't tell which, so we report active_coverage_pct + no_coverage_pct +
# no_coverage_samples and never assume the gap is ETFs.
# Free tier is 60 calls/min, so the whole ~145-name watchlist fits in one daily
# pass (~2.5 min at ~1s throttle) — no rotation needed (MAX_PULL=0 = all). The
# free window is only ~4 months, so DAILY accumulation (dedup by (ticker,period))
# grows the multi-month series the 3-month revision signal needs.
#
# FAIL-CLOSED (silent-degradation lesson): a quota hit / bad key / schema break
# must ntfy ✗ and exit non-zero, NOT pass as "no coverage". We gate on:
#   * --fail-on-error → CLI exits non-zero on ANY quota/fetch error;
#   * an ACTIVE-coverage floor checked here over the FULL requested set
#     (with_data/requested), NOT the coverable set — so a real stock silently
#     dropping into the ambiguous no_coverage bucket still trips the gate.
#
# INERT UNTIL DEPLOYED: calls `renquant_base_data.finnhub_analyst_ratings_refresh`
# (base-data #25). Until that merges + the base-data pin is bumped, the import
# fails — the wrapper reports ✗ "module unavailable" rather than crashing.
# ACTIVATION (do NOT load this plist before): (1) #25 merged + base-data pin
# bumped; (2) one-shot dry-run (`bash scripts/daily_analyst_ratings_finnhub_refresh.sh`)
# proving the active/no_coverage metrics and fail-closed behaviour on today's
# watchlist; (3) THEN `launchctl load` the plist.
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
MAX_PULL=0                  # 0 = whole watchlist daily (60/min fits ~145 in one pass)
SLEEP_SEC=1.1               # throttle for the free 60/min cap
MIN_ACTIVE_COVERAGE_PCT=88  # systemic-collapse floor over the FULL requested set
                            # (with_data/requested). ~9 ETFs/indices are perma-empty
                            # → active ≈94%; 88 catches a real collapse with headroom.
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

echo "[$(date '+%H:%M:%S')] Pulling full watchlist (throttle ${SLEEP_SEC}s, active floor ${MIN_ACTIVE_COVERAGE_PCT}%) …" | tee -a "$LOG"
# --min-coverage-pct 0: the CLI's coverable-set gate is disabled here on purpose;
# fail-closed on quota/fetch errors stays (--fail-on-error), and the HONEST
# coverage gate is the active-set floor enforced below over the full requested set.
SUMMARY=$("$PYTHON" -m renquant_base_data.finnhub_analyst_ratings_refresh \
    --watchlist "$STRATEGY_CONFIG" \
    --output "$OUTPUT" \
    --max-pull "$MAX_PULL" \
    --sleep-sec "$SLEEP_SEC" \
    --min-coverage-pct 0 \
    --fail-on-error \
    2>>"$LOG")
RC=$?
echo "[$(date '+%H:%M:%S')] summary: ${SUMMARY:-<none>}" | tee -a "$LOG"

[[ $RC -eq 0 && -n "$SUMMARY" ]] || fail "refresh exited $RC (fail-on-error or run error) — ${SUMMARY:-no output}"

READOUT=$("$PYTHON" - "$SUMMARY" "$MIN_ACTIVE_COVERAGE_PCT" <<'PY'
import json, sys
try:
    s = json.loads(sys.argv[1])
except Exception as e:
    print(f"PARSE_ERR|{e}"); raise SystemExit(0)
floor = float(sys.argv[2])
wd = s.get("with_data", 0) or 0
active = s.get("active_coverage_pct", 0.0) or 0.0
state = "ZERO_DATA" if wd == 0 else ("LOW_ACTIVE" if active < floor else "OK")
print("{}|with_data={} active={}% cov(coverable)={}% no_cov={}%({}) quota_err={} fetch_err={} store={}names rows={}".format(
    state, wd, active, s.get("coverage_pct"), s.get("no_coverage_pct"), s.get("no_coverage"),
    s.get("quota_error"), s.get("fetch_error"), s.get("tickers_in_store"), s.get("total_rows")))
PY
)
STATE="${READOUT%%|*}"; BODY="${READOUT#*|}"
case "$STATE" in
    OK)         notify "FINNHUB ANALYST ✓" "$DATE  $BODY" ;;
    ZERO_DATA)  fail "0 names returned recommendations — $BODY" ;;
    LOW_ACTIVE) fail "active coverage below ${MIN_ACTIVE_COVERAGE_PCT}% — possible systemic collapse — $BODY" ;;
    *)          fail "could not parse refresh summary — $READOUT" ;;
esac
echo "[$(date '+%H:%M:%S')] DONE — $BODY" | tee -a "$LOG"
