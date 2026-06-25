#!/usr/bin/env bash
# daily_analyst_ratings_finnhub_refresh.sh — daily broad-coverage Finnhub pull.
#
# WHY FINNHUB, DAILY (2026-06-25):
# Finnhub's free `/stock/recommendation` gives monthly analyst recommendation
# distributions with broad US-stock coverage — unlike FMP free's ~30% plan-lock
# (HTTP 402). Coverage is broad but NOT assumed full: live probe 136/145 returned
# data, 9 empty (no_coverage). An empty response is AMBIGUOUS (ETF/index,
# delisted/unsupported, vendor-empty, or a real stock with no current recs) — the
# fetcher can't tell which, so the CLI reports active_coverage_pct +
# no_coverage_pct + no_coverage_samples and never assumes the gap is ETFs.
# Free tier is 60 calls/min, so the whole ~145-name watchlist fits in one daily
# pass (~2.5 min at ~1s throttle) — no rotation needed (MAX_PULL=0 = all). The
# free window is only ~4 months, so DAILY accumulation (dedup by (ticker,period))
# grows the multi-month series the 3-month revision signal needs.
#
# FAIL-CLOSED (silent-degradation lesson): a quota hit / bad key / schema break
# must ntfy ✗ and exit non-zero, NOT pass as "no coverage". The gating now lives
# entirely in the base-data CLI (tested Python, base-data #25), NOT in this shell:
#   * --fail-on-error            → CLI exits non-zero on ANY quota/fetch error;
#   * --min-active-coverage-pct  → CLI fail-closes on active_coverage_pct
#                                  (with_data/requested, the FULL requested set),
#                                  so a real stock silently dropping into the
#                                  ambiguous no_coverage bucket counts AGAINST the
#                                  floor and trips the gate. THIS is the
#                                  fail-closed coverage control.
#   * --min-coverage-pct         → DIAGNOSTIC ONLY (coverage over the coverable
#                                  set, which excludes the ambiguous no_coverage);
#                                  left at 0 here — it cannot fail-close on a
#                                  widespread-empty run, so it is not the gate.
# This wrapper is intentionally THIN: it invokes the CLI (which fail-closes on
# active coverage + errors and exits non-zero), then reports the summary to ntfy.
# It does NOT re-implement the coverage gate in bash — that gate is the tested
# Python one (base-data #25 added boundary tests for 0%, at-floor, below-floor,
# missing-metric, and quota/fetch-error interactions).
#
# THRESHOLD (provisional, pre-registered — see the progress doc): the 88% active
# floor is a COARSE systemic-collapse guard derived from a SINGLE 136/145 probe
# (active ≈ 93.8%), not a tuned threshold. During a baseline window (first ~10
# daily cron runs) we record active_coverage_pct, then re-set the floor at the
# observed lower-tail (min / 5th-pct) minus explicit headroom. Until then treat 88
# as a placeholder that catches a real collapse, not a calibrated bound.
#
# INERT UNTIL DEPLOYED: calls `renquant_base_data.finnhub_analyst_ratings_refresh`
# (base-data #25). Until that merges + the base-data pin is bumped, the import
# fails — the wrapper reports ✗ "module unavailable" rather than crashing.
# ACTIVATION ORDER (do NOT load this plist before): (1) base-data #25 merged +
# the base-data pin bumped; (2) a one-shot dry-run
# (`bash scripts/daily_analyst_ratings_finnhub_refresh.sh`) proving BOTH paths on
# today's watchlist — the PASS path (healthy active coverage → ntfy ✓) AND the
# fail-closed path (force a low floor / simulated error → ✗, exit non-zero);
# (3) THEN `launchctl load` the plist.
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
MIN_ACTIVE_COVERAGE_PCT=88  # provisional fail-closed floor on active_coverage_pct
                            # (with_data/requested, the FULL requested set), enforced
                            # by the CLI. Coarse collapse guard from one 136/145 probe
                            # (active ≈94%); to be recalibrated over a baseline window
                            # (see the progress doc). 88 catches a real collapse.
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
# The CLI owns the fail-closed gating (tested Python, base-data #25):
#   --min-active-coverage-pct → fail-close on active_coverage_pct (the gate);
#   --fail-on-error           → fail-close on ANY quota/fetch error;
#   --min-coverage-pct 0      → diagnostic coverable cov only (NOT a safety gate).
# The CLI prints the JSON summary to stdout and exits non-zero on any violation.
SUMMARY=$("$PYTHON" -m renquant_base_data.finnhub_analyst_ratings_refresh \
    --watchlist "$STRATEGY_CONFIG" \
    --output "$OUTPUT" \
    --max-pull "$MAX_PULL" \
    --sleep-sec "$SLEEP_SEC" \
    --min-active-coverage-pct "$MIN_ACTIVE_COVERAGE_PCT" \
    --min-coverage-pct 0 \
    --fail-on-error \
    2>>"$LOG")
RC=$?
echo "[$(date '+%H:%M:%S')] summary: ${SUMMARY:-<none>}" | tee -a "$LOG"

# The CLI already fail-closed on active coverage + errors (exit non-zero). The
# wrapper just maps that to ntfy: non-zero exit OR empty summary → ✗.
if [[ $RC -ne 0 || -z "$SUMMARY" ]]; then
    fail "refresh exited $RC (active-coverage gate / --fail-on-error / run error) — ${SUMMARY:-no output}"
fi

# Thin readout: parse the CLI summary for the ntfy ✓ body. No gate here — the CLI
# already enforced the active-coverage floor and error fail-close above.
BODY=$("$PYTHON" - "$SUMMARY" <<'PY'
import json, sys
try:
    s = json.loads(sys.argv[1])
except Exception as e:
    print(f"PARSE_ERR|{e}"); raise SystemExit(0)
print("OK|with_data={} active={}% cov(coverable)={}% no_cov={}%({}) quota_err={} fetch_err={} store={}names rows={}".format(
    s.get("with_data"), s.get("active_coverage_pct"), s.get("coverage_pct"),
    s.get("no_coverage_pct"), s.get("no_coverage"),
    s.get("quota_error"), s.get("fetch_error"), s.get("tickers_in_store"), s.get("total_rows")))
PY
)
STATE="${BODY%%|*}"; BODY="${BODY#*|}"
if [[ "$STATE" != "OK" ]]; then
    fail "could not parse refresh summary — $BODY"
fi
notify "FINNHUB ANALYST ✓" "$DATE  $BODY"
echo "[$(date '+%H:%M:%S')] DONE — $BODY" | tee -a "$LOG"
