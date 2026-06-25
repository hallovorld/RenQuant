#!/usr/bin/env bash
# daily_analyst_ratings_refresh.sh — incremental FMP analyst-ratings pull.
#
# WHY DAILY + SMALL BATCHES (2026-06-24):
# FMP's free BASIC tier (`/stable/grades-historical`, ~7.5y monthly rating
# distributions) is capped at 250 calls/day AND a per-minute rate limit. One
# ticker = one call. Bursting the whole ~142-name watchlist trips the
# per-minute cap (a one-shot backfill missed 104/142). So instead of one weekly
# burst we pull a SMALL incremental batch every day: the refresh CLI's
# `select_to_refresh` ranks never-fetched tickers first, then oldest
# `fetched_at`, and takes `--max-pull` of them — rotating the full watchlist
# every few days, always under both caps. Ratings update monthly, so a few-day
# rotation keeps every name fresh enough for the consensus-revision signal.
#
# FAIL-CLOSED (2026-06-24 silent-degradation lesson, same bug fixed in
# weekly_fundamental_refresh): a quota hit / bad key / schema break must ntfy ✗
# and exit non-zero, NOT pass silently as "no coverage". The refresh CLI
# distinguishes with_data / no_coverage / premium_restricted / quota_error /
# fetch_error, and we gate on BOTH:
#   * --fail-on-error   → exit non-zero if ANY quota_error/fetch_error occurs
#                         (the true fail-closed contract; premium_restricted is
#                         the permanent free-tier plan ceiling and is EXCLUDED
#                         from the error count, so it never false-alarms);
#   * --min-coverage-pct → exit non-zero on a SYSTEMIC coverage collapse over the
#                         *coverable* (non-premium) set.
# So a transient 429 or a creeping breakage trips ✗ instead of passing as "no
# coverage"; the per-outcome counts are also surfaced in the ntfy body.
#
# SCOPE (free tier ~30% active): --min-coverage-pct gates the COVERABLE subset
# (the only fraction a free key can move). The ntfy body ALSO reports active=%
# (with_data/requested) and premium_locked=% so a high coverable cov is never
# misread as full active-watchlist coverage. This is subset-only ingestion infra,
# NOT a production analyst feature — no model/retrain decision rides on it.
#
# INERT UNTIL DEPLOYED: this calls `renquant_base_data.fmp_analyst_ratings_refresh`
# (base-data PR #24). Until that merges and the base-data pin is bumped, the
# module import fails — the wrapper reports that explicitly (✗ "module
# unavailable") rather than crashing cryptically.
#
# Schedule: daily 04:20 PT (offline hours, after the weekly fund refresh slot).
# Plist: scripts/launchd/com.renquant.daily-analyst-ratings.plist
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
VENV_DIR="/Users/renhao/git/github/RenQuant/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$REPO_DIR/logs/daily_analyst_ratings_refresh"
NTFY_TOPIC="renquant"
OUTPUT="$REPO_DIR/data/analyst_ratings_fmp.parquet"
MAX_PULL=40            # names per run; rotates the ~142 watchlist every ~4 days
SLEEP_SEC=1            # throttle for the free per-minute cap
MIN_COVERAGE_PCT=75    # secondary floor on the COVERABLE set; --fail-on-error already fails on ANY quota/fetch error (no per-429 tolerance)
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"

notify() {
    local title="$1" body="$2"
    curl -s -H "Title: $title" -d "$body" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
}

fail() {  # message — alert + exit non-zero so launchd/operator see it
    local msg="$1"
    echo "[$(date '+%H:%M:%S')] FAIL: $msg" | tee -a "$LOG"
    notify "ANALYST RATINGS ✗" "$DATE  $msg"
    exit 1
}

cd "$REPO_DIR" || fail "cannot cd to $REPO_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
# FMP_API_KEY lives in .env (gitignored, never committed)
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
echo "[$(date '+%H:%M:%S')] Daily analyst-ratings refresh — $DATE" | tee -a "$LOG"

[[ -n "${FMP_API_KEY:-}" ]] || fail "FMP_API_KEY not set (.env)"

# Resolve the pinned strategy watchlist (the names we actually trade/score).
if ! STRATEGY_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json)"; then
    fail "pinned renquant-strategy-104 strategy_config.json unavailable"
fi

# Inert-until-deployed guard: the base-data FMP module must be importable.
if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import renquant_base_data.fmp_analyst_ratings_refresh  # noqa: F401
PY
then
    fail "renquant_base_data.fmp_analyst_ratings_refresh unavailable (base-data #24 not merged / pin not bumped)"
fi

# ── Incremental pull (JSON summary → stdout, logs → stderr) ─────────────
echo "[$(date '+%H:%M:%S')] Pulling up to $MAX_PULL most-stale names (throttle ${SLEEP_SEC}s, floor ${MIN_COVERAGE_PCT}%) …" | tee -a "$LOG"
SUMMARY=$("$PYTHON" -m renquant_base_data.fmp_analyst_ratings_refresh \
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

# Compact the JSON summary for the ntfy body + a defensive with_data>0 check.
READOUT=$("$PYTHON" - "$SUMMARY" <<'PY'
import json, sys
try:
    s = json.loads(sys.argv[1])
except Exception as e:
    print(f"PARSE_ERR|{e}"); raise SystemExit(0)
ok = "OK" if s.get("with_data", 0) > 0 else "ZERO_DATA"
# Show coverable cov (what the gate moves) AND active cov + premium-lock so a
# high coverable % is never read as full active-watchlist coverage (Codex #402).
print("{}|with_data={} cov(coverable)={}% active={}% premium_locked={}% quota_err={} fetch_err={} store={}names rows={}".format(
    ok, s.get("with_data"), s.get("coverage_pct"), s.get("active_coverage_pct"),
    s.get("premium_restricted_pct"), s.get("quota_error"),
    s.get("fetch_error"), s.get("tickers_in_store"), s.get("total_rows")))
PY
)
STATE="${READOUT%%|*}"; BODY="${READOUT#*|}"
case "$STATE" in
    OK)        notify "ANALYST RATINGS ✓" "$DATE  $BODY" ;;
    ZERO_DATA) fail "0 names returned ratings — $BODY" ;;
    *)         fail "could not parse refresh summary — $READOUT" ;;
esac
echo "[$(date '+%H:%M:%S')] DONE — $BODY" | tee -a "$LOG"
