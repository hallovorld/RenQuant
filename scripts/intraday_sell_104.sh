#!/usr/bin/env bash
# intraday_sell_104.sh — Mid-market-hours exit-only pass with Alpaca 5-min bars.
# ─── LIVE MODE (restored 2026-05-11 PM) ─────────────────────────────────
# 2026-05-11 PM: live trading restored per user request (Bug C fix shows
# strategy is profitable: +11.6% APY mean Sharpe 0.77 across 3 windows).
# To restore PAPER mode for safety testing:
#   sed -i "" "s/--broker alpaca/--broker paper/g" scripts/*.sh
# Or add ALPACA_PAPER_API_KEY/SECRET to .env + switch to --broker alpaca-paper
# for Alpaca's paper-trading sandbox (real API, no real money).
# ─────────────────────────────────────────────────────────────────────────
#
# Runs every ~30 min during market hours. Uses Alpaca's IEX 5-min feed to
# overlay the latest intraday close on today's daily bar, then calls
# SellOnlyPipeline → triggers stop-loss / trailing-stop / SDL / max-hold /
# model-sell signals against the fresh price. Never places buys.
#
# Complements daily_104.sh (runs once, after close). Catches gap-down days
# where EOD evaluation would have already locked in the loss.
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
# 2026-05-11 audit M-env: switched conda → .venv per feedback_python_env.md
VENV_DIR="/Users/renhao/git/github/RenQuant/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$REPO_DIR/logs/intraday_104"
NTFY_TOPIC="renquant"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
HHMM=$(date +%H%M)
LOG="$LOG_DIR/$DATE.log"

notify() {
    local title="$1" body="$2"
    if command -v terminal-notifier &>/dev/null; then
        terminal-notifier -title "$title" -message "$body" -sound Glass 2>/dev/null || true
    fi
    curl -s -H "Title: $title" -d "$body" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
}

# Load Alpaca credentials
CRED_FILE="$REPO_DIR/.env"
if [ -f "$CRED_FILE" ]; then
    set -a; source "$CRED_FILE"; set +a
else
    echo "ERROR: $CRED_FILE not found." | tee -a "$LOG"
    exit 1
fi

# Resolve pinned subrepo runtime before invoking live_multirepo.py. Missing
# assembly env falls back to sibling checkouts.
GITHUB_DIR="$(dirname "$REPO_DIR")"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh"
renquant_load_subrepo_env "$REPO_DIR"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR")"
export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"
export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-orchestrator renquant-common renquant-base-data renquant-artifacts renquant-model renquant-pipeline renquant-execution renquant-strategy-104 renquant-backtesting):${PYTHONPATH:-}"

exec >> "$LOG" 2>&1
echo "=== intraday_sell started at $(date) (HHMM=$HHMM) ==="

# NYSE calendar guard — skip on holidays/weekends
TODAY_DATE=$(date +%Y-%m-%d)
if ! "$PYTHON" -c "
import sys, pandas_market_calendars as mcal
cal = mcal.get_calendar('NYSE')
sched = cal.schedule('$TODAY_DATE', '$TODAY_DATE')
sys.exit(0 if len(sched) > 0 else 1)
"; then
    echo "NYSE closed today ($TODAY_DATE) — skipping."
    exit 0
fi

# Lock file — prevent stepping on daily_104.sh or another intraday run
LOCK_FILE="/tmp/renquant_104_intraday.lock"
if ! ( set -C; echo $$ > "$LOCK_FILE" ) 2>/dev/null; then
    EXISTING_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "?")
    echo "Another intraday_sell is active (PID=$EXISTING_PID) — skipping."
    exit 0
fi
trap "rm -f '$LOCK_FILE'" EXIT

cd "$REPO_DIR"

# ── Preflight: align subrepo checkouts to the audited pins, fail-closed ────────
# Run only after duplicate/holiday exits so sync checkout is serialized and only
# happens for a real trading run. Umbrella-lag check is left to the daily run.
# shellcheck source=scripts/preflight_pin_align.sh
source "$REPO_DIR/scripts/preflight_pin_align.sh"

if [ "${RQ_DAILY_RUNNER:-multirepo}" = "umbrella" ]; then
    RUNNER_ARGS=(-m live.runner)
else
    # Sanity-gate the multirepo path (mirrors daily_104.sh PR #147 +
    # PR #155 runbook). Fails fast with a runbook pointer instead of
    # leaving the cron to surface a cryptic argparse error every 12
    # minutes (the 2026-06-03 incident:
    # doc/ops/2026-06-03-orchestrator-bridge-runtime-drift-incident.md).
    if ! "$PYTHON" "$REPO_DIR/scripts/runtime_qp_sanity_check.py"; then
        echo "=== intraday_sell RUNTIME-SANITY-FAIL at $(date) ==="
        notify "RenQuant 104 RUNTIME-SANITY-FAIL" \
            "Stale or incomplete multirepo runtime; run make subrepo-runtime-root and paper-smoke daily_104. Runbook: doc/ops/subrepo-runtime-refresh-runbook.md"
        exit 1
    fi
    RUNNER_ARGS=(-m renquant_orchestrator live-bridge --repo-dir "$REPO_DIR")
fi

# ── Software-stop registry BOOTSTRAP (seed) — unconditional, never blocking ──
# orch#1078 follow-up (2026-08-29). Creates, once and only if absent, an
# empty schema-valid software-stop registry at EXACTLY the path the liveness
# pager (renquant_execution.software_stops_liveness --data-root
# ~/.renquant/runtime/software-stops) and the runner (adapters/
# software_stops_wiring.py, same neutral root) resolve, so the registry is
# LOCATABLE. It runs on the pinned PYTHONPATH set above (orchestrator +
# pipeline) and MUST use the same --broker the runner receives below (the
# registry file is broker-tagged: software_stops.<broker>.json).
#
# A seed is bootstrap plumbing, NOT readiness (readiness = a real writer
# heartbeat, `... readiness --broker alpaca`), and it is NOT a trading
# precondition: this loop is the live book's EXIT path, so a failed seed is
# logged loudly (and paged) and the sell pass CONTINUES. There is no `exit`
# in this block by design; `set -e` is not in effect (see `set -uo pipefail`).
# Exit codes (orchestrator contract): 0 SEEDED/EXISTS; 1 usage; 2 an existing
# file is CORRUPT (left untouched — operator decision); 3 pipeline module not
# importable. A pinned orchestrator that predates the seeder (< orch#1078)
# exits 0 WITHOUT a verdict line — flagged as a WARNING, not silently green.
SEED_MODULE="renquant_orchestrator.software_stops_registry_contract"
SEED_OUT=$("$PYTHON" -m "$SEED_MODULE" seed --broker alpaca 2>&1)
SEED_RC=$?
if [ "$SEED_RC" -eq 0 ]; then
    if printf '%s\n' "$SEED_OUT" | grep -qE '^(SEEDED|EXISTS): '; then
        echo "software-stops registry seed OK: $(printf '%s\n' "$SEED_OUT" | grep -E '^(SEEDED|EXISTS): ')"
    else
        echo "WARNING: software-stops registry seed exited 0 WITHOUT a SEEDED/EXISTS verdict (pinned orchestrator predates the seeder, orch#1078?) — registry NOT confirmed locatable; continuing with the sell pass. Output: ${SEED_OUT:-<none>}"
    fi
else
    echo "ERROR: software-stops registry seed FAILED (exit $SEED_RC) — continuing with the sell pass (the seed is bootstrap plumbing and never blocks the exit path). Output: ${SEED_OUT:-<none>}"
    notify "RenQuant 104 software-stops SEED FAILED" "exit $SEED_RC (1 usage / 2 CORRUPT registry left untouched / 3 pipeline import) — sell pass continued; check $LOG"
fi

if "$PYTHON" "${RUNNER_ARGS[@]}" --strategy renquant_104 --broker alpaca --once \
        --sell-only --intraday; then
    echo "=== intraday_sell finished at $(date) ==="
else
    echo "=== intraday_sell FAILED at $(date) ==="
    notify "RenQuant 104 intraday ERROR" "Check $LOG"
    exit 1
fi
