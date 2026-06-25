#!/usr/bin/env bash
# check_readonly_e2e.sh — gold-standard deploy verify: run a FULL readonly
# daily-full end-to-end and assert it produces a decision (does not crash, does
# not go silent). The heavier companion to check_conviction_admits.py: it
# exercises the WHOLE pipeline code path (panel assembly → scoring → gates → QP →
# sizing → execution-plan), so a broad pin bump (e.g. the orchestrator) that
# breaks any stage is caught here, not in production.
#
# SAFE BY CONSTRUCTION: reuses the daily shadow mechanism — `--broker
# readonly-alpaca` + the shadow config whose broker_name=alpaca_shadow isolates
# ALL state to live_state.alpaca_shadow.json + runs_alpaca_shadow.db. It places
# NO orders and NEVER touches prod state/db. (The shadow scorer differs from
# prod, but the pipeline CODE PATH is shared — which is exactly what a code/pin
# bump verify must exercise; prod-scorer specifics are covered by
# check_conviction_admits.py + the bundle check.)
#
# Intended as the `promote_pin.py --verify-cmd` for BROAD bumps, and a
# `make doctor` deep check. Exit 0 = clean decision produced, 1 = crash/timeout/
# no-decision (would-not-trade), 2 = setup error.
set -uo pipefail

REPO_DIR="${RENQUANT_REPO_DIR:-/Users/renhao/git/github/RenQuant}"
PYTHON="${RENQUANT_PYTHON:-$REPO_DIR/.venv/bin/python}"
TIMEOUT_SEC="${RENQUANT_E2E_TIMEOUT_SEC:-1200}"
LOG="${RENQUANT_E2E_LOG:-/tmp/check_readonly_e2e.$$.log}"

cd "$REPO_DIR" || { echo "SETUP: cannot cd $REPO_DIR"; exit 2; }
[ -f "$REPO_DIR/.env" ] && { set -a; # shellcheck disable=SC1091
  source "$REPO_DIR/.env"; set +a; }
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh" || { echo "SETUP: subrepo_env"; exit 2; }
renquant_load_subrepo_env "$REPO_DIR"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$(dirname "$REPO_DIR")")"
export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"
export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-orchestrator renquant-common renquant-base-data renquant-artifacts renquant-model renquant-pipeline renquant-execution renquant-strategy-104 renquant-backtesting):${PYTHONPATH:-}"

# Machine-checked ISOLATION contract: fingerprint the PROD state/db before the
# run; assert UNCHANGED after (a config/broker regression that wrote prod state
# must FAIL this guard, not pass). mtime+size, "MISSING" if absent.
PROD_DB="$REPO_DIR/data/runs.alpaca.db"
SHADOW_DB="$REPO_DIR/data/runs.alpaca_shadow.db"
PROD_STATE="$REPO_DIR/backtesting/renquant_104/live_state.alpaca.json"
fingerprint() { stat -f '%m-%z' "$1" 2>/dev/null || echo "MISSING"; }
PROD_DB_BEFORE="$(fingerprint "$PROD_DB")"
PROD_STATE_BEFORE="$(fingerprint "$PROD_STATE")"
SHADOW_DB_BEFORE="$(fingerprint "$SHADOW_DB")"

echo "[readonly-e2e] running isolated shadow e2e (timeout ${TIMEOUT_SEC}s) → $LOG"
RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1 "$PYTHON" - "$REPO_DIR" "$TIMEOUT_SEC" > "$LOG" 2>&1 <<'PY'
import os, subprocess, sys
repo, timeout = sys.argv[1], float(sys.argv[2])
runner = ([sys.executable, "-m", "live.runner"]
          if os.environ.get("RQ_DAILY_RUNNER", "multirepo") == "umbrella"
          else [sys.executable, "-m", "renquant_orchestrator", "live-bridge", "--repo-dir", repo])
cmd = runner + ["--strategy", "renquant_104", "--broker", "readonly-alpaca",
                "--once", "--strategy-config-name", "strategy_config.shadow.json"]
try:
    raise SystemExit(subprocess.run(cmd, cwd=repo, timeout=timeout).returncode)
except subprocess.TimeoutExpired:
    print("E2E_TIMEOUT", flush=True); raise SystemExit(124)
PY
RC=$?
tail -3 "$LOG" 2>/dev/null

# (1) ISOLATION assertion (machine-checked): prod db/state must be UNCHANGED.
PROD_DB_AFTER="$(fingerprint "$PROD_DB")"
PROD_STATE_AFTER="$(fingerprint "$PROD_STATE")"
SHADOW_DB_AFTER="$(fingerprint "$SHADOW_DB")"
if [ "$PROD_DB_AFTER" != "$PROD_DB_BEFORE" ] || [ "$PROD_STATE_AFTER" != "$PROD_STATE_BEFORE" ]; then
    echo "READONLY_E2E: FAIL — ISOLATION BREACH (prod state changed: db $PROD_DB_BEFORE→$PROD_DB_AFTER, state $PROD_STATE_BEFORE→$PROD_STATE_AFTER)"
    exit 1
fi

# Exit code first.
if [ "$RC" -ne 0 ]; then
    echo "READONLY_E2E: FAIL (rc=$RC$( [ "$RC" = 124 ] && echo '/timeout' ))"
    exit 1
fi

# (2) DECISION assertion: require DECISION-SPECIFIC evidence (a committed cycle
# decision / persisted gate verdicts), NOT mere pipeline-progress markers.
DECISION=$(grep -cE "ntfy sent:|SHADOW-DECISION|cycle decision|gate_verdicts: wrote|RunnerAdapter\.commit:" "$LOG" 2>/dev/null)
if [ "${DECISION:-0}" -lt 1 ]; then
    echo "READONLY_E2E: FAIL — ran but produced NO committed decision (silent)"
    exit 1
fi

# The run must actually have EXECUTED in isolation (shadow db advanced).
if [ "$SHADOW_DB_AFTER" = "$SHADOW_DB_BEFORE" ]; then
    echo "READONLY_E2E: WARN — shadow db unchanged; e2e may not have persisted (decision=$DECISION)"
fi
echo "READONLY_E2E: OK — isolated readonly pipeline produced a committed decision ($DECISION marker(s)); prod state untouched"
exit 0
