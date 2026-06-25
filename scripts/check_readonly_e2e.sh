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

# Verdict: the run must (a) exit 0 and (b) have emitted a cycle decision.
DECISION=$(grep -cE "ntfy sent:|cycle decision|InferencePipeline DONE|orders placed|candidates from" "$LOG" 2>/dev/null)
if [ "$RC" -ne 0 ]; then
    echo "READONLY_E2E: FAIL (rc=$RC$( [ "$RC" = 124 ] && echo '/timeout' ))"
    exit 1
fi
if [ "${DECISION:-0}" -lt 1 ]; then
    echo "READONLY_E2E: FAIL — completed but produced NO decision (silent/would-not-trade)"
    exit 1
fi
echo "READONLY_E2E: OK — full readonly pipeline produced a decision ($DECISION marker(s))"
exit 0
