#!/bin/bash
# Queue runner for 2026-05-15 regime-conditional re-evaluation panels.
#
# **STRICTLY SEQUENTIAL** — runs panels ONE AT A TIME (each panel
# already saturates 6 cores via xargs -P 6 inside run_p0_panel.sh).
# Earlier parallel-attempt overloaded the host (36 workers on 10 cores,
# load avg 300+).
#
# Usage:
#   nohup ./scripts/run_regime_reeval_queue.sh > logs/reeval_queue.log 2>&1 &
#
# Idempotent: skips panels whose 16/16 equity files already exist.

set -u
cd /Users/renhao/git/github/RenQuant
RUNNER=/tmp/run_p0_panel.sh
test -x "${RUNNER}" || { echo "FATAL: ${RUNNER} not executable"; exit 1; }

# Wait until prior panel's workers have COMPLETELY drained before launching next.
# A panel script (run_p0_panel.sh) blocks until xargs finishes ⇒ we just need
# to ensure no other run_sim_104.py is in flight.
wait_for_drain() {
  # macOS pgrep -fc is NOT supported (returns usage error). Use ps + grep.
  while true; do
    local busy
    busy=$(ps -eo command | grep -c "scripts/run_sim_104\.py")
    # Subtract 1 for the grep itself (always matches "run_sim_104")
    busy=$(( busy - 1 ))
    if (( busy <= 0 )); then
      return
    fi
    echo "$(date +%H:%M:%S) — ${busy} sim workers still running, waiting…"
    sleep 60
  done
}

PANELS=(
  p0activated_regime_aware
  re_stop007
  re_sdl_n2
  re_trail015
  re_cvar025
  re_cvar050
  re_kelly_t1_035
)

is_done() {
  local label="$1"
  local equity_dir="data/logs/sim_2026-05-15_${label}/equity"
  if [[ ! -d "${equity_dir}" ]]; then return 1; fi
  local n
  n=$(ls "${equity_dir}" 2>/dev/null | wc -l | tr -d ' ')
  if (( n == 16 )); then return 0; fi
  return 1
}

for label in "${PANELS[@]}"; do
  if is_done "${label}"; then
    echo "$(date +%H:%M:%S) — ${label} already complete (16/16); skip"
    continue
  fi
  wait_for_drain
  echo "$(date +%H:%M:%S) — launching ${label} (sequential)"
  # FOREGROUND — block until this panel's 16 windows fully complete.
  # No `&` — that's what caused the overload before.
  "${RUNNER}" "${label}"
  echo "$(date +%H:%M:%S) — ${label} DONE"
done

echo "$(date +%H:%M:%S) — all regime-reeval panels DONE"
