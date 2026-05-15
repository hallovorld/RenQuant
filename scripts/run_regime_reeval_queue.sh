#!/bin/bash
# Queue runner for 2026-05-15 regime-conditional re-evaluation panels.
#
# Waits until host has ≤ N busy sim/NGBoost workers, then triggers the
# next panel from the queue. See doc/research/2026-05-15-regime-reeval-plan.md.
#
# Usage:
#   nohup ./scripts/run_regime_reeval_queue.sh > logs/reeval_queue.log 2>&1 &
#
# Idempotent: skips panels whose 16/16 equity files already exist.

set -u
cd /Users/renhao/git/github/RenQuant
RUNNER=/tmp/run_p0_panel.sh
test -x "${RUNNER}" || { echo "FATAL: ${RUNNER} not executable"; exit 1; }

# Wait until we have <= MAX_BUSY heavy compute processes running
MAX_BUSY=${MAX_BUSY:-2}    # default: allow up to 2 sim panels concurrently (6 workers each = 12 cores)

PANELS=(
  re_stop007
  re_sdl_n2
  re_trail015
  re_cvar025
  re_cvar050
  re_kelly_t1_035
)

wait_for_slot() {
  while true; do
    local busy
    busy=$(pgrep -fc "run_sim_104.py\|train_ngboost_proper" 2>/dev/null || echo 0)
    # Each panel launches up to 6 workers. So MAX_BUSY=2 ⇒ allow ≤12 workers total.
    if (( busy <= MAX_BUSY * 6 )); then
      return
    fi
    echo "$(date +%H:%M:%S) — ${busy} busy workers, waiting…"
    sleep 60
  done
}

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
  wait_for_slot
  echo "$(date +%H:%M:%S) — launching ${label}…"
  "${RUNNER}" "${label}" &
  # Brief stagger so xargs workers from this panel and the prior one
  # don't all start within the same second (helps RAM peak smooth)
  sleep 30
done

# Wait for the final panel to finish
wait
echo "$(date +%H:%M:%S) — all 6 regime-reeval panels DONE"
