#!/bin/bash
# RETIRED (2026-07-14, PR #471 r5): completed re-evaluation queue.
# run_sim_104.py now requires strict-pinned configs or an experiment manifest;
# --strategy-config-name with local sweep configs no longer works.
# To re-run: register experiment manifests in experiments/manifests/ and
# use --experiment-manifest instead.
echo "ERROR: this script is RETIRED — see header for migration instructions" >&2
exit 1
# --- original below for reference ---
# Wait for re_trail015 16/16, then:
#   1. Stop the sequential queue (so it doesn't double-launch)
#   2. Re-run missing re_sdl_n2 Q13 (silent race-loss earlier)
#   3. Launch the remaining 3 panels (re_cvar025, re_cvar050,
#      re_kelly_t1_035) in PARALLEL with reduced per-panel -P.
#
# Per-panel concurrency: 3 (was 6). 3 panels × 3 workers = 9 workers
# = ~16 GB RAM on 32 GB host. Fits comfortably.
#
# Usage:
#   nohup ./scripts/run_parallel_after_trail015.sh > logs/reeval_queue/parallel.log 2>&1 &
set -u
cd /Users/renhao/git/github/RenQuant
source .venv/bin/activate

is_done() {
  local label="$1"
  local n
  n=$(ls "data/logs/sim_2026-05-15_${label}/equity/" 2>/dev/null | wc -l | tr -d ' ')
  [[ "$n" == "16" ]]
}

ntfy() {
  local title="$1" body="$2" prio="${3:-default}"
  curl -sf -H "Title: $title" -H "Priority: $prio" \
       -d "$body" https://ntfy.sh/renquant >/dev/null 2>&1 || true
}

echo "$(date +%H:%M:%S) — waiting for re_trail015 to hit 16/16…"
while ! is_done re_trail015; do
  n=$(ls "data/logs/sim_2026-05-15_re_trail015/equity/" 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date +%H:%M:%S) — re_trail015 ${n}/16, sleeping 60s"
  sleep 60
done
echo "$(date +%H:%M:%S) — re_trail015 DONE; killing sequential queue"

# Kill the sequential queue (don't kill its in-flight workers — they're already done)
pkill -f 'scripts/run_regime_reeval_queue.sh' 2>/dev/null || true
sleep 2

# Re-run missing re_sdl_n2 Q13 (silent race-loss)
if ! [ -s "data/logs/sim_2026-05-15_re_sdl_n2/equity/Q13.json" ]; then
  echo "$(date +%H:%M:%S) — re-running re_sdl_n2 Q13 (the race-loss)"
  python scripts/run_sim_104.py \
    --strategy-config-name strategy_config.sim_re_sdl_n2.json \
    --start 2025-04-01 --end 2025-07-01 \
    --no-compare --no-persist \
    --equity-json data/logs/sim_2026-05-15_re_sdl_n2/equity/Q13.json \
    > data/logs/sim_2026-05-15_re_sdl_n2/logs/Q13.log 2>&1 &
  Q13_PID=$!
else
  echo "$(date +%H:%M:%S) — re_sdl_n2 Q13 already present, skip"
  Q13_PID=
fi

# Build per-panel runner that uses -P 3 (overriding the /tmp/run_p0_panel.sh -P 6)
PANEL_RUNNER=$(mktemp /tmp/run_panel_P3_XXXX.sh)
cat > "${PANEL_RUNNER}" <<'EOF'
#!/bin/bash
set -u
cd /Users/renhao/git/github/RenQuant
source .venv/bin/activate
LABEL=${1:-}
OUT_DIR="data/logs/sim_2026-05-15_${LABEL}"
mkdir -p "${OUT_DIR}/equity" "${OUT_DIR}/logs"
WINDOWS=(
  "Q01 2022-04-01 2022-07-01" "Q02 2022-07-01 2022-10-01"
  "Q03 2022-10-01 2023-01-01" "Q04 2023-01-01 2023-04-01"
  "Q05 2023-04-01 2023-07-01" "Q06 2023-07-01 2023-10-01"
  "Q07 2023-10-01 2024-01-01" "Q08 2024-01-01 2024-04-01"
  "Q09 2024-04-01 2024-07-01" "Q10 2024-07-01 2024-10-01"
  "Q11 2024-10-01 2025-01-01" "Q12 2025-01-01 2025-04-01"
  "Q13 2025-04-01 2025-07-01" "Q14 2025-07-01 2025-10-01"
  "Q15 2025-10-01 2026-01-01" "Q16 2026-01-01 2026-04-01"
)
run_one() {
  local q="$1" start="$2" end="$3" label="$4"
  local eq_path="data/logs/sim_2026-05-15_${label}/equity/${q}.json"
  if [ -s "${eq_path}" ]; then echo "  ${label} ${q} skip"; return 0; fi
  local cfg
  if [[ "${start}" < "2024-01-01" ]]; then
    cfg="strategy_config.sim_${label}_pre2024.json"
  else
    cfg="strategy_config.sim_${label}.json"
  fi
  python scripts/run_sim_104.py \
    --strategy-config-name "${cfg}" \
    --start "${start}" --end "${end}" \
    --no-compare --no-persist \
    --equity-json "${eq_path}" \
    > "data/logs/sim_2026-05-15_${label}/logs/${q}.log" 2>&1
  echo "  ${label} ${q} done"
}
export -f run_one
printf '%s\n' "${WINDOWS[@]}" | xargs -I{} -P 3 bash -c 'set -- {}; run_one "$1" "$2" "$3" '"${LABEL}"
echo "==> ${LABEL} complete"
EOF
chmod +x "${PANEL_RUNNER}"
echo "$(date +%H:%M:%S) — panel runner: ${PANEL_RUNNER}"

# Wait briefly so the Q13 retry warms SPY cache before parallel launch
# (avoid the same FileNotFoundError race the original queue hit)
if [ -n "${Q13_PID}" ]; then
  echo "$(date +%H:%M:%S) — waiting 90s for Q13 to warm caches"
  sleep 90
fi

# Launch 3 panels in parallel (each with -P 3 internally)
PARALLEL_PANELS=(re_cvar025 re_cvar050 re_kelly_t1_035)
PIDS=()
for label in "${PARALLEL_PANELS[@]}"; do
  if is_done "${label}"; then
    echo "$(date +%H:%M:%S) — ${label} already done, skip"
    continue
  fi
  echo "$(date +%H:%M:%S) — launching ${label} in parallel"
  "${PANEL_RUNNER}" "${label}" > "logs/reeval_queue/${label}_parallel.log" 2>&1 &
  PIDS+=($!)
  # Stagger 30s so SPY cache writes don't race
  sleep 30
done

# Wait for Q13 + parallel panels
[ -n "${Q13_PID}" ] && wait ${Q13_PID}
for pid in "${PIDS[@]}"; do
  wait $pid
done

# Final tally
echo "$(date +%H:%M:%S) — all parallel panels complete:"
for label in re_sdl_n2 re_trail015 "${PARALLEL_PANELS[@]}"; do
  n=$(ls "data/logs/sim_2026-05-15_${label}/equity/" 2>/dev/null | wc -l | tr -d ' ')
  printf "  %s  %s/16\n" "${label}" "${n}"
done

ntfy "RenQuant: parallel reeval panels DONE" "Parallel run for re_cvar025+re_cvar050+re_kelly_t1_035 complete. notify_when_panels_done.sh will auto-analyze." high
