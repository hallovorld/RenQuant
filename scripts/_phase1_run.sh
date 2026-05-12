#!/usr/bin/env bash
# Phase 1 config-only experiment screening — 3 features × 6 windows = 18 sims
# Run AFTER CVaR sweep completes (no concurrent contention beyond 10x).
set -uo pipefail

cd /Users/renhao/git/github/RenQuant
mkdir -p data/logs/sim_2026-05-11_Phase1

WINDOWS=(
  "W1:2025-04-01:2025-08-01"
  "W2:2025-08-01:2025-12-01"
  "W3:2024-12-01:2025-12-01"
  "W4:2024-08-01:2024-12-01"
  "W5:2024-04-01:2024-08-01"
  "W6:2024-12-01:2025-04-01"
)
CONFIGS=(sim_E43_voltarget_007 sim_B5_trend_isolated sim_B6_ddkelly_005)

count=0
for w in "${WINDOWS[@]}"; do
  IFS=":" read -r tag start end <<< "$w"
  for cfg in "${CONFIGS[@]}"; do
    log=data/logs/sim_2026-05-11_Phase1/${tag}_${cfg}.log
    .venv/bin/python scripts/run_sim_104.py --start $start --end $end \
        --strategy-config-name strategy_config.${cfg}.json --no-persist --no-compare \
        > $log 2>&1 &
    count=$((count+1))
  done
done
echo "Phase 1 launched: $count sims (3 configs × 6 windows)"
sleep 3
echo "Running: $(ps aux | grep run_sim_104 | grep -v grep | wc -l)"
