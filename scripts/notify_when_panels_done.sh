#!/bin/bash
# Watch the 7 regime-reeval panels; ntfy when ALL hit 16/16.
#
# Companion to scripts/run_regime_reeval_queue.sh — runs in parallel,
# polls every 5 minutes, posts a single completion alert + auto-runs
# the regime-stratified analyzer for each.
#
# Usage:
#   nohup ./scripts/notify_when_panels_done.sh > logs/reeval_notify.log 2>&1 &

set -u
cd /Users/renhao/git/github/RenQuant
source .venv/bin/activate

PANELS=(
  p0activated_regime_aware
  re_stop007
  re_sdl_n2
  re_trail015
  re_cvar025
  re_cvar050
  re_kelly_t1_035
)
BASELINE_DIR="data/logs/sim_2026-05-14_baseline_hmm"

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

mkdir -p data/logs/reeval_results

# Wait until ALL panels are 16/16
while true; do
  ALL_DONE=1
  REPORT=""
  for label in "${PANELS[@]}"; do
    if is_done "$label"; then
      REPORT="${REPORT}  ✓ ${label}\n"
    else
      n=$(ls "data/logs/sim_2026-05-15_${label}/equity/" 2>/dev/null | wc -l | tr -d ' ')
      REPORT="${REPORT}  · ${label}: ${n}/16\n"
      ALL_DONE=0
    fi
  done
  if (( ALL_DONE == 1 )); then
    echo "$(date +%H:%M:%S) — all 7 panels DONE; running stratified analyzer…"
    break
  fi
  echo "$(date +%H:%M:%S) — progress:"
  printf "%b" "$REPORT"
  sleep 300   # 5 min
done

# Run analyzer for each
SUMMARY="ALL 7 regime-reeval panels DONE\n\n"
for label in "${PANELS[@]}"; do
  treat_dir="data/logs/sim_2026-05-15_${label}"
  out_json="data/logs/reeval_results/${label}.json"
  python scripts/analyze_regime_stratified.py \
    --baseline "$BASELINE_DIR" \
    --treatment "$treat_dir" \
    --label "$label" \
    --json "$out_json" \
    > "data/logs/reeval_results/${label}.txt" 2>&1
  verdict=$(python3 -c "import json;d=json.load(open('$out_json'));print(d.get('verdict','?'))" 2>/dev/null || echo "?")
  pooled_mean=$(python3 -c "import json;d=json.load(open('$out_json'));print(f\"{d['pooled']['mean_pp']:+.2f}\")" 2>/dev/null || echo "?")
  win_regimes=$(python3 -c "import json;d=json.load(open('$out_json'));print(','.join(d.get('win_regimes',[])) or '—')" 2>/dev/null || echo "?")
  SUMMARY="${SUMMARY}  ${label}: ${verdict} (pooled ${pooled_mean}pp; win regimes: ${win_regimes})\n"
done

echo
printf "%b" "$SUMMARY"
ntfy "RenQuant: regime-reeval panels DONE" "$(printf "%b" "$SUMMARY")" high
