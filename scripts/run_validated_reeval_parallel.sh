#!/bin/bash
# Parallel runner for the 6 validated regime-reeval panels (2026-05-16).
#
# Launches all 6 panels concurrently at -P 2 internal parallelism each.
# 6 × 2 = 12 workers on 10-core M2 Pro. RAM: 12 × 1.8 GB = 22 GB on 32 GB
# host = 10 GB headroom.
#
# SPY-fetch race mitigation: stagger panel launches by 30s so the first
# wave's fetch_ohlcv(SPY) completes and warms data/ohlcv/SPY/1d.parquet
# before subsequent workers try the same write.
#
# Idempotent: each window's equity JSON is skipped if non-zero size
# already exists.
#
# Usage:
#   nohup ./scripts/run_validated_reeval_parallel.sh > logs/reeval_queue/parallel_2026-05-16.log 2>&1 &
set -u
cd /Users/renhao/git/github/RenQuant
source .venv/bin/activate

PANELS=(
  re_stop007
  re_sdl_n2
  re_trail015
  re_cvar025
  re_cvar050
  re_kelly_t1_035
)

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
  local eq_path="data/logs/sim_2026-05-16_${label}/equity/${q}.json"
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
    > "data/logs/sim_2026-05-16_${label}/logs/${q}.log" 2>&1
  echo "  ${label} ${q} done"
}
export -f run_one

run_panel() {
  local label="$1"
  local out_dir="data/logs/sim_2026-05-16_${label}"
  mkdir -p "${out_dir}/equity" "${out_dir}/logs"
  echo "$(date +%H:%M:%S) — panel ${label} STARTING (-P 2)"
  printf '%s\n' "${WINDOWS[@]}" | xargs -I{} -P 2 bash -c "set -- {}; run_one \"\$1\" \"\$2\" \"\$3\" ${label}"
  echo "$(date +%H:%M:%S) — panel ${label} COMPLETE"
}
export -f run_panel

# Sanity: every config exists + passes validator
echo "=== pre-flight: validating all 6 configs ==="
ok=1
for label in "${PANELS[@]}"; do
  if ! python3 scripts/validate_sim_config_active.py \
       --baseline strategy_config.sim_baseline_hmm.json \
       --candidate "strategy_config.sim_${label}.json" \
       > /tmp/validate_${label}.log 2>&1; then
    echo "  ✗ ${label} validator FAILED — aborting launch"
    cat /tmp/validate_${label}.log | tail -5
    ok=0
  else
    echo "  ✓ ${label} validator ACTIVE"
  fi
done
if (( ok == 0 )); then
  echo "Pre-flight failed. NOT launching."
  exit 1
fi

# Launch all 6 panels in parallel with 30s stagger (SPY-fetch race avoidance)
echo
echo "=== launching 6 panels in parallel (-P 2 each, 30s stagger) ==="
PIDS=()
for label in "${PANELS[@]}"; do
  run_panel "${label}" > "logs/reeval_queue/${label}_2026-05-16.log" 2>&1 &
  PIDS+=($!)
  sleep 30
done
echo "$(date +%H:%M:%S) — all 6 panels launched. PIDs: ${PIDS[*]}"

# Wait for completion
for pid in "${PIDS[@]}"; do
  wait $pid
done

echo
echo "=== final tally $(date +%H:%M:%S) ==="
for label in "${PANELS[@]}"; do
  n=$(ls "data/logs/sim_2026-05-16_${label}/equity/" 2>/dev/null | wc -l | tr -d ' ')
  printf "  %s  %s/16\n" "${label}" "${n}"
done

# Auto-run stratified analyzer + ntfy
# IMPORTANT: pick the baseline THE ANALYZER USES carefully — a stale baseline
# generated against an older calibrator/model will silently contaminate every
# "win" verdict (the diff between candidate and baseline is partly artifact-
# refit effect, not knob effect). See `scripts/preflight_analyzer.sh`.
#
# Strategy:
#   1. Try the dedicated baseline (sim_2026-05-14_baseline_hmm) FIRST,
#      gated on preflight_analyzer.sh.
#   2. If preflight blocks (artifacts refit since baseline was generated),
#      fall back to a PROXY BASELINE: the treatment panel whose knob is a
#      static-validated but in-batch no-op (e.g. re_kelly_t1_035, whose
#      tier1 raise has no effect because Kelly is tier-agnostic). This
#      gives clean knob-only deltas at the cost of losing the kelly panel
#      itself as a treatment.
echo
echo "=== analyzer baseline selection ==="
BASELINE_DIR="data/logs/sim_2026-05-14_baseline_hmm"
if ./scripts/preflight_analyzer.sh "$BASELINE_DIR" "data/logs/sim_2026-05-16_${PANELS[0]}" >/dev/null 2>&1; then
  echo "  using ${BASELINE_DIR} (fresh)"
else
  PROXY="data/logs/sim_2026-05-16_re_kelly_t1_035"
  echo "  ${BASELINE_DIR} STALE — falling back to proxy baseline ${PROXY}"
  BASELINE_DIR="${PROXY}"
fi

echo
echo "=== running regime-stratified analyzer (baseline=${BASELINE_DIR}) ==="
mkdir -p data/logs/reeval_results
SUMMARY="✓ 6 validated regime-reeval panels DONE (2026-05-16)\nbaseline: ${BASELINE_DIR}\n\n"
for label in "${PANELS[@]}"; do
  # Skip self-comparison when proxy is the baseline
  if [[ "data/logs/sim_2026-05-16_${label}" == "${BASELINE_DIR}" ]]; then
    SUMMARY="${SUMMARY}  ${label}: PROXY-BASELINE (no analysis vs self)\n"
    continue
  fi
  out_json="data/logs/reeval_results/${label}_validated.json"
  python scripts/analyze_regime_stratified.py \
    --baseline "${BASELINE_DIR}" \
    --treatment "data/logs/sim_2026-05-16_${label}" \
    --label "${label}_validated" \
    --json "$out_json" \
    > "data/logs/reeval_results/${label}_validated.txt" 2>&1
  verdict=$(python3 -c "import json;d=json.load(open('$out_json'));print(d.get('verdict','?'))" 2>/dev/null || echo "?")
  pooled=$(python3 -c "import json;d=json.load(open('$out_json'));print(f\"{d['pooled']['mean_pp']:+.2f}\")" 2>/dev/null || echo "?")
  win_reg=$(python3 -c "import json;d=json.load(open('$out_json'));print(','.join(d.get('win_regimes',[])) or '—')" 2>/dev/null || echo "?")
  SUMMARY="${SUMMARY}  ${label}: ${verdict} (pooled ${pooled}pp; win: ${win_reg})\n"
done
echo
printf "%b" "$SUMMARY"
curl -sf -H "Title: RenQuant: validated reeval DONE" -H "Priority: high" \
     -d "$(printf "%b" "$SUMMARY")" https://ntfy.sh/renquant >/dev/null 2>&1 || true
