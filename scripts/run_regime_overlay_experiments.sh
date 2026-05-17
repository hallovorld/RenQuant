#!/bin/bash
# A-track parallel runner: fresh-baseline + A1 + A2 overlay experiments.
#
# Design (per doc/research/2026-05-16-experiment-master-plan.md §2):
#   - 3 panels in parallel, -P 2 internal each → 6 workers, ~10.8 GB RAM
#   - 30s stagger between panel launches (SPY-fetch race avoidance)
#   - Idempotent: skips windows whose equity JSON already exists non-empty
#   - Auto-runs rigorous analyzer when all panels complete
#   - ntfys summary to https://ntfy.sh/renquant
#
# Resumability: kill at any time; re-run resumes from missing windows.
# Laptop sleep: workers pause; re-run on wake.
#
# Usage:
#   nohup ./scripts/run_regime_overlay_experiments.sh \
#     > logs/reeval_queue/overlay_$(date +%Y%m%d).log 2>&1 &
#   echo $! > /tmp/overlay_runner.pid
set -u
cd /Users/renhao/git/github/RenQuant
source .venv/bin/activate

BATCH="overlay_2026-05-16"
mkdir -p "logs/reeval_queue" "data/logs/monitor"

PANELS=(
  baseline_2026-05-16
  overlay_sdl_n2_BC
  overlay_cvar025_control
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
  local eq_path="data/logs/sim_${label}/equity/${q}.json"
  if [ -s "${eq_path}" ]; then
    echo "  ${label} ${q} skip (exists)"; return 0
  fi
  local cfg
  if [[ "${start}" < "2024-01-01" ]]; then
    cfg="strategy_config.sim_${label}_pre2024.json"
  else
    cfg="strategy_config.sim_${label}.json"
  fi
  # baseline configs are exempted from preflight by run_sim_104.py
  # (label starts with "baseline"). Treatment configs go through gate.
  python scripts/run_sim_104.py \
    --strategy-config-name "${cfg}" \
    --start "${start}" --end "${end}" \
    --no-compare --no-persist \
    --equity-json "${eq_path}" \
    > "data/logs/sim_${label}/logs/${q}.log" 2>&1
  local rc=$?
  if (( rc != 0 )); then
    echo "  ${label} ${q} FAILED rc=${rc}"
  else
    echo "  ${label} ${q} done"
  fi
  return $rc
}
export -f run_one

run_panel() {
  local label="$1"
  local out_dir="data/logs/sim_${label}"
  mkdir -p "${out_dir}/equity" "${out_dir}/logs"
  echo "$(date +%H:%M:%S) — panel ${label} STARTING (-P 2)"
  printf '%s\n' "${WINDOWS[@]}" | xargs -I{} -P 2 bash -c "set -- {}; run_one \"\$1\" \"\$2\" \"\$3\" ${label}"
  echo "$(date +%H:%M:%S) — panel ${label} COMPLETE"
}
export -f run_panel

# ── Pre-flight: ensure configs exist + treatments are validated ──
echo "=== ${BATCH}: pre-flight ==="
TREATMENTS=(overlay_sdl_n2_BC overlay_cvar025_control)
fail=0
for label in baseline_2026-05-16 "${TREATMENTS[@]}"; do
  cfg="backtesting/renquant_104/strategy_config.sim_${label}.json"
  if [[ ! -f "$cfg" ]]; then
    echo "  ✗ ${cfg} missing — run scripts/build_regime_overlay_configs.py first"
    fail=1
  fi
done
for label in "${TREATMENTS[@]}"; do
  if ! python scripts/validate_sim_config_active.py \
       --baseline strategy_config.sim_baseline_hmm.json \
       --candidate "strategy_config.sim_${label}.json" \
       > /tmp/preflight_${label}.log 2>&1; then
    echo "  ✗ ${label} static validator failed"
    tail -5 /tmp/preflight_${label}.log
    fail=1
  else
    echo "  ✓ ${label} static-validated"
  fi
done
if (( fail )); then echo "Pre-flight failed. Aborting."; exit 1; fi

# Save batch state so monitor can find us
BATCH_STATE="data/logs/monitor/${BATCH}_state.json"
cat > "${BATCH_STATE}" <<EOF
{
  "batch": "${BATCH}",
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "runner_pid": $$,
  "panels": $(printf '"%s",' "${PANELS[@]}" | sed 's/,$//' | awk '{print "["$0"]"}'),
  "baseline_dir": "data/logs/sim_baseline_2026-05-16",
  "treatment_dirs": [
    "data/logs/sim_overlay_sdl_n2_BC",
    "data/logs/sim_overlay_cvar025_control"
  ]
}
EOF
echo "  wrote ${BATCH_STATE}"

# ── Launch ──
echo
echo "=== launching 3 panels in parallel (-P 2 each, 30s stagger) ==="
PIDS=()
for label in "${PANELS[@]}"; do
  mkdir -p "data/logs/sim_${label}/equity" "data/logs/sim_${label}/logs"
  run_panel "${label}" > "logs/reeval_queue/${label}_2026-05-16.log" 2>&1 &
  PIDS+=($!)
  sleep 30
done
echo "$(date +%H:%M:%S) — all 3 panels launched. PIDs: ${PIDS[*]}"

# ── Wait ──
for pid in "${PIDS[@]}"; do
  wait $pid
done

# ── Final tally ──
echo
echo "=== final tally $(date +%H:%M:%S) ==="
for label in "${PANELS[@]}"; do
  n=$(ls "data/logs/sim_${label}/equity/" 2>/dev/null | wc -l | tr -d ' ')
  printf "  %-30s %s/16\n" "${label}" "${n}"
done

# ── Analyzer baseline gating ──
echo
echo "=== analyzer baseline freshness check ==="
BASELINE_DIR="data/logs/sim_baseline_2026-05-16"
if ./scripts/preflight_analyzer.sh "$BASELINE_DIR" "data/logs/sim_overlay_sdl_n2_BC" >/dev/null 2>&1; then
  echo "  ✓ ${BASELINE_DIR} is fresh — using as analyzer baseline"
else
  echo "  ✗ ${BASELINE_DIR} stale — abort analysis (artifacts refit during run?)"
  curl -sf -H "Title: RenQuant: overlay batch baseline STALE" \
       -H "Priority: high" \
       -d "Baseline ${BASELINE_DIR} failed preflight after batch completion. Manual investigation required." \
       https://ntfy.sh/renquant >/dev/null 2>&1 || true
  exit 2
fi

# ── Run rigorous analyzer ──
echo
echo "=== running rigorous analyzer (bootstrap + DSR + PBO) ==="
mkdir -p data/logs/reeval_results
SUMMARY="✓ A-track overlay batch DONE (${BATCH})\nbaseline: ${BASELINE_DIR}\n\n"
for label in "${TREATMENTS[@]}"; do
  out_json="data/logs/reeval_results/${label}_rigorous.json"
  out_txt="data/logs/reeval_results/${label}_rigorous.txt"
  python scripts/analyze_panels_rigorous.py \
    --baseline "${BASELINE_DIR}" \
    --treatment "data/logs/sim_${label}" \
    --label "${label}" \
    --json "$out_json" \
    > "$out_txt" 2>&1 || true
  if [[ -s "$out_json" ]]; then
    verdict=$(python3 -c "import json;d=json.load(open('$out_json'));print(d.get('verdict','?'))" 2>/dev/null || echo "?")
    pooled=$(python3 -c "import json;d=json.load(open('$out_json'));print(f\"{d.get('pooled',{}).get('mean_pp',0):+.2f}\")" 2>/dev/null || echo "?")
    dsr=$(python3 -c "import json;d=json.load(open('$out_json'));print(f\"{d.get('dsr',0):.2f}\")" 2>/dev/null || echo "?")
    pbo=$(python3 -c "import json;d=json.load(open('$out_json'));print(f\"{d.get('pbo_pct',0):.0f}%\")" 2>/dev/null || echo "?")
    SUMMARY="${SUMMARY}  ${label}: ${verdict} (pooled ${pooled}pp, DSR ${dsr}, PBO ${pbo})\n"
  else
    SUMMARY="${SUMMARY}  ${label}: ANALYZER FAILED — see ${out_txt}\n"
  fi
done

printf "%b" "$SUMMARY"
curl -sf -H "Title: RenQuant: overlay batch DONE" -H "Priority: high" \
     -d "$(printf "%b" "$SUMMARY")" https://ntfy.sh/renquant >/dev/null 2>&1 || true

echo
echo "Done. State: ${BATCH_STATE}"
