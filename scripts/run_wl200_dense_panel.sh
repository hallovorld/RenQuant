#!/bin/bash
# Roadmap P0 #2: wl200 dense A/B panel — 8 windows × 2 panels.
#
# Smoke verdict (2024-06 → 2024-09): wl200 vs baseline
#   APY +4.4pp, Sharpe +0.31, MaxDD -2.1pp, α +7.0pp — every metric improved.
#
# Dense panel scales to 8 windows for Tier 2/3 statistical validation:
#   Tier 2 SCREEN: mean Δ_APY > 0, mean Δ_Sharpe ≥ 0, ≥ 4/8 consistent
#   Tier 3 LIVE:   Tier 2 + DSR > 0.5 OR PBO < 0.5
#
# Reuses the 8 dense windows from scripts/run_dense_panel.sh (5/16 design).
# Reuses sim_baseline_2026-05-16 panel (same as previous A/B baseline).
set -u
cd /Users/renhao/git/github/RenQuant
source .venv/bin/activate

BATCH="wl200_dense_2026-05-18"
mkdir -p logs/reeval_queue data/logs/monitor

# Panels: baseline (wl103) + treatment (wl200)
PANELS=(baseline_2026-05-16 wl200)

WINDOWS=(
  "W1 2022-04-01 2022-05-15"
  "W2 2022-05-15 2022-07-01"
  "W3 2022-07-01 2022-08-15"
  "W4 2022-08-15 2022-10-01"
  "W5 2023-02-15 2023-04-01"
  "W6 2023-10-15 2023-12-01"
  "W7 2024-07-15 2024-08-31"
  "W8 2025-01-15 2025-03-01"
)

run_one() {
  local q="$1" start="$2" end="$3" label="$4"
  local eq_path="data/logs/sim_${label}_dense/equity/${q}.json"
  if [ -s "${eq_path}" ]; then echo "  ${label} ${q} skip"; return 0; fi
  local cfg
  if [[ "${start}" < "2024-01-01" ]]; then
    cfg="strategy_config.sim_${label}_pre2024.json"
  else
    cfg="strategy_config.sim_${label}.json"
  fi
  # --skip-preflight for wl200 (watchlist swap is intentional, not a knob change)
  local extra_flags=""
  if [[ "${label}" == "wl200" ]]; then extra_flags="--skip-preflight"; fi
  python scripts/run_sim_104.py \
    --strategy-config-name "${cfg}" \
    --start "${start}" --end "${end}" \
    --no-compare --no-persist \
    --equity-json "${eq_path}" \
    ${extra_flags} \
    > "data/logs/sim_${label}_dense/logs/${q}.log" 2>&1
  local rc=$?
  if (( rc != 0 )); then echo "  ${label} ${q} FAILED rc=${rc}"; else echo "  ${label} ${q} done"; fi
  return $rc
}
export -f run_one

run_panel() {
  local label="$1"
  local out_dir="data/logs/sim_${label}_dense"
  mkdir -p "${out_dir}/equity" "${out_dir}/logs"
  echo "$(date +%H:%M:%S) — panel ${label} STARTING (-P 2)"
  printf '%s\n' "${WINDOWS[@]}" | xargs -I{} -P 2 bash -c "set -- {}; run_one \"\$1\" \"\$2\" \"\$3\" ${label}"
  echo "$(date +%H:%M:%S) — panel ${label} COMPLETE"
}
export -f run_panel

# Pre-flight: configs exist
echo "=== ${BATCH}: pre-flight ==="
fail=0
for label in "${PANELS[@]}"; do
  cfg="backtesting/renquant_104/strategy_config.sim_${label}.json"
  pre_cfg="backtesting/renquant_104/strategy_config.sim_${label}_pre2024.json"
  if [[ ! -f "$cfg" ]]; then echo "  ✗ ${cfg} missing"; fail=1; fi
  if [[ ! -f "$pre_cfg" ]]; then echo "  ✗ ${pre_cfg} missing"; fail=1; fi
done
if (( fail )); then echo "Pre-flight failed."; exit 1; fi
echo "  ✓ all 4 configs exist"

# State file
STATE="data/logs/monitor/${BATCH}_state.json"
cat > "$STATE" <<EOF
{
  "batch": "${BATCH}",
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "runner_pid": $$,
  "panels": ["baseline_2026-05-16","wl200"],
  "baseline_dir": "data/logs/sim_baseline_2026-05-16_dense",
  "treatment_dirs": [
    "data/logs/sim_wl200_dense"
  ]
}
EOF
echo "  wrote ${STATE}"

# Launch
echo
echo "=== launching 2 panels in parallel (-P 2 each, 30s stagger) ==="
PIDS=()
for label in "${PANELS[@]}"; do
  mkdir -p "data/logs/sim_${label}_dense/equity" "data/logs/sim_${label}_dense/logs"
  run_panel "${label}" > "logs/reeval_queue/${label}_dense.log" 2>&1 &
  PIDS+=($!)
  sleep 30
done
echo "$(date +%H:%M:%S) — launched. PIDs: ${PIDS[*]}"

# Wait
for pid in "${PIDS[@]}"; do wait $pid; done

# Tally
echo
echo "=== final tally $(date +%H:%M:%S) ==="
for label in "${PANELS[@]}"; do
  n=$(ls "data/logs/sim_${label}_dense/equity/" 2>/dev/null | wc -l | tr -d ' ')
  printf "  %-30s %s/8\n" "${label}" "${n}"
done

# Analyzer
echo
echo "=== rigorous analyzer ==="
mkdir -p data/logs/reeval_results
BASE_DIR="data/logs/sim_baseline_2026-05-16_dense"
TREAT_DIR="data/logs/sim_wl200_dense"
OUT_MD="data/logs/reeval_results/wl200_dense_2026-05-18.md"
python scripts/analyze_panels_rigorous.py \
  --baseline "$BASE_DIR" \
  --treatments "$TREAT_DIR" \
  --out "$OUT_MD" 2>&1 | tail -30

# Per-window summary
python3 - <<'PY'
import json, os
panels = {
  "baseline": "data/logs/sim_baseline_2026-05-16_dense",
  "wl200":    "data/logs/sim_wl200_dense",
}
windows = sorted(set.intersection(*[
  set(os.listdir(f"{d}/equity")) for d in panels.values()
]))
print()
print(f"{'win':<5} {'base_APY':>9} {'wl200_APY':>10}  {'ΔAPY':>7}  fire")
for w in windows:
    b = json.load(open(f"{panels['baseline']}/equity/{w}"))
    t = json.load(open(f"{panels['wl200']}/equity/{w}"))
    diff = (t.get('apy', 0) - b.get('apy', 0)) * 100
    fire = "FIRED" if t.get('equity') != b.get('equity') else "noop"
    print(f"{w:<5} {b.get('apy', 0)*100:>+8.2f}% {t.get('apy', 0)*100:>+9.2f}% {diff:>+6.2f}  {fire}")
PY

# ntfy
SUMMARY="✓ wl200 dense panel DONE (${BATCH})\nreport: ${OUT_MD}"
curl -sf -H "Title: RenQuant: wl200 dense DONE" -H "Priority: high" \
     -d "${SUMMARY}" https://ntfy.sh/renquant >/dev/null 2>&1 || true
echo
echo "Done."
