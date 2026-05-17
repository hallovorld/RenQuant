#!/bin/bash
# Dense BEAR/CHOPPY panel runner for A1-v2 + B-track.
#
# Design (per doc/research/2026-05-16-experiment-master-plan.md A1 power
# diagnosis):
#   5/16 overlay A1 fired on only 5/14 windows because most quarterly
#   windows had zero BEAR/CHOPPY bars per HMM. Re-running on dense
#   windows (6-week panels in known volatile zones) lifts n_effective.
#
# Windows (8 × ~6 weeks each, all in zones where 5/16 detected BEAR/CHOPPY):
#   W1 2022-04-01..2022-05-15  (early 2022 BEAR)
#   W2 2022-05-15..2022-07-01  (mid 2022 BEAR)
#   W3 2022-07-01..2022-08-15  (Q3 bear-rally)
#   W4 2022-08-15..2022-10-01  (Sep rate-fear leg)
#   W5 2023-02-15..2023-04-01  (SVB crisis)
#   W6 2023-10-15..2023-12-01  (late-Oct vol)
#   W7 2024-07-15..2024-08-31  (Aug vol spike)
#   W8 2025-01-15..2025-03-01  (DeepSeek + tariff vol)
#
# Three panels in parallel (-P 2 each, 6 workers, 30s stagger):
#   - sim_baseline_2026-05-16 (fresh same-day baseline, REUSE)
#   - sim_overlay_sdl_n2_BC  (A1-v2: σ-SDL overlay on BEAR/CHOPPY)
#   - sim_btrack_cvar025_BC  (B-track: per-regime CVaR via kernel patch)
#
# Idempotent. Auto-runs rigorous analyzer.
set -u
cd /Users/renhao/git/github/RenQuant
source .venv/bin/activate

BATCH="dense_2026-05-16"
mkdir -p logs/reeval_queue data/logs/monitor

PANELS=(baseline_2026-05-16 overlay_sdl_n2_BC btrack_cvar025_BC)

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

# Output directory uses _dense suffix to keep separate from the 5/16
# 16-quarter panels (which lived in data/logs/sim_<label>/).
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
  python scripts/run_sim_104.py \
    --strategy-config-name "${cfg}" \
    --start "${start}" --end "${end}" \
    --no-compare --no-persist \
    --equity-json "${eq_path}" \
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

# Pre-flight: configs exist + validated
echo "=== ${BATCH}: pre-flight ==="
fail=0
for label in "${PANELS[@]}"; do
  cfg="backtesting/renquant_104/strategy_config.sim_${label}.json"
  if [[ ! -f "$cfg" ]]; then
    echo "  ✗ ${cfg} missing"; fail=1
  fi
done
TREATMENTS=(overlay_sdl_n2_BC btrack_cvar025_BC)
for label in "${TREATMENTS[@]}"; do
  if ! python scripts/validate_sim_config_active.py \
       --baseline strategy_config.sim_baseline_hmm.json \
       --candidate "strategy_config.sim_${label}.json" \
       > /tmp/preflight_dense_${label}.log 2>&1; then
    echo "  ✗ ${label} static validator failed"
    tail -5 /tmp/preflight_dense_${label}.log; fail=1
  else
    echo "  ✓ ${label} static-validated"
  fi
done
if (( fail )); then echo "Pre-flight failed."; exit 1; fi

# State file for monitor
STATE="data/logs/monitor/${BATCH}_state.json"
cat > "$STATE" <<EOF
{
  "batch": "${BATCH}",
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "runner_pid": $$,
  "panels": ["baseline_2026-05-16","overlay_sdl_n2_BC","btrack_cvar025_BC"],
  "baseline_dir": "data/logs/sim_baseline_2026-05-16_dense",
  "treatment_dirs": [
    "data/logs/sim_overlay_sdl_n2_BC_dense",
    "data/logs/sim_btrack_cvar025_BC_dense"
  ],
  "is_overlay_batch": true
}
EOF
echo "  wrote ${STATE}"

# Launch
echo
echo "=== launching 3 panels in parallel (-P 2 each, 30s stagger) ==="
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

# Final tally
echo
echo "=== final tally $(date +%H:%M:%S) ==="
for label in "${PANELS[@]}"; do
  n=$(ls "data/logs/sim_${label}_dense/equity/" 2>/dev/null | wc -l | tr -d ' ')
  printf "  %-30s %s/8\n" "${label}" "${n}"
done

# Analyzer (using CORRECT --treatments / --out interface)
echo
echo "=== rigorous analyzer ==="
mkdir -p data/logs/reeval_results
BASE_DIR="data/logs/sim_baseline_2026-05-16_dense"

if ! ./scripts/preflight_analyzer.sh "$BASE_DIR" "data/logs/sim_overlay_sdl_n2_BC_dense" >/dev/null 2>&1; then
  echo "  ✗ baseline freshness check FAILED — analysis aborted"
  curl -sf -H "Title: RenQuant: dense batch BASELINE STALE" -H "Priority: high" \
       -d "${BASE_DIR} stale; analysis aborted" \
       https://ntfy.sh/renquant >/dev/null 2>&1 || true
  exit 2
fi

OUT_MD="data/logs/reeval_results/dense_2026-05-16_rigorous.md"
python scripts/analyze_panels_rigorous.py \
  --baseline "$BASE_DIR" \
  --treatments \
      "data/logs/sim_overlay_sdl_n2_BC_dense" \
      "data/logs/sim_btrack_cvar025_BC_dense" \
  --out "$OUT_MD" 2>&1 | tail -30

# Build per-window NOOP diagnostic (overlay-aware)
python3 - <<'PY'
import json, os, sys
panels = {
  "baseline":   "data/logs/sim_baseline_2026-05-16_dense",
  "sdl_n2_BC":  "data/logs/sim_overlay_sdl_n2_BC_dense",
  "cvar025_BC": "data/logs/sim_btrack_cvar025_BC_dense",
}
windows = sorted(set.intersection(*[
  set(os.listdir(f"{d}/equity")) for d in panels.values()
]))
print()
print(f"{'win':<5} {'base':>9} {'sdl':>9} {'cvar':>9}  {'Δsdl':>7} {'Δcvar':>7}  fire")
for w in windows:
    b = json.load(open(f"{panels['baseline']}/equity/{w}"))
    s = json.load(open(f"{panels['sdl_n2_BC']}/equity/{w}"))
    c = json.load(open(f"{panels['cvar025_BC']}/equity/{w}"))
    fs = "SDL" if s["equity"] != b["equity"] else "noop"
    fc = "CVAR" if c["equity"] != b["equity"] else "noop"
    print(f"{w:<5} {b['apy']*100:>+8.2f}% {s['apy']*100:>+8.2f}% {c['apy']*100:>+8.2f}% "
          f"{(s['apy']-b['apy'])*100:>+6.2f} {(c['apy']-b['apy'])*100:>+6.2f}  {fs}/{fc}")
PY

# ntfy
SUMMARY="✓ dense panel DONE (${BATCH})\nbaseline: ${BASE_DIR}\nreport: ${OUT_MD}"
curl -sf -H "Title: RenQuant: dense batch DONE" -H "Priority: high" \
     -d "${SUMMARY}" https://ntfy.sh/renquant >/dev/null 2>&1 || true
echo
echo "Done."
