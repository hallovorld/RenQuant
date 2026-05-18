#!/bin/bash
# σ-wire A/B runner: 8-window dense panel, sigma_on only (reuses
# existing sim_baseline_2026-05-16_dense equity as σ-off control).
#
# Parallelism: -P 2 with 60s stagger (W1+W2 both ran fine in earlier
# rerun once we serialized; -P 2 with stagger is the middle ground).
# Earlier OOM came from launching 6 sims concurrent across 3 panels
# at once. Single panel × -P 2 = 2 sims peak, well within 32GB.
set -u
cd /Users/renhao/git/github/RenQuant
source .venv/bin/activate

BATCH="sigma_wire_2026-05-17"
mkdir -p logs/reeval_queue data/logs/monitor

LABEL="sigma_on_2026-05-17"
OUT_DIR="data/logs/sim_${LABEL}_dense"
mkdir -p "${OUT_DIR}/equity" "${OUT_DIR}/logs"

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
  local q="$1" start="$2" end="$3"
  local eq_path="${OUT_DIR}/equity/${q}.json"
  if [ -s "${eq_path}" ]; then echo "  ${q} skip"; return 0; fi
  local cfg
  if [[ "${start}" < "2024-01-01" ]]; then
    cfg="strategy_config.sim_${LABEL}_pre2024.json"
  else
    cfg="strategy_config.sim_${LABEL}.json"
  fi
  echo "  $(date +%H:%M:%S) ${q} starting (${cfg})"
  python scripts/run_sim_104.py \
    --strategy-config-name "${cfg}" \
    --start "${start}" --end "${end}" \
    --no-compare --no-persist \
    --equity-json "${eq_path}" \
    > "${OUT_DIR}/logs/${q}.log" 2>&1
  local rc=$?
  if (( rc != 0 )); then echo "  ${q} FAILED rc=${rc}"; else echo "  ${q} done"; fi
  return $rc
}
export -f run_one
export LABEL OUT_DIR

echo "=== ${BATCH}: pre-flight ==="
CFG="backtesting/renquant_104/strategy_config.sim_${LABEL}.json"
if [[ ! -f "$CFG" ]]; then echo "✗ missing $CFG"; exit 1; fi
echo "  ✓ config present"
echo "  σ wire: enabled=true, score_mode=mu_minus_lambda_sigma, lambda_sigma=1.0"
echo "  NGB artifact: $(md5 backtesting/renquant_104/artifacts/prod/ngboost-head.alpha158_fund.json | awk '{print $NF}')"

STATE="data/logs/monitor/${BATCH}_state.json"
cat > "$STATE" <<EOF
{
  "batch": "${BATCH}",
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "runner_pid": $$,
  "panels": ["${LABEL}"],
  "baseline_dir": "data/logs/sim_baseline_2026-05-16_dense",
  "treatment_dirs": ["${OUT_DIR}"],
  "is_overlay_batch": true
}
EOF

echo
echo "=== launching σ_on 8 windows (-P 2, 60s stagger) ==="
echo "$(date +%H:%M:%S) — start"
# 60s stagger to avoid the 2 sims both hitting heavy panel-load at once
printf '%s\n' "${WINDOWS[@]}" | (
  i=0
  while read line; do
    (set -- $line; run_one "$1" "$2" "$3") &
    i=$((i+1))
    # cap concurrency at 2: when 2 in flight, wait for one to finish
    if (( i % 2 == 0 )); then wait -n; fi
    sleep 60  # stagger between launches
  done
  wait  # drain remaining
)
echo "$(date +%H:%M:%S) — sigma_on panel COMPLETE"

# Tally
echo
echo "=== final tally $(date +%H:%M:%S) ==="
n=$(ls "${OUT_DIR}/equity/" 2>/dev/null | wc -l | tr -d ' ')
printf "  %-30s %s/8\n" "${LABEL}" "${n}"

# Analyzer if at 8/8
if (( n == 8 )); then
  echo
  echo "=== rigorous analyzer ==="
  mkdir -p data/logs/reeval_results
  OUT_MD="data/logs/reeval_results/sigma_wire_2026-05-17_rigorous.md"
  python scripts/analyze_panels_rigorous.py \
    --baseline data/logs/sim_baseline_2026-05-16_dense \
    --treatments "${OUT_DIR}" \
    --out "$OUT_MD" 2>&1 | tail -25

  # Per-window NOOP/fire diagnostic
  python3 - <<PY
import json, os
b_dir = "data/logs/sim_baseline_2026-05-16_dense"
t_dir = "${OUT_DIR}"
windows = sorted(set(os.listdir(f"{b_dir}/equity")) & set(os.listdir(f"{t_dir}/equity")))
print()
print(f"{'win':<5} {'base_apy':>10} {'sigma_on':>10} {'Δσ_on':>8}  fire")
for w in windows:
    b = json.load(open(f"{b_dir}/equity/{w}"))
    t = json.load(open(f"{t_dir}/equity/{w}"))
    fired = "SIG" if b["equity"] != t["equity"] else "noop"
    print(f"{w:<5} {b['apy']*100:>+9.2f}% {t['apy']*100:>+9.2f}% {(t['apy']-b['apy'])*100:>+7.2f}  {fired}")
PY

  curl -sf -H "Title: RenQuant: σ-wire A/B DONE" -H "Priority: high" \
       -d "report: ${OUT_MD}" https://ntfy.sh/renquant >/dev/null 2>&1 || true
else
  curl -sf -H "Title: RenQuant: σ-wire incomplete" -H "Priority: high" \
       -d "Only ${n}/8 windows complete" https://ntfy.sh/renquant >/dev/null 2>&1 || true
fi
echo "Done."
