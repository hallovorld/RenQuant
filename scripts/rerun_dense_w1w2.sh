#!/bin/bash
# Re-run W1 and W2 of dense_2026-05-16 panel sequentially.
#
# Why: original run (run_dense_panel.sh) launched 6 sims concurrent
# (3 panels × -P 2) at startup; load avg hit 21.85 on 10 cores and
# the macOS resource killer SIGTERM'd the W1/W2 workers (rc=143).
# W3-W8 ran fine because they came later when earlier ones had
# finished. Re-run sequentially (-P 1, one panel at a time) so
# the 6 missing windows complete cleanly.
set -u
cd /Users/renhao/git/github/RenQuant
source .venv/bin/activate

PANELS=(baseline_2026-05-16 overlay_sdl_n2_BC btrack_cvar025_BC)
WINDOWS=(
  "W1 2022-04-01 2022-05-15"
  "W2 2022-05-15 2022-07-01"
)

run_one() {
  local q="$1" start="$2" end="$3" label="$4"
  local eq_path="data/logs/sim_${label}_dense/equity/${q}.json"
  if [ -s "${eq_path}" ]; then echo "  ${label} ${q} already complete"; return 0; fi
  local cfg
  if [[ "${start}" < "2024-01-01" ]]; then
    cfg="strategy_config.sim_${label}_pre2024.json"
  else
    cfg="strategy_config.sim_${label}.json"
  fi
  echo "  $(date +%H:%M:%S) ${label} ${q} starting ..."
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

echo "=== dense W1/W2 re-run (sequential, no parallelism) ==="
for label in "${PANELS[@]}"; do
  for w in "${WINDOWS[@]}"; do
    set -- $w; run_one "$1" "$2" "$3" "${label}"
  done
done

echo
echo "=== final tally $(date +%H:%M:%S) ==="
for label in "${PANELS[@]}"; do
  n=$(ls "data/logs/sim_${label}_dense/equity/" 2>/dev/null | wc -l | tr -d ' ')
  printf "  %-30s %s/8\n" "${label}" "${n}"
done

# Run analyzer if all panels at 8/8
all_done=1
for label in "${PANELS[@]}"; do
  n=$(ls "data/logs/sim_${label}_dense/equity/" 2>/dev/null | wc -l | tr -d ' ')
  if (( n < 8 )); then all_done=0; fi
done
if (( all_done )); then
  echo
  echo "=== rigorous analyzer ==="
  mkdir -p data/logs/reeval_results
  BASE_DIR="data/logs/sim_baseline_2026-05-16_dense"
  OUT_MD="data/logs/reeval_results/dense_2026-05-16_rigorous.md"
  python scripts/analyze_panels_rigorous.py \
    --baseline "$BASE_DIR" \
    --treatments \
        "data/logs/sim_overlay_sdl_n2_BC_dense" \
        "data/logs/sim_btrack_cvar025_BC_dense" \
    --out "$OUT_MD" 2>&1 | tail -30

  # Per-window NOOP diagnostic
  python3 - <<'PY'
import json, os
panels = {
  "baseline":   "data/logs/sim_baseline_2026-05-16_dense",
  "sdl_n2_BC":  "data/logs/sim_overlay_sdl_n2_BC_dense",
  "cvar025_BC": "data/logs/sim_btrack_cvar025_BC_dense",
}
windows = sorted(set.intersection(*[set(os.listdir(f"{d}/equity")) for d in panels.values()]))
print()
print(f"{'win':<5} {'base':>9} {'sdl':>9} {'cvar':>9}  {'Δsdl':>7} {'Δcvar':>7}  fire")
for w in windows:
    b = json.load(open(f"{panels['baseline']}/equity/{w}"))
    s = json.load(open(f"{panels['sdl_n2_BC']}/equity/{w}"))
    c = json.load(open(f"{panels['cvar025_BC']}/equity/{w}"))
    fs = "SDL"  if s["equity"] != b["equity"] else "noop"
    fc = "CVAR" if c["equity"] != b["equity"] else "noop"
    print(f"{w:<5} {b['apy']*100:>+8.2f}% {s['apy']*100:>+8.2f}% {c['apy']*100:>+8.2f}% "
          f"{(s['apy']-b['apy'])*100:>+6.2f} {(c['apy']-b['apy'])*100:>+6.2f}  {fs}/{fc}")
PY

  curl -sf -H "Title: RenQuant: dense rerun DONE 8/8" -H "Priority: high" \
       -d "dense_2026-05-16 complete: ${OUT_MD}" \
       https://ntfy.sh/renquant >/dev/null 2>&1 || true
else
  curl -sf -H "Title: RenQuant: dense rerun incomplete" -H "Priority: high" \
       -d "Some panels still <8/8 after sequential rerun. Check logs." \
       https://ntfy.sh/renquant >/dev/null 2>&1 || true
fi
echo "Done."
