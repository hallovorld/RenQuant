#!/bin/bash
# 16-window sim panel runner. Idempotent: skips windows whose equity JSON
# already exists (resume after interrupt).
set -u
cd /Users/renhao/git/github/RenQuant
source .venv/bin/activate

LABEL=${1:-p0activated}
OUT_DIR="data/logs/sim_2026-05-15_${LABEL}"
mkdir -p "${OUT_DIR}/equity" "${OUT_DIR}/logs"

WINDOWS=(
  "Q01 2022-04-01 2022-07-01"
  "Q02 2022-07-01 2022-10-01"
  "Q03 2022-10-01 2023-01-01"
  "Q04 2023-01-01 2023-04-01"
  "Q05 2023-04-01 2023-07-01"
  "Q06 2023-07-01 2023-10-01"
  "Q07 2023-10-01 2024-01-01"
  "Q08 2024-01-01 2024-04-01"
  "Q09 2024-04-01 2024-07-01"
  "Q10 2024-07-01 2024-10-01"
  "Q11 2024-10-01 2025-01-01"
  "Q12 2025-01-01 2025-04-01"
  "Q13 2025-04-01 2025-07-01"
  "Q14 2025-07-01 2025-10-01"
  "Q15 2025-10-01 2026-01-01"
  "Q16 2026-01-01 2026-04-01"
)

run_one() {
  local q="$1" start="$2" end="$3" label="$4"
  local eq_path="data/logs/sim_2026-05-15_${label}/equity/${q}.json"
  # Idempotent: skip if equity JSON exists AND has non-zero size
  if [ -s "${eq_path}" ]; then
    echo "  ${label} ${q} already done — skip"
    return 0
  fi
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

printf '%s\n' "${WINDOWS[@]}" | xargs -I{} -P 6 bash -c 'set -- {}; run_one "$1" "$2" "$3" '"${LABEL}"

echo "==> ${LABEL} panel complete: ${OUT_DIR}"
