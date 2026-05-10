#!/usr/bin/env bash
# multi_seed_baseline.sh — 5-seed A/A baseline characterization.
#
# 2026-05-09 audit FIX-G (per user "find baseline"): the +6.77% claim was a
# single measurement that didn't reproduce 8h later (+1.97%). Same config
# + artifact + window. Either sim has non-deterministic threads (XGBoost
# multi-thread) or a hidden state mutation. Multi-seed A/A characterizes
# σ_APY and σ_Sharpe so future single-cut claims have a confidence floor.
#
# Strategy:
#   - 5 sequential 27-mo sim runs (no actual seed arg in sim — XGBoost
#     thread-non-determinism IS the noise source we're measuring)
#   - Each run captured to logs/baseline_seeds/run_N.log
#   - Final summary line parsed + aggregated mean ± std
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
LOG_DIR="$REPO_DIR/logs/baseline_seeds"
mkdir -p "$LOG_DIR"

cd "$REPO_DIR"
source .venv/bin/activate
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10

START=$(date +%s)
N_SEEDS=5

for i in $(seq 1 $N_SEEDS); do
    echo "═══ Seed $i / $N_SEEDS — started $(date) ═══"
    LOG="$LOG_DIR/run_${i}.log"
    python scripts/run_sim_104.py --start 2024-01-02 --end 2026-03-28 \
        --strategy-config-name strategy_config.json 2>&1 | tee "$LOG" >/dev/null
    APY=$(grep "APY=" "$LOG" | tail -1 | grep -oE "APY=[+-]?[0-9.]+%" | head -1)
    SHARPE=$(grep "Sharpe=" "$LOG" | tail -1 | grep -oE "Sharpe=[+-][0-9.]+" | head -1)
    echo "  → $APY  $SHARPE  (run_${i}.log)"
done

END=$(date +%s)
echo "═══ All $N_SEEDS seeds done in $(( (END-START)/60 )) min ═══"
echo "═══ Aggregating results ═══"
python - <<'PY'
import re, statistics, glob
apys, sharpes, finals = [], [], []
for f in sorted(glob.glob("logs/baseline_seeds/run_*.log")):
    txt = open(f).read()
    # 'APY +X.YZ%'   or   'APY=+X.YZ%'
    m = re.search(r'APY[=\s]+([+-]?\d+\.\d+)%', txt[-3000:])
    s = re.search(r'Sharpe=([+-]?\d+\.\d+)', txt[-3000:])
    fv = re.search(r'Final value:\s+\$([\d,]+)', txt[-3000:])
    if m: apys.append(float(m.group(1)))
    if s: sharpes.append(float(s.group(1)))
    if fv: finals.append(float(fv.group(1).replace(",", "")))
def fmt(xs, lbl):
    if len(xs) >= 2:
        mu, sd = statistics.mean(xs), statistics.stdev(xs)
        return f"{lbl}: n={len(xs)}  mean={mu:+.4f}  std={sd:.4f}  values={xs}"
    return f"{lbl}: n={len(xs)}  values={xs}"
print(fmt(apys,    "APY (%)"))
print(fmt(sharpes, "Sharpe "))
print(fmt(finals,  "FinalVal $"))
PY
