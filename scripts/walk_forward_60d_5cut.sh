#!/usr/bin/env bash
# walk_forward_60d_5cut.sh
#
# Proper walk-forward grid per the evaluation standard documented at
# user request 2026-05-07:
#   5 cuts × 60-day forward sim, each cut retrains alpha158_linear with
#   ALL data through train_end (matches user's daily-retrain design).
#
# Cuts cover 18 months of OOS history across regimes:
#   T1 train_end=2024-05-04, sim 2024-05-05 → 2024-07-04 (~44 trading days)
#   T2 train_end=2024-08-04, sim 2024-08-05 → 2024-10-04
#   T3 train_end=2024-11-04, sim 2024-11-05 → 2025-01-04
#   T4 train_end=2025-02-04, sim 2025-02-05 → 2025-04-04
#   T5 train_end=2025-05-04, sim 2025-05-05 → 2025-07-04
#
# Decision rule (per evaluation standard):
#   - Mean Sharpe across 5 cuts > 0.5
#   - Min Sharpe (worst cut)    > -0.3
#   - σ(Sharpe) (5 cuts std)    < 1.0
#   - Max DD any cut            < 15%
#   - Mean per-day IC           > +0.02 (deferred — measured in next phase)
# All must pass for PROCEED to live promotion.
#
# Same architecture as the 30d driver: 3-phase (train parallel → build
# per-cut configs → 5 sims parallel → aggregate). No production files
# touched (each cut writes its artifact to /tmp/wf60d_artifacts/).

set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
STRATEGY_DIR="$REPO_DIR/backtesting/renquant_104"
PYTHON="/Users/renhao/miniconda3/envs/renquant/bin/python"

cd "$REPO_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# 5 parallel sims × 2 threads = 10 (matches M2 Pro 10 cores; avoids
# oversubscription that would slow each individual sim)

CUTS=(
  "2024-05-04|2024-05-05|2024-07-04"
  "2024-08-04|2024-08-05|2024-10-04"
  "2024-11-04|2024-11-05|2025-01-04"
  "2025-02-04|2025-02-05|2025-04-04"
  "2025-05-04|2025-05-05|2025-07-04"
)

mkdir -p /tmp/wf60d_artifacts

# ── Phase 1: train 5 models in parallel ───────────────────────────────────
echo "=== Phase 1/3: train 5 alpha158_linear models (each train ≤ cut_date) ==="
TRAIN_PIDS=()
for cut in "${CUTS[@]}"; do
    IFS='|' read -r te ss se <<<"$cut"
    out="/tmp/wf60d_artifacts/panel-ltr.${te}.json"
    "$PYTHON" scripts/train_panel_linear.py \
        --train-end-date "$te" \
        --output "$out" \
        > "/tmp/wf60d_train_${te}.log" 2>&1 &
    TRAIN_PIDS+=($!)
    last_pid=$!
    echo "  train cut $te → $out  (PID $last_pid)"
done
for pid in "${TRAIN_PIDS[@]}"; do wait "$pid"; done
echo "All 5 training jobs done."
echo ""
for cut in "${CUTS[@]}"; do
    IFS='|' read -r te _ _ <<<"$cut"
    grep -E "DIAGNOSTIC IC" "/tmp/wf60d_train_${te}.log" | head -3 \
        | sed "s/^/  cut $te: /"
done

# ── Phase 2: build per-cut side configs pointing at the cut's artifact ────
echo ""
echo "=== Phase 2/3: build per-cut side configs ==="
for cut in "${CUTS[@]}"; do
    IFS='|' read -r te _ _ <<<"$cut"
    cfg_in="$STRATEGY_DIR/strategy_config.alpha158_linear.json"
    cfg_out="$STRATEGY_DIR/strategy_config.wf60d_${te}.json"
    artifact="/tmp/wf60d_artifacts/panel-ltr.${te}.json"
    "$PYTHON" -c "
import json
cfg = json.load(open('$cfg_in'))
cfg['panel_ltr']['artifact_path'] = '$artifact'
cfg.setdefault('ranking', {}).setdefault('panel_scoring', {})
# CRITICAL: SimAdapter._try_load_panel_scorer reads THIS path, not panel_ltr.
cfg['ranking']['panel_scoring']['artifact_path'] = '$artifact'
cfg['ranking']['panel_scoring']['kind'] = 'panel_linear'
json.dump(cfg, open('$cfg_out', 'w'), indent=2)
print(f'wrote $cfg_out')
"
done

# ── Phase 3: run 5 sims in parallel ───────────────────────────────────────
echo ""
echo "=== Phase 3/3: run 5 × 60-day sims in parallel (--skip-train) ==="
SIM_PIDS=()
for cut in "${CUTS[@]}"; do
    IFS='|' read -r te ss se <<<"$cut"
    out="/tmp/wf60d_${te}.json"
    cfg_name="strategy_config.wf60d_${te}.json"
    "$PYTHON" scripts/holdout_backtest.py \
        --strategy renquant_104 \
        --strategy-config-name "$cfg_name" \
        --train-end "$te" --sim-start "$ss" --sim-end "$se" \
        --skip-train --out "$out" \
        > "/tmp/wf60d_sim_${te}.log" 2>&1 &
    SIM_PIDS+=($!)
    last_pid=$!
    echo "  sim cut $te ($ss → $se)  (PID $last_pid)"
done
for pid in "${SIM_PIDS[@]}"; do wait "$pid"; done
echo "All 5 sims done."
echo ""

# Cleanup per-cut configs
for cut in "${CUTS[@]}"; do
    IFS='|' read -r te _ _ <<<"$cut"
    rm -f "$STRATEGY_DIR/strategy_config.wf60d_${te}.json"
done

# ── Phase 4: aggregate + decide ───────────────────────────────────────────
echo "=== RESULTS ==="
"$PYTHON" -c "
import json, statistics
cuts = [
    ('T1 2024-05', '2024-05-05 → 2024-07-04', '/tmp/wf60d_2024-05-04.json'),
    ('T2 2024-08', '2024-08-05 → 2024-10-04', '/tmp/wf60d_2024-08-04.json'),
    ('T3 2024-11', '2024-11-05 → 2025-01-04', '/tmp/wf60d_2024-11-04.json'),
    ('T4 2025-02', '2025-02-05 → 2025-04-04', '/tmp/wf60d_2025-02-04.json'),
    ('T5 2025-05', '2025-05-05 → 2025-07-04', '/tmp/wf60d_2025-05-04.json'),
]
sharps, apys, dds = [], [], []
print(f'{\"cut\":<12} {\"window\":<32} {\"APY\":>10} {\"Sharpe\":>9} {\"MaxDD\":>8} {\"Win\":>6} {\"trades\":>8}')
print('-' * 84)
for label, win, path in cuts:
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        print(f'{label:<12} {win:<32}  (file missing)')
        continue
    sh = d.get('sharpe_holdout')
    sh_str = f'{sh:+.3f}' if sh is not None else '   NaN'
    apy = d.get('apy_holdout', 0)
    dd = d.get('max_dd_holdout', 0)
    win_rate = d.get('win_rate_holdout', 0) * 100
    trades = d.get('n_buys', 0) + d.get('n_sells', 0)
    print(f'{label:<12} {win:<32} {apy:+9.2f}% {sh_str:>9} {dd:+7.2f}% {win_rate:>5.0f}% {trades:>8}')
    if sh is not None: sharps.append(sh); apys.append(apy); dds.append(dd)
print('-' * 84)
if len(sharps) >= 2:
    mean_sh = statistics.mean(sharps)
    std_sh = statistics.stdev(sharps)
    mean_apy = statistics.mean(apys)
    min_sh = min(sharps)
    max_dd = max(dds)
    print(f'mean: Sharpe={mean_sh:+.3f}  APY={mean_apy:+.2f}%')
    print(f'std:  Sharpe={std_sh:.3f}')
    print(f'min:  Sharpe={min_sh:+.3f}')
    print(f'max:  MaxDD={max_dd:+.2f}%')
    print()
    # Decision rule
    pass_mean    = mean_sh > 0.5
    pass_min     = min_sh > -0.3
    pass_std     = std_sh < 1.0
    pass_dd      = max_dd < 15.0
    print(f'  mean Sharpe > 0.5 ? {\"PASS\" if pass_mean else \"FAIL\"}  ({mean_sh:+.3f})')
    print(f'  min  Sharpe > -0.3? {\"PASS\" if pass_min  else \"FAIL\"}  ({min_sh:+.3f})')
    print(f'  std  Sharpe < 1.0 ? {\"PASS\" if pass_std  else \"FAIL\"}  ({std_sh:.3f})')
    print(f'  max  MaxDD  < 15%?  {\"PASS\" if pass_dd   else \"FAIL\"}  ({max_dd:+.2f}%)')
    if pass_mean and pass_min and pass_std and pass_dd:
        print(f'\\n==> Decision: PROCEED — alpha158_linear + daily-retrain works across regimes.')
    elif pass_mean and pass_dd:
        print(f'\\n==> Decision: CONDITIONAL — mean Sharpe positive but variance high.')
        print(f'    Suggestions: position-size down (qp_min_invested_pct 0.3),')
        print(f'                 or test alpha158 + XGBoost (architecture).')
    else:
        print(f'\\n==> Decision: NO-GO — alpha158_linear is insufficient even with daily retrain.')
        print(f'    Next: try alpha158 + XGBoost (scripts/train_panel_alpha158_xgb.py).')
"
echo ""
echo "Per-cut JSONs: /tmp/wf60d_<date>.json | logs: /tmp/wf60d_sim_<date>.log"
