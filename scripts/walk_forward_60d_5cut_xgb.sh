#!/usr/bin/env bash
# walk_forward_60d_5cut_xgb.sh
#
# Same 5-cut × 60-day grid as walk_forward_60d_5cut.sh but uses
# alpha158 + XGBoost (rank:pairwise) instead of sklearn LinearRegression.
# Tests E29 resume condition #1: maybe XGB's non-linear interactions
# can find signal in the 158 alpha158 features that the linear model
# missed.
#
# Same evaluation standard:
#   - Mean Sharpe across 5 cuts > 0.5
#   - Min Sharpe (worst cut)    > -0.3
#   - σ(Sharpe) (5 cuts std)    < 1.0
#   - Max DD any cut            < 15%
#   - Mean alpha vs SPY         > 0
#
# Total wall: ~30 min (5 XGB trains parallel ~5min + 5 sims parallel ~25min).

set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
STRATEGY_DIR="$REPO_DIR/backtesting/renquant_104"
PYTHON="/Users/renhao/miniconda3/envs/renquant/bin/python"

cd "$REPO_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

CUTS=(
  "2024-05-04|2024-05-05|2024-07-04"
  "2024-08-04|2024-08-05|2024-10-04"
  "2024-11-04|2024-11-05|2025-01-04"
  "2025-02-04|2025-02-05|2025-04-04"
  "2025-05-04|2025-05-05|2025-07-04"
)

mkdir -p /tmp/wf60d_xgb_artifacts

# ── Phase 1: train 5 XGB models in parallel ───────────────────────────────
echo "=== Phase 1/3: train 5 alpha158_xgb models (each train ≤ cut_date) ==="
TRAIN_PIDS=()
for cut in "${CUTS[@]}"; do
    IFS='|' read -r te ss se <<<"$cut"
    out="/tmp/wf60d_xgb_artifacts/panel-ltr.${te}.json"
    "$PYTHON" scripts/train_panel_alpha158_xgb.py \
        --train-end-date "$te" \
        --output "$out" \
        > "/tmp/wf60d_xgb_train_${te}.log" 2>&1 &
    TRAIN_PIDS+=($!)
    last_pid=$!
    echo "  train cut $te (alpha158+XGB)  PID $last_pid"
done
for pid in "${TRAIN_PIDS[@]}"; do wait "$pid"; done
echo "All 5 XGB training jobs done."
echo ""
for cut in "${CUTS[@]}"; do
    IFS='|' read -r te _ _ <<<"$cut"
    grep -E "best_iter|Test mean IC" "/tmp/wf60d_xgb_train_${te}.log" | tail -2 \
        | sed "s/^/  cut $te: /"
done

# ── Phase 2: build per-cut side configs pointing at the cut's XGB artifact ─
echo ""
echo "=== Phase 2/3: build per-cut side configs ==="
for cut in "${CUTS[@]}"; do
    IFS='|' read -r te _ _ <<<"$cut"
    cfg_in="$STRATEGY_DIR/strategy_config.alpha158_linear.json"
    cfg_out="$STRATEGY_DIR/strategy_config.wf60d_xgb_${te}.json"
    artifact="/tmp/wf60d_xgb_artifacts/panel-ltr.${te}.json"
    "$PYTHON" -c "
import json
cfg = json.load(open('$cfg_in'))
cfg['panel_ltr']['artifact_path'] = '$artifact'
# XGB artifact uses kind=panel_ltr_xgboost (auto-detected by PanelScorer.load)
cfg.setdefault('ranking', {}).setdefault('panel_scoring', {})
cfg['ranking']['panel_scoring']['kind'] = 'panel_ltr_xgboost'
json.dump(cfg, open('$cfg_out', 'w'), indent=2)
"
done
echo "wrote 5 per-cut side configs."

# ── Phase 3: run 5 sims in parallel ───────────────────────────────────────
echo ""
echo "=== Phase 3/3: run 5 × 60-day sims in parallel (--skip-train) ==="
SIM_PIDS=()
for cut in "${CUTS[@]}"; do
    IFS='|' read -r te ss se <<<"$cut"
    out="/tmp/wf60d_xgb_${te}.json"
    cfg_name="strategy_config.wf60d_xgb_${te}.json"
    "$PYTHON" scripts/holdout_backtest.py \
        --strategy renquant_104 \
        --strategy-config-name "$cfg_name" \
        --train-end "$te" --sim-start "$ss" --sim-end "$se" \
        --skip-train --out "$out" \
        > "/tmp/wf60d_xgb_sim_${te}.log" 2>&1 &
    SIM_PIDS+=($!)
    last_pid=$!
    echo "  sim cut $te ($ss → $se)  PID $last_pid"
done
for pid in "${SIM_PIDS[@]}"; do wait "$pid"; done
echo "All 5 sims done."
echo ""

# Cleanup
for cut in "${CUTS[@]}"; do
    IFS='|' read -r te _ _ <<<"$cut"
    rm -f "$STRATEGY_DIR/strategy_config.wf60d_xgb_${te}.json"
done

# ── Phase 4: aggregate + decide (alpha vs SPY) ────────────────────────────
echo "=== RESULTS (alpha158 + XGB) ==="
"$PYTHON" -c "
import json, statistics
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

cuts = [
    ('T1 2024-05', '2024-05-05', '2024-07-04', '/tmp/wf60d_xgb_2024-05-04.json'),
    ('T2 2024-08', '2024-08-05', '2024-10-04', '/tmp/wf60d_xgb_2024-08-04.json'),
    ('T3 2024-11', '2024-11-05', '2025-01-04', '/tmp/wf60d_xgb_2024-11-04.json'),
    ('T4 2025-02', '2025-02-05', '2025-04-04', '/tmp/wf60d_xgb_2025-02-04.json'),
    ('T5 2025-05', '2025-05-05', '2025-07-04', '/tmp/wf60d_xgb_2025-05-04.json'),
]
spy = yf.download('SPY', start='2024-04-01', end='2025-08-01', auto_adjust=True, progress=False)
spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]

print(f'{\"cut\":<12} {\"window\":<28} {\"model%\":>8} {\"SPY%\":>8} {\"alpha\":>8} {\"DD\":>7} {\"sharpe\":>8} {\"trades\":>7}')
print('-' * 92)
alphas, sharps, dds = [], [], []
for label, ss, se, path in cuts:
    try: d = json.load(open(path))
    except FileNotFoundError:
        print(f'{label:<12} (missing)'); continue
    model_ret = d['total_return_holdout']
    model_dd = d['max_dd_holdout']
    sharpe = d.get('sharpe_holdout')
    sh_str = f'{sharpe:+.2f}' if sharpe is not None else 'NaN'
    spy_close = spy.loc[ss:se, 'Close']
    spy_ret = (spy_close.iloc[-1] / spy_close.iloc[0] - 1) * 100
    alpha = model_ret - spy_ret
    trades = d.get('n_buys', 0) + d.get('n_sells', 0)
    print(f'{label:<12} {ss + \" → \" + se:<28} {model_ret:+7.2f}% {spy_ret:+7.2f}% {alpha:+7.2f}% {model_dd:+6.2f}% {sh_str:>8} {trades:>7}')
    alphas.append(alpha); dds.append(model_dd)
    if sharpe is not None: sharps.append(sharpe)

print('-' * 92)
if len(alphas) >= 2:
    mean_alpha = statistics.mean(alphas)
    std_alpha = statistics.stdev(alphas)
    min_alpha = min(alphas)
    max_dd = max(dds)
    n_beat_spy = sum(1 for a in alphas if a > 0)
    print(f'mean alpha vs SPY:  {mean_alpha:+.2f} pts')
    print(f'std alpha:          {std_alpha:.2f} pts')
    print(f'min alpha (worst):  {min_alpha:+.2f} pts')
    print(f'beat SPY:           {n_beat_spy}/{len(alphas)}')
    print(f'max DD:             {max_dd:+.2f}%')
    print()
    pass_alpha   = mean_alpha > 0
    pass_min     = min_alpha > -3.0
    pass_majority = n_beat_spy >= 3
    pass_dd      = max_dd < 15.0
    print(f'  mean alpha > 0     ? {\"PASS\" if pass_alpha    else \"FAIL\"}  ({mean_alpha:+.2f})')
    print(f'  min alpha > -3 pts ? {\"PASS\" if pass_min      else \"FAIL\"}  ({min_alpha:+.2f})')
    print(f'  ≥ 3/5 beat SPY     ? {\"PASS\" if pass_majority else \"FAIL\"}  ({n_beat_spy}/5)')
    print(f'  max DD < 15%       ? {\"PASS\" if pass_dd       else \"FAIL\"}  ({max_dd:+.2f}%)')
    if pass_alpha and pass_min and pass_majority and pass_dd:
        print(f'\\n==> Decision: PROCEED — alpha158 + XGB beats SPY across regimes.')
    elif pass_alpha and pass_dd:
        print(f'\\n==> Decision: CONDITIONAL — mean alpha positive but variance high.')
    else:
        print(f'\\n==> Decision: NO-GO — alpha158 + XGB also fails.')
"
echo ""
echo "Per-cut JSONs: /tmp/wf60d_xgb_<date>.json | logs: /tmp/wf60d_xgb_sim_<date>.log"
