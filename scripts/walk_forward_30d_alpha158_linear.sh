#!/usr/bin/env bash
# walk_forward_30d_alpha158_linear.sh
#
# 3-cut walk-forward TEST OF THE USER'S DAILY-RETRAIN DESIGN for
# alpha158_linear. Each cut:
#   1. Retrain alpha158_linear with `--train-end-date <cut_date>` so
#      train data = ALL data up through cut_date.
#   2. Run a 30-day forward sim using the freshly-retrained artifact.
#
# Why 30 days: matches the user's design semantics — the model is
# fresh at sim_start, max staleness during sim is 30 days. Avoids the
# false-baseline of 6-month sims where the last 5mo are 5mo-stale.
#
# Why 3 cuts: variance estimate. If all 3 are positive, design works
# across regimes. If 1 negative + 2 positive, regime-conditional. If
# all negative, the model is structurally broken (no amount of fresh
# training fixes it).
#
# Cut dates chosen to span 12mo of OOS regimes:
#   A: train_end=2024-05-04 → sim 2024-05-05 → 2024-06-04
#   B: train_end=2024-11-04 → sim 2024-11-05 → 2024-12-04
#   C: train_end=2025-05-04 → sim 2025-05-05 → 2025-06-04 (= V7's start window)
#
# Cuts run in parallel (each gets a unique side-config pointing at its
# own per-cut artifact path); no production-file races.
#
# Auto-restore: production artifacts are never touched (each cut writes
# its artifact to /tmp/<cut>/...) so no rollback needed.
#
# Results: /tmp/wf30d_<cut>.json + a final summary table to stdout.

set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
STRATEGY_DIR="$REPO_DIR/backtesting/renquant_104"
PYTHON="/Users/renhao/miniconda3/envs/renquant/bin/python"

cd "$REPO_DIR"
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10

CUTS=(
  "2024-05-04|2024-05-05|2024-06-04"
  "2024-11-04|2024-11-05|2024-12-04"
  "2025-05-04|2025-05-05|2025-06-04"
)

mkdir -p /tmp/wf30d_artifacts

# ── Phase 1: train 3 models in parallel ───────────────────────────────────
echo "=== Phase 1/3: train 3 alpha158_linear models (each train ≤ cut_date) ==="
TRAIN_PIDS=()
for cut in "${CUTS[@]}"; do
    IFS='|' read -r te ss se <<<"$cut"
    out="/tmp/wf30d_artifacts/panel-ltr.${te}.json"
    "$PYTHON" scripts/train_panel_linear.py \
        --train-end-date "$te" \
        --output "$out" \
        > "/tmp/wf30d_train_${te}.log" 2>&1 &
    TRAIN_PIDS+=($!)
    echo "  train cut $te → $out  (PID ${TRAIN_PIDS[-1]})"
done
for pid in "${TRAIN_PIDS[@]}"; do wait "$pid"; done
echo "All 3 training jobs done."
echo ""
for cut in "${CUTS[@]}"; do
    IFS='|' read -r te _ _ <<<"$cut"
    grep -E "DIAGNOSTIC IC" "/tmp/wf30d_train_${te}.log" | head -3 \
        | sed "s/^/  cut $te: /"
done

# ── Phase 2: build per-cut side configs pointing at the cut's artifact ────
echo ""
echo "=== Phase 2/3: build per-cut side configs ==="
for cut in "${CUTS[@]}"; do
    IFS='|' read -r te _ _ <<<"$cut"
    cfg_in="$STRATEGY_DIR/strategy_config.alpha158_linear.json"
    cfg_out="$STRATEGY_DIR/strategy_config.wf30d_${te}.json"
    artifact="/tmp/wf30d_artifacts/panel-ltr.${te}.json"
    # Replace artifact_path with the per-cut artifact
    "$PYTHON" -c "
import json, sys
cfg = json.load(open('$cfg_in'))
cfg['panel_ltr']['artifact_path'] = '$artifact'
cfg.setdefault('ranking', {}).setdefault('panel_scoring', {})
cfg['ranking']['panel_scoring']['kind'] = 'panel_linear'
json.dump(cfg, open('$cfg_out', 'w'), indent=2)
print(f'wrote $cfg_out (artifact=$artifact)')
"
done

# ── Phase 3: run 3 sims in parallel (each with own config + artifact) ─────
echo ""
echo "=== Phase 3/3: run 3 × 30-day sims in parallel (--skip-train; just sim phase) ==="
SIM_PIDS=()
for cut in "${CUTS[@]}"; do
    IFS='|' read -r te ss se <<<"$cut"
    out="/tmp/wf30d_${te}.json"
    cfg_name="strategy_config.wf30d_${te}.json"
    "$PYTHON" scripts/holdout_backtest.py \
        --strategy renquant_104 \
        --strategy-config-name "$cfg_name" \
        --train-end "$te" --sim-start "$ss" --sim-end "$se" \
        --skip-train --out "$out" \
        > "/tmp/wf30d_sim_${te}.log" 2>&1 &
    SIM_PIDS+=($!)
    echo "  sim cut $te ($ss → $se)  (PID ${SIM_PIDS[-1]})"
done
for pid in "${SIM_PIDS[@]}"; do wait "$pid"; done
echo "All 3 sims done."
echo ""

# ── Cleanup: remove per-cut side configs (DB has the originals if needed) ─
for cut in "${CUTS[@]}"; do
    IFS='|' read -r te _ _ <<<"$cut"
    rm -f "$STRATEGY_DIR/strategy_config.wf30d_${te}.json"
done

# ── Phase 4: aggregate + decide ───────────────────────────────────────────
echo "=== RESULTS ==="
"$PYTHON" -c "
import json
cuts = [('2024-05', '2024-05-05 → 2024-06-04', '/tmp/wf30d_2024-05-04.json'),
        ('2024-11', '2024-11-05 → 2024-12-04', '/tmp/wf30d_2024-11-04.json'),
        ('2025-05', '2025-05-05 → 2025-06-04', '/tmp/wf30d_2025-05-04.json')]
sharps, apys, dds = [], [], []
print(f'{\"cut\":<8} {\"window\":<32} {\"APY\":>10} {\"Sharpe\":>9} {\"MaxDD\":>8} {\"Win\":>6} {\"trades\":>8}')
print('-' * 80)
for label, win, path in cuts:
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        print(f'{label:<8} {win:<32}  (file missing)')
        continue
    sh = d.get('sharpe_holdout')
    sh_str = f'{sh:+.3f}' if sh is not None else '   NaN'
    apy = d.get('apy_holdout', 0)
    dd = d.get('max_dd_holdout', 0)
    win_rate = d.get('win_rate_holdout', 0) * 100
    trades = d.get('n_buys', 0) + d.get('n_sells', 0)
    print(f'{label:<8} {win:<32} {apy:+9.2f}% {sh_str:>9} {dd:+7.2f}% {win_rate:>5.0f}% {trades:>8}')
    if sh is not None: sharps.append(sh); apys.append(apy); dds.append(dd)
print('-' * 80)
if sharps:
    import statistics
    print(f'mean: Sharpe={statistics.mean(sharps):+.3f}  APY={statistics.mean(apys):+.2f}%  MaxDD={statistics.mean(dds):+.2f}%')
    if len(sharps) >= 2:
        print(f'std:  Sharpe={statistics.stdev(sharps):.3f}  APY={statistics.stdev(apys):.2f}%')
    print()
    decision = 'PROCEED to production' if (
        statistics.mean(sharps) > 0.5 and min(sharps) > -0.3
    ) else (
        'INCONCLUSIVE — variance too high'
        if min(sharps) < -0.3 < max(sharps) else
        'NO-GO — alpha158_linear is structurally broken'
    )
    print(f'==> Decision: {decision}')
"
echo ""
echo "Per-cut JSONs: /tmp/wf30d_<date>.json"
echo "Per-cut sim logs: /tmp/wf30d_sim_<date>.log"
echo "Per-cut artifacts: /tmp/wf30d_artifacts/"
