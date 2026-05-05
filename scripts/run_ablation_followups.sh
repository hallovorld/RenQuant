#!/usr/bin/env bash
# After scripts/run_feature_ablation_4way.sh finishes, this script:
#   1. Picks the highest-OOS-IC arm from the four ablation runs
#   2. Requires the lift over D (control) to be ≥ 1bp before promoting —
#      smaller margins are run-to-run noise (~0.6bp σ baseline)
#   3. Runs §5.2 sanity triple (A/A + shuffled-label + time-shift placebo)
#      on the winner — all three must clear
#   4. Runs B2 hold-out sim on the winner artifact for APY/Sharpe
#
# Usage:
#   bash scripts/run_ablation_followups.sh
set -euo pipefail

export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export VECLIB_MAXIMUM_THREADS=10
export NUMEXPR_NUM_THREADS=10

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate renquant

# ── 1. Pick winner ───────────────────────────────────────────────────────
read -r WINNER WIN_IC CTRL_IC <<EOF
$(python3 - <<'PYEOF'
import sqlite3
arms = ["ablation_A_drop8", "ablation_B_add2", "ablation_C_ultra", "ablation_D_control"]
conn = sqlite3.connect("data/runs.db")
results = {}
for arm in arms:
    row = conn.execute(
        "SELECT oos_mean_ic FROM training_runs "
        "WHERE artifact_type='panel-ltr' AND artifact_path LIKE ? "
        "ORDER BY run_date DESC LIMIT 1",
        (f"%{arm}%",),
    ).fetchone()
    if row:
        results[arm] = row[0]

if not results:
    raise SystemExit("ERROR: no ablation results in DB — did training finish?")
if "ablation_D_control" not in results:
    raise SystemExit("ERROR: control arm missing — promote logic needs a baseline")

winner = max(results, key=results.get)
print(f"{winner} {results[winner]:.6f} {results['ablation_D_control']:.6f}")
PYEOF
)
EOF

LIFT=$(python3 -c "print(f'{($WIN_IC - $CTRL_IC) * 10000:.2f}')")
echo "Winner: $WINNER  oos_ic=$WIN_IC  control=$CTRL_IC  lift=${LIFT}bp"

# Run-to-run σ on this panel is ~0.6bp; require ≥1bp lift to promote.
PROMOTE=$(python3 -c "print('YES' if ($WIN_IC - $CTRL_IC) >= 0.0001 else 'NO')")
if [ "$PROMOTE" != "YES" ]; then
    echo "Lift below promotion threshold (1bp) — not running §5.2 / B2."
    exit 0
fi

# ── 2. §5.2 sanity triple ────────────────────────────────────────────────
echo
echo "============================================================"
echo "Running §5.2 sanity triple on $WINNER"
echo "============================================================"
SANITY_LOG="/tmp/${WINNER}_sanity.log"
# 2026-05-04 P0 fix: pre-fix used `|| true` here, which silently swallowed
# any failure from run_sanity_checks.py (including ImportError on a
# rename-rotted symbol — three of which slipped past on 2026-05-03). The
# wrapper then proceeded to "B2 hold-out" claiming sanity was passed when
# it had never run. This is the same SILENT-FAILURE class as the buy_floor
# scale mismatch — every "sanity PASS" produced by this wrapper before
# 2026-05-04 should be considered unverified.
#
# New behavior: sanity script's exit code propagates. If it exits non-zero
# OR if any of A/A / shuffle / shift report FAIL, this wrapper STOPS the
# chain immediately. Caller must look at the log and either fix the issue
# OR explicitly bypass via SKIP_SANITY=1.
if [ "${SKIP_SANITY:-0}" != "1" ]; then
    if ! python scripts/run_sanity_checks.py \
            --strategy renquant_104 \
            --strategy-config-name "strategy_config.${WINNER}.json" \
            --test all \
            > "$SANITY_LOG" 2>&1; then
        echo "✗ Sanity script EXITED NON-ZERO. See $SANITY_LOG."
        echo "  followups STOPPED. Fix the script or set SKIP_SANITY=1 to bypass."
        exit 2
    fi

    echo "Sanity log: $SANITY_LOG"
    grep -E "A/A|shuffle|shift|placebo|PASS|FAIL|ic_mean|spearman" "$SANITY_LOG" | tail -30
    if grep -q "FAIL" "$SANITY_LOG"; then
        echo
        echo "✗ Sanity reported one or more FAIL — followups STOPPED."
        echo "  Investigate the failing test before proceeding to B2."
        echo "  Set SKIP_SANITY=1 to bypass (do NOT do this for ship decisions)."
        exit 3
    fi
else
    echo "⚠ SKIP_SANITY=1 — bypassing §5.2 triple. NOT for ship decisions."
fi

# ── 3. B2 hold-out sim ───────────────────────────────────────────────────
echo
echo "============================================================"
echo "Running B2 hold-out sim on $WINNER"
echo "============================================================"
B2_LOG="/tmp/${WINNER}_b2.log"
python scripts/holdout_backtest.py \
    --strategy-config-name "strategy_config.${WINNER}.json" \
    --train-end "${TRAIN_END:-2024-12-31}" \
    --sim-start "${SIM_START:-2025-01-02}" \
    --sim-end   "${SIM_END:-2026-04-30}" \
    --skip-train \
    > "$B2_LOG" 2>&1

echo
echo "=== B2 result ==="
grep -E "apy_holdout|sharpe_holdout|sortino_holdout|calmar_holdout|max_dd_holdout|win_rate_holdout|buys / sells" "$B2_LOG" | tail -10

echo
echo "Followups complete. Promotion decision input:"
echo "  Winner:  $WINNER  (lift=${LIFT}bp over control)"
echo "  Sanity:  see $SANITY_LOG"
echo "  B2 APY:  see $B2_LOG"
