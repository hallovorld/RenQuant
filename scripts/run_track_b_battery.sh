#!/usr/bin/env bash
# Track B sanity-triad battery — confirms the 4 BULL_CALM recovery
# features survive the panel build and produce a non-zero real signal
# vs baseline. Does NOT fire a full WF retrain; that lives in
# scripts/train_walkforward_panel.py and is user-fired explicitly.
#
# Source: doc/research/2026-06-03-track-b-fire-instructions.md (path A).
# Audit:  doc/research/2026-06-02-track-b-feature-audit.md (§7.2.1 R2 gate).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
# NOTE: scripts/wf_sanity_paired.py hardcodes both the baseline and
# candidate panel paths in main() (alpha158_291_fundamental_dataset.parquet
# and alpha158_291_fund_regime_dataset.parquet). The precheck below
# only confirms the candidate panel exists and carries the 4 Track B
# columns; if you need a different candidate panel, edit
# scripts/wf_sanity_paired.py main() directly (PANEL_PATH override is
# not threaded into the battery — that is intentional, do not add a
# misleading override here).
PANEL_PATH="data/alpha158_291_fund_regime_dataset.parquet"
LOG_DIR="${LOG_DIR:-logs}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/track_b_triad_${STAMP}.log"
VERDICT_JSON="data/sanity_paired_baseline_vs_regime.json"

mkdir -p "$LOG_DIR"

echo "track_b_battery: starting at $(date)" | tee -a "$LOG"
echo "track_b_battery: panel=$PANEL_PATH" | tee -a "$LOG"
echo "track_b_battery: log=$LOG" | tee -a "$LOG"

# 1. Confirm the 4 Track B columns are present before spending compute.
"$PYTHON" - <<PY 2>&1 | tee -a "$LOG"
import pandas as pd, sys
p = pd.read_parquet("$PANEL_PATH")
missing = [f for f in ("mom_carry_12_1", "beta_dm", "rvar_total",
                       "idio_vol_market") if f not in p.columns]
if missing:
    print(f"track_b_battery: FAIL — missing Track B columns: {missing}")
    print("track_b_battery: rebuild the panel via "
          "scripts/build_alpha158_fund_panel.py --include-features="
          "mom_carry_12_1,beta_dm,rvar_total,idio_vol_market")
    sys.exit(1)
print(f"track_b_battery: panel OK ({len(p):,} rows, "
      f"{len(p.columns)} columns, all 4 Track B columns present)")
PY

# 2. Saturate hardware per CLAUDE.md §6.5 then fire the paired triad.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-14}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-14}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-14}"

echo "track_b_battery: launching wf_sanity_paired.py (OMP=$OMP_NUM_THREADS)" \
    | tee -a "$LOG"
"$PYTHON" scripts/wf_sanity_paired.py 2>&1 | tee -a "$LOG"

# 3. Surface the verdict in a single readable block.
if [ -f "$VERDICT_JSON" ]; then
    echo "" | tee -a "$LOG"
    echo "track_b_battery: SMOKE-ONLY triad complete." | tee -a "$LOG"
    echo "track_b_battery: This is NOT an R2-compliant verdict block —" | tee -a "$LOG"
    echo "  wf_sanity_paired.py uses shift+60d (1× horizon); R2 requires" | tee -a "$LOG"
    echo "  120d (2× horizon). Treat the block below as a smoke/shape" | tee -a "$LOG"
    echo "  check, NOT a promotion gate." | tee -a "$LOG"
    echo "  See doc/research/2026-06-03-track-b-fire-instructions.md §0" | tee -a "$LOG"
    echo "  for the R2 contract." | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    "$PYTHON" - <<PY 2>&1 | tee -a "$LOG"
import json
v = json.load(open("$VERDICT_JSON"))
b, c = v["baseline"], v["regime"]
print(f"  baseline real_signal   = {b['real_signal']:+.4f}")
print(f"  candidate real_signal  = {c['real_signal']:+.4f}")
print(f"  delta                  = {c['real_signal']-b['real_signal']:+.4f}")
print()
print("  shuffle gate (must be within ±2σ of 0):")
print(f"    baseline  shuffle_ic = {b['shuffle_ic']:+.4f}")
print(f"    candidate shuffle_ic = {c['shuffle_ic']:+.4f}")
print()
print("  A/A reproducibility (3 seeds):")
print(f"    baseline  aa_std     = {b['aa_std']:.4f}")
print(f"    candidate aa_std     = {c['aa_std']:.4f}")
PY
fi

echo "" | tee -a "$LOG"
echo "track_b_battery: done at $(date)" | tee -a "$LOG"
echo "track_b_battery: log saved to $LOG" | tee -a "$LOG"
echo "track_b_battery: verdict JSON at $VERDICT_JSON" | tee -a "$LOG"
