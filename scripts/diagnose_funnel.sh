#!/bin/bash
# Single-command funnel diagnosis — runs sim with --skip-train against
# whatever artifacts are currently on disk, then immediately reports the
# funnel histogram.
#
# Usage:
#   bash scripts/diagnose_funnel.sh [LABEL]
#
# Example:
#   bash scripts/diagnose_funnel.sh post-partial-retrain
#
# Output: /tmp/funnel/<label>/{sim.json,sim.log,funnel.txt}
#
# Why a script: the manual sequence is (a) source conda env, (b) run
# holdout_backtest with skip-train + log capture, (c) parse log via
# scripts/funnel_trace.py. Bundling avoids the human-in-the-loop step
# every time we need to verify a fix.

set -euo pipefail

LABEL="${1:-$(date +%H%M%S)}"
OUT="/tmp/funnel/${LABEL}"
mkdir -p "${OUT}"

REPO="/Users/renhao/git/github/RenQuant"
cd "${REPO}"

# 2026-05-04 fix: must activate conda renquant env explicitly. The project
# also has a `.venv/` (Python 3.9, missing pyarrow) which `python` resolves
# to first if conda isn't sourced — sim crashes on parquet import.
# Per memory/feedback_python_env.md: renquant conda env is the one and only.
# shellcheck disable=SC1091
source /Users/renhao/miniconda3/etc/profile.d/conda.sh
conda activate renquant

echo "=== diagnose_funnel.sh: ${LABEL}"
echo "    OUT=${OUT}"
echo

# Sanity: artifacts on disk
echo "Current artifact state:"
python <<PYEOF
"""2026-05-04: prefix-only match (used to be substring 'p in c'), which
false-positived on names like 'mom_12_1_z' (matched 'm_'),
'earnings_surprise_cum_z' (matched 'cum_'… actually didn't, but
'mom_…' matched 'm_'), etc. The intent is to flag NAMES that came
from the intraday/hourly feature builders — those have the
documented prefixes 'm_' (minute), 'vwap_', 'afternoon_drift',
'intraday_realized_vol', 'morning_drift', 'overnight_gap',
'reversal_ratio'. Using `c.startswith(p)` makes that semantics
explicit."""
import json
INTRADAY_PREFIXES = (
    "m_",                # minute aggregates from minute_features.py
    "vwap_",             # hourly_features VWAP premium
    "afternoon_drift",
    "intraday_realized_vol",
    "morning_drift",
    "overnight_gap",
    "reversal_ratio",
)
for name in ("panel-ltr","ngboost-head"):
    try:
        d = json.load(open(f"backtesting/renquant_104/artifacts/{name}.json"))
        feats = d.get('feature_cols', [])
        intra = [c for c in feats
                 if any(c.startswith(p) for p in INTRADAY_PREFIXES)]
        fp = (d.get('config_fingerprint_fields') or {})
        res_flags = (fp.get('training_resolution'),
                      fp.get('hourly_enabled'),
                      fp.get('minute_enabled'))
        print(f"  {name}: trained={d.get('trained_date')}  "
              f"n_features={len(feats)}  intraday_named={len(intra)} "
              f"{intra if intra else ''}  fp_resolution={res_flags}")
    except Exception as e:
        print(f"  {name}: ERROR {e}")
PYEOF
echo

# Run sim
echo "Running sim (--skip-train, ~18 min wallclock)…"
OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 \
  python scripts/holdout_backtest.py --skip-train \
    --output "${OUT}/sim.json" 2>&1 | tee "${OUT}/sim.log" >/dev/null

echo "Sim done."
echo

# Headline metrics
python <<PYEOF
import json
d = json.load(open("${OUT}/sim.json"))
print("Headline metrics:")
for k in ("apy_holdout","sharpe_holdout","sortino_holdout",
          "max_dd_holdout","ann_vol_holdout","total_return_holdout",
          "win_rate_holdout","n_buys","n_sells"):
    v = d.get(k)
    if v is None:
        v = "NaN"
    elif isinstance(v, float):
        v = f"{v:.4f}"
    print(f"  {k:<22} {v}")
PYEOF
echo

# Funnel trace
echo "Funnel histogram:"
python scripts/funnel_trace.py "${OUT}/sim.log" --db data/sim_runs.db \
  > "${OUT}/funnel.txt"
cat "${OUT}/funnel.txt"
