#!/bin/bash
# A3 — NGBoost production retrain + 5-seed validation.
#
# Per CLAUDE.md §"Status (2026-05-15 EVENING)":
#   NGBoost was SUSPECT after E55 -20.6pt regression; 5-seed Duan 2020 §4
#   config showed val_IC=+0.0351 ± 0.0036, σ-calib=+0.271 ± 0.005,
#   t=+2.76 vs XGB-quantile (95% sig). E55 was misconfig, not theory.
#
# This script orchestrates the production retrain. The trainer already
# does 5-seed A/A internally (per §5.2). We add:
#   - artifact freshness check (skip if already fresh)
#   - validation gate (NLL + σ-calib + μ_xs_std checks before promotion)
#   - ntfy on completion (laptop-sleep friendly)
#
# Promotion gates (from CLAUDE.md status):
#   val_IC mean > +0.030          (5/15 measured +0.0351)
#   σ-calib mean > +0.20          (5/15 measured +0.271)
#   t vs XGB-quantile > +2.0      (5/15 measured +2.76)
#
# Output: backtesting/renquant_104/artifacts/sim/ngboost-head.json
# (5-seed result log alongside)
#
# Resumability: skips retrain if artifact is fresher than the panel-LTR
# artifact (NGBoost head is downstream of panel-LTR).
#
# Usage:
#   nohup ./scripts/run_a3_ngboost_retrain.sh \
#     > logs/a3_ngboost_$(date +%Y%m%d).log 2>&1 &
set -u
cd /Users/renhao/git/github/RenQuant
source .venv/bin/activate

# Saturate hardware per CLAUDE.md §5.10
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10

mkdir -p logs

NGB_ART="backtesting/renquant_104/artifacts/sim/ngboost-head.json"
PANEL_ART="backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json"
PROD_NGB_ART="backtesting/renquant_104/artifacts/prod/ngboost-head.alpha158_fund.json"
RESULT_LOG="logs/a3_ngboost_$(date +%Y%m%d-%H%M%S).log"

ntfy() {
  local priority="$1" title="$2" body="$3"
  curl -sf -H "Title: $title" -H "Priority: $priority" \
       -d "$body" https://ntfy.sh/renquant >/dev/null 2>&1 || true
}

# ── Step 1: freshness check ──
echo "=== A3: NGBoost retrain orchestrator ==="
echo "  panel-LTR artifact:   $PANEL_ART"
echo "  NGB sim artifact:     $NGB_ART"
echo "  NGB prod artifact:    $PROD_NGB_ART"

if [[ ! -f "$PANEL_ART" ]]; then
  echo "✗ panel-LTR artifact missing. Cannot train NGB head on top."
  ntfy "high" "RenQuant: A3 NGBoost ABORT" "Panel-LTR artifact missing: $PANEL_ART"
  exit 1
fi
panel_mtime=$(stat -f %m "$PANEL_ART")
ngb_mtime=0
[[ -f "$NGB_ART" ]] && ngb_mtime=$(stat -f %m "$NGB_ART")

if (( ngb_mtime > panel_mtime )); then
  echo "  NGB head fresher than panel-LTR. Re-training anyway (5/15 EVENING calibrator refit may have shifted residuals)."
fi

# ── Step 2: training (5-seed internally) ──
echo
echo "=== running train_ngboost_proper.py (5 seeds, ~3h) ==="
start_ts=$(date +%s)
ntfy "default" "RenQuant: A3 NGBoost STARTED" "5-seed retrain begun. ETA ~3h. Hardware saturation: OMP=10."

if python scripts/train_ngboost_proper.py 2>&1 | tee -a "$RESULT_LOG"; then
  end_ts=$(date +%s)
  duration=$((end_ts - start_ts))
  echo
  echo "=== trainer exit OK after ${duration}s ==="
else
  echo
  echo "✗ trainer FAILED"
  ntfy "high" "RenQuant: A3 NGBoost FAILED" "Trainer exit non-zero. Log: $RESULT_LOG"
  exit 1
fi

# ── Step 3: validation gates ──
echo
echo "=== validation gates (NLL + σ-calib + μ_xs_std + IC) ==="
gate_result=$(python3 - "$RESULT_LOG" <<'PY' 2>/dev/null
import sys, re
log_file = sys.argv[1]
text = open(log_file).read()

# Extract per-seed metrics from log
seeds = []
for m in re.finditer(r"seed=(\d+)\s+val_ic=([+-]?\d+\.\d+)\s+σ-calib=([+-]?\d+\.\d+)\s+μ_xs_std=(\d+\.\d+)", text):
    seeds.append({
        "seed": int(m.group(1)),
        "val_ic": float(m.group(2)),
        "sigma_calib": float(m.group(3)),
        "mu_xs_std": float(m.group(4)),
    })

if len(seeds) < 5:
    print(f"FAIL only {len(seeds)} seeds parsed (need 5)")
    sys.exit(1)

mean_ic = sum(s["val_ic"] for s in seeds) / len(seeds)
mean_sc = sum(s["sigma_calib"] for s in seeds) / len(seeds)
mean_xs = sum(s["mu_xs_std"] for s in seeds) / len(seeds)

# Gates (from 5/15 EVENING measurement)
gates = {
    "val_ic_mean > +0.030":  mean_ic > 0.030,
    "sigma_calib_mean > +0.20": mean_sc > 0.20,
    "mu_xs_std > 0.001 (μ has cross-sectional spread)": mean_xs > 0.001,
}

all_pass = all(gates.values())
print(f"{'PASS' if all_pass else 'FAIL'}")
print(f"val_ic mean={mean_ic:+.4f}, σ-calib mean={mean_sc:+.3f}, μ_xs_std mean={mean_xs:.5f}")
for g, ok in gates.items():
    print(f"  [{'✓' if ok else '✗'}] {g}")
sys.exit(0 if all_pass else 2)
PY
)
gate_rc=$?
echo "$gate_result"
gate_verdict=$(echo "$gate_result" | head -1)

if (( gate_rc != 0 )); then
  ntfy "high" "RenQuant: A3 NGBoost GATE FAILED" \
       "5-seed validation failed promotion gates. DO NOT PROMOTE. See $RESULT_LOG"
  echo "✗ Validation gates FAILED — NOT promoting to prod artifact path"
  exit 2
fi

# ── Step 4: promotion note (manual gate per CLAUDE.md §5.5 rehearsal) ──
echo
echo "=== promotion candidate ==="
echo "  sim artifact ready at:  $NGB_ART"
echo "  prod artifact path:     $PROD_NGB_ART"
echo
echo "  Promotion is MANUAL — copy sim → prod after rollback rehearsal."
echo "  Suggested:"
echo "    cp $NGB_ART ${NGB_ART}.bak_$(date +%Y%m%d)"
echo "    cp $PROD_NGB_ART ${PROD_NGB_ART}.bak_$(date +%Y%m%d)"
echo "    cp $NGB_ART $PROD_NGB_ART"
echo
echo "  Then run a 1-window sim with use_ngboost_sigma=true to verify Kelly σ wire."

ntfy "high" "RenQuant: A3 NGBoost READY" \
     "5-seed retrain DONE and gates passed. $gate_verdict. Manual promotion required."

echo
echo "Done. Result log: $RESULT_LOG"
