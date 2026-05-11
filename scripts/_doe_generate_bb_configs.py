#!/usr/bin/env python
"""Generate 27-run Box-Behnken sim configs for the stop-loss optimization sweep.

Per CLAUDE.md §5.14: Box-Behnken Response Surface Design at 3 levels for
4 BULL_CALM stop-loss knobs. Bounds chosen from baseline distribution
analysis (data/logs/baseline_distributions.json, §5.14.5).

Knobs (4D BB → 27 runs = 24 design points + 3 center replicates):
  K1 = stop_loss_pct                 levels [0.10, 0.15, 0.20]
  K2 = trailing_stop_trigger_pct     levels [0.12, 0.20, 0.30]
  K3 = trailing_stop_trail_pct       levels [0.10, 0.18, 0.25]
  K4 = drawdown_halt_pct             levels [0.20, 0.27, 0.35]

Outputs:
  backtesting/renquant_104/strategy_config.sim_BB_NN.json  (27 files)
  data/logs/bb_design_matrix.csv  (knob values + run id)
"""
from __future__ import annotations
import csv, json
from pathlib import Path
from pyDOE2 import bbdesign

REPO  = Path(__file__).resolve().parent.parent
STRAT = REPO / "backtesting" / "renquant_104"
BASE  = json.loads((STRAT / "strategy_config.sim_baseline.json").read_text())

# Knob levels — derived from baseline distributions per §5.14.5.
# (low, mid=golden, high)
LEVELS = {
    "stop_loss_pct":              (0.10, 0.15, 0.20),
    "trailing_stop_trigger_pct":  (0.12, 0.20, 0.30),
    "trailing_stop_trail_pct":    (0.10, 0.18, 0.25),
    "drawdown_halt_pct":          (0.20, 0.27, 0.35),
}
KNOBS = list(LEVELS.keys())

def coded_to_real(coded: float, levels: tuple) -> float:
    if coded == -1.0: return levels[0]
    if coded ==  0.0: return levels[1]
    if coded == +1.0: return levels[2]
    raise ValueError(f"unexpected coded value {coded}")

# Box-Behnken 4-knob, 3 center replicates → 27 runs.
design = bbdesign(4, center=3)
print(f"BB design shape: {design.shape}")

# CSV record for downstream response-surface fitting
csv_rows = [["run_id", "coded_K1", "coded_K2", "coded_K3", "coded_K4"] + KNOBS + ["config_filename"]]

for i, row in enumerate(design):
    knob_vals = {KNOBS[j]: coded_to_real(row[j], LEVELS[KNOBS[j]]) for j in range(4)}
    cfg = json.loads(json.dumps(BASE))   # deep copy
    bc = cfg["regime_params"]["BULL_CALM"]
    for k, v in knob_vals.items():
        bc[k] = v
    # Auto-set drawdown_resume_pct just below halt_pct (no hysteresis baseline)
    bc["drawdown_resume_pct"] = max(0.10, knob_vals["drawdown_halt_pct"] - 0.05)
    label = f"BB_{i:02d}"
    cfg["_side_config_label"] = f"sim_{label}"
    cfg["_doe_design_row"]    = {KNOBS[j]: int(row[j]) for j in range(4)}
    out = STRAT / f"strategy_config.sim_{label}.json"
    out.write_text(json.dumps(cfg, indent=2))
    csv_rows.append([i, *[int(v) for v in row], *[knob_vals[k] for k in KNOBS], str(out.name)])
    print(f"  run {i:02d}: K1={knob_vals[KNOBS[0]]:.2f} K2={knob_vals[KNOBS[1]]:.2f} "
          f"K3={knob_vals[KNOBS[2]]:.2f} K4={knob_vals[KNOBS[3]]:.2f}")

# Persist matrix
csv_out = REPO / "data" / "logs" / "bb_design_matrix.csv"
csv_out.parent.mkdir(parents=True, exist_ok=True)
with open(csv_out, "w") as f:
    csv.writer(f).writerows(csv_rows)
print(f"\nDesign matrix → {csv_out}")
print(f"Total runs: {len(design)}")
