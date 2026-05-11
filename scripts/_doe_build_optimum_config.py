#!/usr/bin/env python
"""Translate bb_optimum.json (continuous optimum on the response surface)
into a concrete strategy_config JSON consumable by run_sim_104.

Reads:
    data/logs/bb_optimum.json     ← produced by _doe_fit_response_surface.py

Writes:
    backtesting/renquant_104/strategy_config.sim_BB_optimum.json
"""
from __future__ import annotations
import json
from pathlib import Path

REPO  = Path(__file__).resolve().parent.parent
STRAT = REPO / "backtesting" / "renquant_104"

opt = json.loads((REPO / "data" / "logs" / "bb_optimum.json").read_text())
real = opt["real_optimum"]
print("Real optimum knob values:")
for k, v in real.items():
    print(f"  {k:32s} = {v:.4f}")

cfg = json.loads((STRAT / "strategy_config.sim_baseline.json").read_text())
bc = cfg["regime_params"]["BULL_CALM"]
for k in ("stop_loss_pct", "trailing_stop_trigger_pct",
          "trailing_stop_trail_pct", "drawdown_halt_pct"):
    if k in real:
        bc[k] = round(float(real[k]), 4)
# Resume threshold set just below halt
bc["drawdown_resume_pct"] = max(0.10, round(bc["drawdown_halt_pct"] - 0.05, 4))
cfg["_side_config_label"]    = "sim_BB_optimum"
cfg["_doe_origin"]           = "BB-27 response surface optimum (Track A)"
cfg["_doe_predicted_metrics"] = opt.get("predicted_metrics", {})

out = STRAT / "strategy_config.sim_BB_optimum.json"
out.write_text(json.dumps(cfg, indent=2))
print(f"\nWrote {out}")
