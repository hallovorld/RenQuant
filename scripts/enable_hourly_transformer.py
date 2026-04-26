#!/usr/bin/env python
"""Enable hourly-resolution transformer training (Stage C-3).

What this does:
1. Writes the panel_ltr config flag flip to a side strategy_config:
       backtesting/renquant_104/strategy_config.hourly_transformer.json
2. The hourly transformer config is a copy of strategy_config.json with:
       panel_ltr.backend = "transformer"
       panel_ltr.training_resolution = "hourly"
       panel_ltr.hourly.label_horizon_bars = 7
3. Operator runs `python scripts/train_104.py --strategy-config <path>` to
   train against hourly resolution.

NOT a full sweep — just a single training invocation. Sunday sweep
keeps the production XGBoost backend; hourly transformer is gated to
opt-in until OOS IC clears the +2pt promotion bar.

Usage:
    python scripts/enable_hourly_transformer.py
    python scripts/enable_hourly_transformer.py --label-horizon 14
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--label-horizon", type=int, default=7,
                   help="Forward bars for label horizon (default 7 ≈ 1 session)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the diff but don't write the file")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    base_path = strategy_dir / "strategy_config.json"
    out_path  = strategy_dir / "strategy_config.hourly_transformer.json"

    cfg = json.loads(base_path.read_text())
    panel = cfg.setdefault("panel_ltr", {})
    panel["backend"] = "transformer"
    panel["training_resolution"] = "hourly"
    panel.setdefault("hourly", {})
    panel["hourly"]["label_horizon_bars"] = int(args.label_horizon)
    panel["hourly"].setdefault("cache_dir", "data/intraday")

    if args.dry_run:
        print("=== diff (would write to {}) ===".format(out_path.name))
        print(json.dumps({"panel_ltr": panel}, indent=2))
        return 0

    out_path.write_text(json.dumps(cfg, indent=2))
    print(f"✅ Wrote hourly transformer config → {out_path}")
    print()
    print("To train:")
    print(f"  python scripts/train_104.py --strategy-config "
          f"{out_path.relative_to(REPO_ROOT)}")
    print()
    print(f"Expected panel size growth: ~5-7× (daily ~225k → hourly ~1.5M rows)")
    print(f"Expected wall time: ~30 min (vs ~28 min for daily)")
    print(f"Promotion gate: OOS IC must beat XGBoost daily by ≥+2pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
