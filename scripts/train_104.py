#!/usr/bin/env python
"""End-to-end training driver for renquant_104.

Thin entrypoint — all logic lives in
`backtesting/renquant_104/kernel/pipeline/pp_training_full.py` as a
FullTrainingPipeline (BaselineTournamentJob → PanelTrainingJob →
RecalibrationJob), matching the Job/Task conventions used by the inference
and panel-training pipelines.

Usage::

    python scripts/train_104.py
    python scripts/train_104.py --skip-baseline     # only retrain panel + recalibrate
    python scripts/train_104.py --skip-panel        # only retrain per-ticker tournament
    python scripts/train_104.py --skip-recalibrate  # skip the blend-weight refresh
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train-104")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy",          default="renquant_104")
    p.add_argument("--skip-baseline",     action="store_true")
    p.add_argument("--skip-panel",        action="store_true")
    p.add_argument("--skip-recalibrate",  action="store_true")
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore the training.cadence gate (run even on non-cadence days).",
    )
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    config_path  = strategy_dir / "strategy_config.json"
    if not config_path.exists():
        log.error("Strategy config not found: %s", config_path)
        sys.exit(1)
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    from kernel.pipeline.pp_training_full import (  # noqa: PLC0415
        FullTrainingContext,
        FullTrainingPipeline,
    )

    ctx = FullTrainingContext(
        config=json.loads(config_path.read_text()),
        strategy=args.strategy,
        strategy_dir=strategy_dir,
        skip_baseline=args.skip_baseline,
        skip_panel=args.skip_panel,
        skip_recalibrate=args.skip_recalibrate,
        force_retrain=args.force,
    )
    FullTrainingPipeline().run(ctx)


if __name__ == "__main__":
    main()
