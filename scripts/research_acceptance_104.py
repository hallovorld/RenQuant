#!/usr/bin/env python
"""Run renquant_104 research acceptance jobs through a tested pipeline."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))
sys.path.insert(0, str(REPO))

from kernel.pipeline.pp_research_acceptance import (  # noqa: E402
    ResearchAcceptanceContext,
    ResearchAcceptancePipeline,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target",
        action="append",
        default=None,
        help="Acceptance target to run: contracts, true-oos, wf-gate, or all. "
             "May be repeated. Default: contracts.",
    )
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel outer jobs. Default: cpu_count-2 capped by job count.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print/build command graph without executing commands.")
    p.add_argument("--train-cutoff", default="2024-07-01")
    p.add_argument(
        "--artifact-dir",
        default="backtesting/renquant_104/artifacts/walkforward_truly_oos_2024-07-01",
        help="True-OOS cutoff artifact directory. Must remain a walkforward side path.",
    )
    p.add_argument(
        "--eval-json",
        default="backtesting/renquant_104/artifacts/prod/truly_oos_eval/eval_truly_oos.json",
        help="True-OOS eval JSON path consumed by promotion tests.",
    )
    p.add_argument("--skip-retrain", action="store_true",
                   help="For true-oos: evaluate/stamp an existing cutoff artifact.")
    p.add_argument("--artifact", default=None,
                   help="Staging artifact for wf-gate target.")
    p.add_argument("--strategy-config", default="strategy_config.sim_wl200.json")
    p.add_argument("--wf-jobs", type=int, default=3,
                   help="Concurrent walk-forward cuts for scripts/run_wf_gate.py.")
    p.add_argument("--no-strict", action="store_true",
                   help="Do not pass --strict to scripts/run_wf_gate.py.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    targets = tuple(args.target or ["contracts"])
    ctx = ResearchAcceptanceContext(
        repo=REPO,
        targets=targets,
        workers=args.workers,
        dry_run=args.dry_run,
        train_cutoff=args.train_cutoff,
        artifact_dir=Path(args.artifact_dir),
        eval_json_path=Path(args.eval_json),
        artifact=Path(args.artifact) if args.artifact else None,
        strategy_config=args.strategy_config,
        wf_jobs=args.wf_jobs,
        strict=not args.no_strict,
        skip_retrain=args.skip_retrain,
    )
    ResearchAcceptancePipeline(targets).run(ctx)
    if args.dry_run:
        print("Dry-run command graph:")
        for spec in ctx.executed:
            print(f"- {spec.name}: {' '.join(spec.argv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
