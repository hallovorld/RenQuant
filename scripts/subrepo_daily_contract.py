#!/usr/bin/env python3
"""Run the pinned-subrepo daily contract through renquant-orchestrator CLI."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / "subrepos.lock.json").read_text())


for entry in LOCK["subrepos"]:
    src = Path(entry["local_path"]) / "src"
    if src.exists():
        sys.path.insert(0, str(src))


from renquant_orchestrator.cli import main as orchestrator_main  # noqa: E402


def _entry(name: str) -> dict:
    for entry in LOCK["subrepos"]:
        if entry["name"] == name:
            return entry
    raise KeyError(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--broker-type", default="paper")
    parser.add_argument("--broker-name", default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    as_of = args.as_of or dt.date.today().isoformat()
    run_id = args.run_id or f"subrepo-daily-contract-{as_of}"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / ".subrepo_runs" / run_id
    )
    strategy_config = (
        Path(_entry("renquant-strategy-104")["local_path"])
        / "configs"
        / "strategy_config.json"
    )
    argv = [
        "daily-contract",
        "--strategy-config",
        str(strategy_config),
        "--output-dir",
        str(output_dir),
        "--run-id",
        run_id,
        "--as-of",
        as_of,
        "--code-commit",
        _entry("renquant-model")["commit"],
        "--broker-type",
        args.broker_type,
    ]
    if args.broker_name:
        argv.extend(["--broker-name", args.broker_name])
    if args.execute:
        argv.append("--execute")
    return orchestrator_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
