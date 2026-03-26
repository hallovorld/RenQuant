#!/usr/bin/env python
"""Run a LEAN backtest, render performance charts, and optionally open them.

Examples:

    python scripts/backtest_and_analyze.py --strategy test_001_nvda
    python scripts/backtest_and_analyze.py --strategy test_001_nvda --open
    cd backtesting/test_001_nvda && python ../../scripts/backtest_and_analyze.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BACKTESTING_DIR = REPO_ROOT / "backtesting"
ANALYZE_SCRIPT = REPO_ROOT / "scripts" / "analyze_backtest.py"


def find_strategy_dir(strategy: str | None, path: str | None) -> Path:
    if strategy:
        strategy_dir = BACKTESTING_DIR / strategy
    elif path:
        strategy_dir = Path(path).resolve()
    else:
        strategy_dir = Path.cwd().resolve()

    if not strategy_dir.exists():
        raise FileNotFoundError(f"Strategy directory not found: {strategy_dir}")
    if not (strategy_dir / "config.json").exists():
        raise FileNotFoundError(f"LEAN config.json not found in: {strategy_dir}")
    if not (strategy_dir / "strategy_config.json").exists():
        raise FileNotFoundError(f"strategy_config.json not found in: {strategy_dir}")

    return strategy_dir


def list_run_dirs(strategy_dir: Path) -> set[str]:
    backtests_dir = strategy_dir / "backtests"
    if not backtests_dir.exists():
        return set()
    return {path.name for path in backtests_dir.iterdir() if path.is_dir()}


def detect_new_run(strategy_dir: Path, before_runs: set[str]) -> str:
    after_runs = list_run_dirs(strategy_dir)
    new_runs = sorted(after_runs - before_runs)
    if new_runs:
        return new_runs[-1]

    existing = sorted(after_runs)
    if not existing:
        raise FileNotFoundError(f"No LEAN backtest runs found under {strategy_dir / 'backtests'}")
    return existing[-1]


def run_command(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def open_file(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=True)
        return
    print(f"Open manually: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LEAN backtest and render charts")
    parser.add_argument("--strategy", help="Strategy name under backtesting/")
    parser.add_argument("--path", help="Absolute or relative path to a strategy directory")
    parser.add_argument("--open", action="store_true", help="Open generated chart images after analysis")
    args = parser.parse_args()

    strategy_dir = find_strategy_dir(args.strategy, args.path)
    strategy_name = strategy_dir.name

    before_runs = list_run_dirs(strategy_dir)
    print(f"Running LEAN backtest in {strategy_dir} ...")
    run_command(["lean", "backtest", "."], cwd=strategy_dir)

    run_name = detect_new_run(strategy_dir, before_runs)
    print(f"Rendering analysis for run {run_name} ...")
    run_command(
        [sys.executable, str(ANALYZE_SCRIPT), "--strategy", strategy_name, "--run", run_name],
        cwd=REPO_ROOT,
    )

    run_dir = strategy_dir / "backtests" / run_name
    dashboard_path = run_dir / "dashboard.png"
    normalized_path = run_dir / "normalized-performance.png"

    print()
    print(f"Backtest run        : {run_dir}")
    print(f"Dashboard chart     : {dashboard_path}")
    print(f"Normalized chart    : {normalized_path}")

    if args.open:
        if dashboard_path.exists():
            open_file(dashboard_path)
        if normalized_path.exists():
            open_file(normalized_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())