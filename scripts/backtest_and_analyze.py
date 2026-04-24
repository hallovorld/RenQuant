#!/usr/bin/env python
"""Run a LEAN backtest, render performance charts, and send notifications.

Examples:

    python scripts/backtest_and_analyze.py --strategy renquant_101
    python scripts/backtest_and_analyze.py --strategy renquant_101 --open
    python scripts/backtest_and_analyze.py --strategy renquant_101 --ntfy other     # custom topic
    python scripts/backtest_and_analyze.py --strategy renquant_101 --silent        # no notifications
    cd backtesting/renquant_101 && python ../../scripts/backtest_and_analyze.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BACKTESTING_DIR = REPO_ROOT / "backtesting"
ANALYZE_SCRIPT = REPO_ROOT / "scripts" / "analyze_backtest.py"
DEFAULT_NTFY_TOPIC = "renquant"


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


def build_summary_text(stats: dict) -> str:
    return (
        f"Return: {stats.get('Total Return', 'N/A')} | "
        f"Sharpe: {stats.get('Sharpe Ratio', 'N/A')} | "
        f"Drawdown: {stats.get('Max Drawdown', 'N/A')}\n"
        f"Trades: {stats.get('Total Trades', 'N/A')} | "
        f"Win Rate: {stats.get('Win Rate', 'N/A')} | "
        f"Equity: {stats.get('End Equity', 'N/A')}"
    )


def notify_local(title: str, body: str) -> None:
    """Send a macOS notification via terminal-notifier (preferred) or osascript."""
    import os
    if os.environ.get("RENQUANT_NO_NOTIFY") == "1":
        return   # tests set this to suppress ntfy + local notifications
    if shutil.which("terminal-notifier"):
        try:
            subprocess.run(
                ["terminal-notifier", "-title", title, "-message", body, "-sound", "Glass"],
                check=True,
            )
            print(f"Local notification sent: {title}")
            return
        except subprocess.CalledProcessError:
            pass

    # Fallback to osascript
    title_esc = title.replace('"', '\\"')
    body_esc = body.replace('"', '\\"')
    script = (
        f'display notification "{body_esc}" '
        f'with title "{title_esc}" '
        f'sound name "Glass"'
    )
    try:
        subprocess.run(["osascript", "-e", script], check=True)
        print(f"Local notification sent (osascript): {title}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: could not send local notification")


def notify_ntfy(title: str, body: str, topic: str) -> None:
    """Send a push notification to iPhone via ntfy.sh."""
    import os
    if os.environ.get("RENQUANT_NO_NOTIFY") == "1":
        return
    url = f"https://ntfy.sh/{topic}"
    req = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        headers={"Title": title},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        print(f"ntfy notification sent to topic: {topic}")
    except Exception as e:
        print(f"Warning: could not send ntfy notification: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LEAN backtest and render charts")
    parser.add_argument("--strategy", help="Strategy name under backtesting/")
    parser.add_argument("--path", help="Absolute or relative path to a strategy directory")
    parser.add_argument("--open", action="store_true", help="Open generated chart images after analysis")
    parser.add_argument("--ntfy", metavar="TOPIC", default=DEFAULT_NTFY_TOPIC,
                        help=f"ntfy.sh topic for iPhone push (default: {DEFAULT_NTFY_TOPIC})")
    parser.add_argument("--silent", action="store_true", help="Disable all notifications")
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

    # Send notifications with backtest summary
    if not args.silent:
        summary_path = run_dir / "analysis-summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            stats = summary.get("stats", {})
            title = f"Backtest: {strategy_name}"
            body = build_summary_text(stats)

            notify_local(title, body)
            notify_ntfy(title, body, args.ntfy)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
