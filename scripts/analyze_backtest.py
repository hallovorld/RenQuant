#!/usr/bin/env python
"""Render charts and summary stats for a LEAN backtest run.

Usage::

    python scripts/analyze_backtest.py --strategy test_001_nvda
    python scripts/analyze_backtest.py --strategy test_001_nvda --run 2026-03-25_22-32-29
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import common


def load_backtest_result(strategy_dir: Path, run_name: str | None) -> tuple[dict, Path]:
	if run_name is None:
		return common.load_latest_backtest(strategy_dir)

	run_dir = strategy_dir / "backtests" / run_name
	if not run_dir.exists():
		raise FileNotFoundError(f"Backtest run not found: {run_dir}")

	candidates = [f for f in run_dir.glob("[0-9]*.json") if "-" not in f.stem]
	if not candidates:
		raise FileNotFoundError(f"No result JSON found in {run_dir}")
	path = sorted(candidates)[0]
	result = json.loads(path.read_text())
	if isinstance(result, dict):
		result["_result_path"] = str(path)
	return result, path


def build_price_frame(config: dict, result: dict) -> pd.DataFrame:
	algo_cfg = result.get("algorithmConfiguration", {})
	period_start = algo_cfg.get("startDate", config.get("backtest_start", ""))[:10]
	period_end = algo_cfg.get("endDate", config.get("backtest_end", ""))[:10]

	price_df = common.fetch_ohlcv(
		config["stock_symbol"],
		start=period_start,
		end=period_end,
		provider=config["data_src"],
	).copy()
	price_df["buy_signal"] = False
	price_df["sell_signal"] = False
	price_df.index = pd.to_datetime(price_df.index, utc=True)
	return price_df


def build_benchmark(price_df: pd.DataFrame, equity: pd.Series | None) -> pd.Series | None:
	if equity is None or equity.empty:
		return None

	benchmark = price_df["close"].copy()
	benchmark = benchmark.reindex(equity.index, method="ffill")
	return benchmark.dropna()


def save_dashboard(strategy_dir: Path, run_dir: Path, config: dict, result: dict) -> Path:
	price_df = build_price_frame(config, result)
	fig = common.backtest_dashboard(
		price_df=price_df,
		result=result,
		symbol=config["stock_symbol"],
		initial_cash=config.get("initial_cash", 100_000),
	)
	output_path = run_dir / "dashboard.png"
	fig.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	return output_path


def save_normalized_chart(run_dir: Path, price_df: pd.DataFrame, result: dict) -> Path | None:
	equity = common.parse_equity_series(result)
	benchmark = build_benchmark(price_df, equity)
	trades = common.parse_closed_trades(result)
	if equity is None or equity.empty or benchmark is None or benchmark.empty:
		return None

	fig, ax = plt.subplots(figsize=(14, 5))
	common.plot_normalized_performance(
		ax=ax,
		equity=equity,
		benchmark=benchmark,
		trades=trades,
		title="Normalized Performance vs Buy & Hold",
	)
	output_path = run_dir / "normalized-performance.png"
	fig.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	return output_path


def main() -> int:
	parser = argparse.ArgumentParser(description="Analyze a LEAN backtest run")
	parser.add_argument("--strategy", required=True, help="Strategy directory name under backtesting/")
	parser.add_argument("--run", help="Specific backtest run directory name; defaults to latest")
	args = parser.parse_args()

	strategy_dir = REPO_ROOT / "backtesting" / args.strategy
	if not strategy_dir.exists():
		print(f"Error: strategy directory not found: {strategy_dir}", file=sys.stderr)
		return 1

	config = common.load_strategy_config(strategy_dir / "strategy_config.json")
	result, result_path = load_backtest_result(strategy_dir, args.run)
	run_dir = result_path.parent
	price_df = build_price_frame(config, result)
	stats = common.parse_stats(result)

	dashboard_path = save_dashboard(strategy_dir, run_dir, config, result)
	normalized_path = save_normalized_chart(run_dir, price_df, result)
	summary_payload = {
		"strategy": args.strategy,
		"run": run_dir.name,
		"result_path": str(result_path),
		"dashboard_path": str(dashboard_path),
		"normalized_path": str(normalized_path) if normalized_path else None,
		"stats": stats,
	}
	summary_path = run_dir / "analysis-summary.json"
	summary_path.write_text(json.dumps(summary_payload, indent=2))

	print(f"Run               : {run_dir.name}")
	print(f"Result            : {result_path}")
	print(f"Dashboard         : {dashboard_path}")
	if normalized_path is not None:
		print(f"Normalized Chart  : {normalized_path}")
	else:
		print("Normalized Chart  : skipped (no equity series in LEAN result)")
	print(f"Summary JSON      : {summary_path}")
	print()
	for line in common.format_stats_lines(stats):
		print(line)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())