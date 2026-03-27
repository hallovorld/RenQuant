#!/usr/bin/env python
"""Render charts and summary stats for a LEAN backtest run.

Usage::

    python scripts/analyze_backtest.py --strategy renquant_101
    python scripts/analyze_backtest.py --strategy renquant_101 --run 2026-03-25_22-32-29
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

	# Multi-stock strategies use benchmark (SPY) as price reference
	symbol = config.get("stock_symbol") or config.get("benchmark", "SPY")
	provider = config.get("data_src", "yfinance")

	price_df = common.fetch_ohlcv(
		symbol,
		start=period_start,
		end=period_end,
		provider=provider,
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
	symbol = config.get("stock_symbol") or config.get("benchmark", "SPY")
	fig = common.backtest_dashboard(
		price_df=price_df,
		result=result,
		symbol=symbol,
		initial_cash=config.get("initial_cash", 100_000),
	)
	output_path = run_dir / "dashboard.png"
	fig.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	return output_path


def save_normalized_chart(run_dir: Path, price_df: pd.DataFrame, result: dict,
                         is_multi_stock: bool = False) -> Path | None:
	equity = common.parse_equity_series(result)
	benchmark = build_benchmark(price_df, equity)
	trades = common.parse_closed_trades(result)
	if equity is None or equity.empty or benchmark is None or benchmark.empty:
		return None

	if is_multi_stock and not trades.empty and "symbol" in trades.columns:
		# Multi-stock: color trade markers by symbol
		fig, ax = plt.subplots(figsize=(16, 6))
		common.plot_normalized_performance(
			ax=ax,
			equity=equity,
			benchmark=benchmark,
			trades=pd.DataFrame(),
			title="Normalized Performance vs SPY Buy & Hold",
		)
		unique_symbols = sorted(trades["symbol"].unique())
		colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_symbols), 1)))
		symbol_colors = dict(zip(unique_symbols, colors))
		norm_equity = equity / equity.iloc[0]

		for sym in unique_symbols:
			sym_trades = trades[trades["symbol"] == sym]
			entry_times = sym_trades["entry_time"]
			entry_vals = norm_equity.reindex(entry_times, method="ffill").dropna()
			if not entry_vals.empty:
				ax.scatter(entry_vals.index, entry_vals.values, marker="^",
				           color=symbol_colors[sym], s=60, zorder=5, label=f"{sym}")
			exit_times = sym_trades["exit_time"]
			exit_vals = norm_equity.reindex(exit_times, method="ffill").dropna()
			if not exit_vals.empty:
				ax.scatter(exit_vals.index, exit_vals.values, marker="v",
				           color=symbol_colors[sym], s=60, zorder=5)
		ax.legend(fontsize=7, loc="upper left", ncol=3)
	else:
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


def build_trade_table(result: dict) -> pd.DataFrame | None:
	"""Build a detailed trade table from LEAN closed trades."""
	trades = common.parse_closed_trades(result)
	if trades.empty:
		return None

	equity = common.parse_equity_series(result)

	rows = []
	cumulative_pnl = 0.0
	for _, trade in trades.iterrows():
		cumulative_pnl += trade["pnl"]
		hold_days = (trade["exit_time"] - trade["entry_time"]).days
		ret_pct = (trade["exit_price"] / trade["entry_price"] - 1) * 100 if trade["entry_price"] > 0 else 0

		# Get portfolio value at entry/exit from equity series
		entry_val = exit_val = None
		if equity is not None and not equity.empty:
			entry_val = equity.asof(trade["entry_time"])
			exit_val = equity.asof(trade["exit_time"])

		rows.append({
			"Entry Date": trade["entry_time"].strftime("%Y-%m-%d"),
			"Exit Date": trade["exit_time"].strftime("%Y-%m-%d"),
			"Symbol": trade.get("symbol", ""),
			"Direction": trade.get("direction", "Long"),
			"Qty": int(trade.get("quantity", 0)),
			"Entry $": f"{trade['entry_price']:.2f}",
			"Exit $": f"{trade['exit_price']:.2f}",
			"Return %": f"{ret_pct:+.1f}%",
			"P&L": f"${trade['pnl']:+,.0f}",
			"Cum P&L": f"${cumulative_pnl:+,.0f}",
			"Hold Days": hold_days,
			"Portfolio $": f"${exit_val:,.0f}" if exit_val else "-",
		})

	return pd.DataFrame(rows)


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

	is_multi_stock = "watchlist" in config

	dashboard_path = save_dashboard(strategy_dir, run_dir, config, result)
	normalized_path = save_normalized_chart(run_dir, price_df, result, is_multi_stock=is_multi_stock)

	# Build trade detail table
	trade_table = build_trade_table(result)
	trade_table_path = None
	if trade_table is not None and not trade_table.empty:
		trade_table_path = run_dir / "trade-details.csv"
		trade_table.to_csv(trade_table_path, index=False)

	summary_payload = {
		"strategy": args.strategy,
		"run": run_dir.name,
		"result_path": str(result_path),
		"dashboard_path": str(dashboard_path),
		"normalized_path": str(normalized_path) if normalized_path else None,
		"trade_table_path": str(trade_table_path) if trade_table_path else None,
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
	if trade_table_path:
		print(f"Trade Details     : {trade_table_path}")
	print(f"Summary JSON      : {summary_path}")
	print()
	for line in common.format_stats_lines(stats):
		print(line)

	# Print trade table for multi-stock strategies
	if trade_table is not None and not trade_table.empty:
		print()
		print("Trade Details:")
		print("=" * 120)
		with pd.option_context("display.max_rows", None, "display.width", 120, "display.max_columns", None):
			print(trade_table.to_string(index=False))
		print(f"\nTotal trades: {len(trade_table)}")
		if is_multi_stock:
			symbols_traded = trade_table["Symbol"].unique()
			print(f"Symbols traded: {len(symbols_traded)} — {', '.join(sorted(symbols_traded))}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())