#!/usr/bin/env python
"""Render charts and summary stats for a LEAN backtest run.

Usage::

    python scripts/analyze_backtest.py --strategy renquant_103
    python scripts/analyze_backtest.py --strategy renquant_103 --run 2026-03-25_22-32-29
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# kernel/ lives under backtesting/renquant_104/ — same convention as tests/.
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

try:
	from kernel.metrics.perf_summary import compute_perf_triple  # noqa: E402
except ImportError:  # scipy missing in some minimal envs
	compute_perf_triple = None  # type: ignore[assignment]


# ── Ported from common/plotting.py ────────────────────────────────────────────

def load_latest_backtest(strategy_dir: Path) -> tuple[dict, Path]:
	backtests_dir = strategy_dir / "backtests"
	run_dirs = sorted(
		(d for d in backtests_dir.iterdir() if d.is_dir()),
		reverse=True,
	)
	for run_dir in run_dirs:
		candidates = [f for f in run_dir.glob("[0-9]*.json") if "-" not in f.stem]
		if candidates:
			path = candidates[0]
			result = json.loads(path.read_text())
			if isinstance(result, dict):
				result["_result_path"] = str(path)
			return result, path
	raise FileNotFoundError(f"No backtest results found in {backtests_dir}")


def parse_equity_series(result: dict) -> pd.Series | None:
	try:
		values = result["charts"]["Strategy Equity"]["series"]["Equity"]["values"]
		if not values:
			return None
		if isinstance(values[0], dict):
			idx = [datetime.fromtimestamp(v["x"], tz=timezone.utc) for v in values]
			data = [v["y"] for v in values]
		else:
			idx = [datetime.fromtimestamp(v[0], tz=timezone.utc) for v in values]
			data = [v[1] for v in values]
		return pd.Series(data, index=idx, name="equity")
	except (KeyError, TypeError):
		return None


def parse_chart_series(result: dict, chart_name: str, series_name: str) -> pd.Series | None:
	try:
		values = result["charts"][chart_name]["series"][series_name]["values"]
		if not values:
			return None
		if isinstance(values[0], dict):
			idx = [datetime.fromtimestamp(v["x"], tz=timezone.utc) for v in values]
			data = [v["y"] for v in values]
		else:
			idx = [datetime.fromtimestamp(v[0], tz=timezone.utc) for v in values]
			data = [v[1] for v in values]
		return pd.Series(data, index=idx, name=series_name)
	except (KeyError, TypeError):
		return None


def parse_closed_trades(result: dict) -> pd.DataFrame:
	trades = result.get("totalPerformance", {}).get("closedTrades", [])
	if trades:
		rows = []
		for trade in trades:
			symbol = trade.get("symbol")
			if isinstance(symbol, dict):
				symbol = symbol.get("value")
			direction = trade.get("direction", "Long")
			if isinstance(direction, (int, float)):
				direction = "Long" if int(direction) == 0 else "Short"
			rows.append({
				"symbol": symbol or trade.get("symbolValue", ""),
				"direction": direction,
				"quantity": abs(float(trade.get("quantity", 0))),
				"entry_time": pd.to_datetime(trade["entryTime"], utc=True),
				"entry_price": float(trade["entryPrice"]),
				"exit_time": pd.to_datetime(trade["exitTime"], utc=True),
				"exit_price": float(trade["exitPrice"]),
				"pnl": float(trade.get("profitLoss", 0)),
				"fees": float(trade.get("totalFees", 0)),
			})
		return pd.DataFrame(rows)

	result_path_text = result.get("_result_path")
	if not result_path_text:
		return pd.DataFrame()
	order_events_path = Path(result_path_text).with_name(
		f"{Path(result_path_text).stem}-order-events.json")
	if not order_events_path.exists():
		return pd.DataFrame()

	events = json.loads(order_events_path.read_text())
	filled = [e for e in events if e.get("status") == "filled" and float(e.get("fillQuantity", 0)) != 0]
	rows = []
	open_trade = None
	for e in filled:
		direction = str(e.get("direction", "")).lower()
		t = datetime.fromtimestamp(e["time"], tz=timezone.utc)
		qty = abs(float(e.get("fillQuantity", 0)))
		fee = float(e.get("orderFeeAmount", 0) or 0)
		if direction == "buy":
			open_trade = {
				"symbol": e.get("symbolValue") or e.get("symbol", ""),
				"direction": "Long", "quantity": qty,
				"entry_time": t, "entry_price": float(e.get("fillPrice", 0)), "fees": fee,
			}
		elif direction == "sell" and open_trade is not None:
			ep = float(e.get("fillPrice", 0))
			tf = open_trade["fees"] + fee
			rows.append({
				"symbol": open_trade["symbol"], "direction": open_trade["direction"],
				"quantity": open_trade["quantity"], "entry_time": open_trade["entry_time"],
				"entry_price": open_trade["entry_price"], "exit_time": t,
				"exit_price": ep, "pnl": (ep - open_trade["entry_price"]) * open_trade["quantity"] - tf,
				"fees": tf,
			})
			open_trade = None
	return pd.DataFrame(rows)


def parse_stats(result: dict) -> dict:
	ts = result.get("totalPerformance", {}).get("tradeStatistics", {})
	ps = result.get("totalPerformance", {}).get("portfolioStatistics", {})
	rt = result.get("runtimeStatistics", {})
	cfg = result.get("algorithmConfiguration", {})
	state = result.get("state", {})

	def pct(v):
		try: return f"{float(v) * 100:.2f}%"
		except (ValueError, TypeError): return "—"

	def num(v):
		try: return f"{float(v):.4f}"
		except (ValueError, TypeError): return "—"

	start = cfg.get("startDate", "")[:10]
	end = cfg.get("endDate", "")[:10]

	return {
		"Period": f"{start} → {end}",
		"Status": state.get("Status", "—"),
		"Total Orders": int(state.get("OrderCount", 0)),
		"Total Trades": int(ts.get("totalNumberOfTrades", 0)),
		"Winning Trades": int(ts.get("numberOfWinningTrades", 0)),
		"Losing Trades": int(ts.get("numberOfLosingTrades", 0)),
		"Win Rate": pct(ts.get("winRate", 0)),
		"Loss Rate": pct(ts.get("lossRate", 0)),
		"Avg Trade Duration": ts.get("averageTradeDuration", "—"),
		"Profit Factor": num(ts.get("profitFactor", 0)),
		"Sharpe Ratio": num(ts.get("sharpeRatio", 0)),
		"Sortino Ratio": num(ts.get("sortinoRatio", 0)),
		"End Equity": rt.get("Equity", "—"),
		"Net Profit": rt.get("Net Profit", "—"),
		"Total Return": rt.get("Return", "—"),
		"Ann. Return": pct(ps.get("compoundingAnnualReturn", 0)),
		"Max Drawdown": pct(ps.get("drawdown", 0)),
		"Ann. Std Dev": pct(ps.get("annualStandardDeviation", 0)),
		"Total Fees": rt.get("Fees", "—"),
		"Alpha": num(ps.get("alpha", 0)),
		"Beta": num(ps.get("beta", 0)),
		**{k: rt[k] for k in [
			"Policy", "Wash Sale Days", "Min Hold Days",
			"Buy Decisions", "Sell Decisions", "Hold Decisions",
			"Executed Buys", "Executed Sells",
			"Blocked Wash Sales", "Blocked Min Hold",
		] if k in rt},
	}


def compute_significance_metrics(equity: pd.Series | None, n_trials: int) -> dict:
	"""Compute (DSR, PBO=NaN single-series, n_returns) for an equity curve.

	Per CLAUDE.md §5.13.4 every Sharpe must ship with DSR + PBO. PBO needs
	multiple candidate strategies; analyze_backtest.py only sees the headline
	run, so PBO is reported as NaN here — call sites with multi-seed data
	should use kernel.metrics.compute_perf_triple directly.
	"""
	if equity is None or len(equity) < 5 or compute_perf_triple is None:
		return {
			"DSR": "—", "PBO": "—",
			"Returns Sample Size": 0, "DSR n_trials": int(n_trials),
		}
	rets = equity.astype(float).pct_change().dropna().to_numpy()
	if rets.size < 5:
		return {
			"DSR": "—", "PBO": "—",
			"Returns Sample Size": int(rets.size), "DSR n_trials": int(n_trials),
		}
	triple = compute_perf_triple(rets, n_trials=n_trials)
	def _fmt(v):
		try: return f"{float(v):.4f}"
		except (ValueError, TypeError): return "—"
	return {
		"DSR": _fmt(triple["dsr"]),
		"PBO": _fmt(triple["pbo"]),
		"Returns Sample Size": triple["n_returns"],
		"DSR n_trials": triple["n_trials"],
	}


def format_stats_lines(stats: dict) -> list[str]:
	width = max(len(key) for key in stats)
	return [f"{key:<{width}} : {value}" for key, value in stats.items()]


def _style_date_axis(ax):
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
	ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
	ax.figure.autofmt_xdate(rotation=30, ha="right")


def _plot_price_with_signals(ax, price_df: pd.DataFrame, title: str = "Price + Model Signals"):
	ax.plot(price_df.index, price_df["close"], color="#4a90d9", lw=1.2, label="Close")
	buys = price_df[price_df["buy_signal"].astype(bool)]
	sells = price_df[price_df["sell_signal"].astype(bool)]
	if not buys.empty:
		ax.scatter(buys.index, buys["close"], marker="^", color="#2ecc71", s=90, zorder=5,
		           label=f"Buy signal ({len(buys)})")
	if not sells.empty:
		ax.scatter(sells.index, sells["close"], marker="v", color="#e74c3c", s=90, zorder=5,
		           label=f"Sell signal ({len(sells)})")
	ax.set_title(title, fontsize=11)
	ax.set_ylabel("Price (USD)")
	ax.legend(fontsize=9)
	ax.grid(True, alpha=0.3)
	_style_date_axis(ax)


def _plot_trades_on_price(ax, trades: pd.DataFrame):
	if trades.empty:
		return
	entries = trades.set_index("entry_time")["entry_price"]
	exits = trades.set_index("exit_time")["exit_price"]
	ax.scatter(entries.index, entries.values, marker="^", color="#27ae60",
	           s=130, zorder=6, edgecolors="white", lw=0.6, label="LEAN entry")
	ax.scatter(exits.index, exits.values, marker="v", color="#c0392b",
	           s=130, zorder=6, edgecolors="white", lw=0.6, label="LEAN exit")
	for _, trade in trades.iterrows():
		color = "#27ae60" if trade["pnl"] >= 0 else "#c0392b"
		ax.plot([trade["entry_time"], trade["exit_time"]],
		        [trade["entry_price"], trade["exit_price"]],
		        color=color, lw=0.9, alpha=0.5, zorder=4)
	ax.legend(fontsize=9)


def _plot_equity_curve(ax, equity: pd.Series, initial_cash: float = 100_000):
	ax.plot(equity.index, equity.values, color="#4a90d9", lw=1.5, label="Portfolio equity")
	ax.axhline(initial_cash, color="#aaaaaa", lw=0.8, linestyle="--", label="Initial cash")
	ax.fill_between(equity.index, initial_cash, equity.values,
	                where=(equity.values >= initial_cash), alpha=0.15, color="#2ecc71")
	ax.fill_between(equity.index, initial_cash, equity.values,
	                where=(equity.values < initial_cash), alpha=0.15, color="#e74c3c")
	ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
	ax.set_title("Equity Curve", fontsize=11)
	ax.set_ylabel("Portfolio Value (USD)")
	ax.legend(fontsize=9)
	ax.grid(True, alpha=0.3)
	_style_date_axis(ax)


def _plot_drawdown(ax, equity: pd.Series):
	rolling_max = equity.cummax()
	drawdown = (equity - rolling_max) / rolling_max * 100
	ax.fill_between(drawdown.index, drawdown.values, 0, color="#e74c3c", alpha=0.4)
	ax.plot(drawdown.index, drawdown.values, color="#e74c3c", lw=0.8)
	max_dd = drawdown.min()
	ax.axhline(max_dd, color="#c0392b", lw=0.8, linestyle="--",
	           label=f"Max drawdown: {max_dd:.2f}%")
	ax.set_title("Drawdown", fontsize=11)
	ax.set_ylabel("Drawdown (%)")
	ax.legend(fontsize=9)
	ax.grid(True, alpha=0.3)
	_style_date_axis(ax)


def _plot_stats_table(ax, stats: dict):
	ax.axis("off")
	rows = [(k, str(v)) for k, v in stats.items()]
	midpoint = int(np.ceil(len(rows) / 2))
	for table, bbox, data in [
		(ax.table(cellText=rows[:midpoint], colLabels=["Metric", "Value"],
		          cellLoc="left", bbox=[0.00, 0.0, 0.48, 1.0]), None, rows[:midpoint]),
		(ax.table(cellText=rows[midpoint:], colLabels=["Metric", "Value"],
		          cellLoc="left", bbox=[0.52, 0.0, 0.48, 1.0]), None, rows[midpoint:]),
	]:
		table.auto_set_font_size(False)
		table.set_fontsize(8.5)
		table.scale(1, 1.3)
		for (r, c), cell in table.get_celld().items():
			if r == 0:
				cell.set_facecolor("#2c3e50")
				cell.set_text_props(color="white", fontweight="bold")
			elif r % 2 == 0:
				cell.set_facecolor("#f7f7f7")
			cell.set_edgecolor("#dddddd")
	ax.set_title("Performance Statistics", fontsize=11, pad=16)


def _no_data_panel(ax, title: str, message: str = "No data available"):
	ax.text(0.5, 0.5, message, ha="center", va="center",
	        transform=ax.transAxes, color="#888888", fontsize=10)
	ax.set_title(title, fontsize=11)
	ax.axis("off")


def backtest_dashboard(price_df, result, symbol="", initial_cash=100_000):
	equity = parse_equity_series(result)
	telemetry_score = parse_chart_series(result, "Decision Telemetry", "Score")
	trades = parse_closed_trades(result)
	stats = parse_stats(result)
	period = stats.get("Period", "")
	has_telemetry = telemetry_score is not None and not telemetry_score.empty

	fig = plt.figure(figsize=(16, 14 if has_telemetry else 12))
	fig.suptitle(f"Backtest Analysis — {symbol}  ({period})",
	             fontsize=13, fontweight="bold", y=0.99)
	if has_telemetry:
		gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.30,
		                       height_ratios=[1.35, 0.95, 1.0, 1.1])
	else:
		gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30,
		                       height_ratios=[1.4, 1.0, 1.1])

	ax_price = fig.add_subplot(gs[0, :])
	_plot_price_with_signals(ax_price, price_df)
	if not trades.empty:
		_plot_trades_on_price(ax_price, trades)

	if has_telemetry:
		ax_tel = fig.add_subplot(gs[1, :])
		ax_tel.plot(telemetry_score.index, telemetry_score.values, color="#1f4e79", lw=1.3, label="Score")
		ax_tel.axhline(0, color="#cccccc", lw=0.8, linestyle=":")
		ax_tel.set_title("Decision Telemetry", fontsize=11)
		ax_tel.set_ylabel("Model Score")
		ax_tel.legend(fontsize=8)
		ax_tel.grid(True, alpha=0.3)
		_style_date_axis(ax_tel)
		eq_row, stats_row = 2, 3
	else:
		eq_row, stats_row = 1, 2

	ax_eq = fig.add_subplot(gs[eq_row, 0])
	if equity is not None:
		_plot_equity_curve(ax_eq, equity, initial_cash=initial_cash)
	else:
		_no_data_panel(ax_eq, "Equity Curve", "No equity data\n(0 LEAN trades)")

	ax_dd = fig.add_subplot(gs[eq_row, 1])
	if equity is not None:
		_plot_drawdown(ax_dd, equity)
	else:
		_no_data_panel(ax_dd, "Drawdown", "No drawdown data\n(0 LEAN trades)")

	_plot_stats_table(fig.add_subplot(gs[stats_row, :]), stats)
	return fig


def plot_normalized_performance(ax, equity, benchmark=None, trades=None, title="Normalized Performance"):
	norm_eq = equity / equity.iloc[0]
	ax.plot(norm_eq.index, norm_eq.values, color="#4a90d9", lw=1.5, label="Strategy")
	if benchmark is not None:
		norm_bm = benchmark / benchmark.iloc[0]
		ax.plot(norm_bm.index, norm_bm.values, color="#aaaaaa", lw=1.2, linestyle="--", label="Benchmark")
	if trades is not None and not trades.empty:
		directions = trades["direction"].fillna("").astype(str).str.lower()
		longs = trades[directions.str.contains("long")]
		if not longs.empty:
			long_idx = longs["entry_time"]
			ax.scatter(long_idx, [norm_eq.asof(t) for t in long_idx], marker="^",
			           color="#2ecc71", s=100, zorder=5, edgecolors="white", lw=0.6,
			           label=f"Long entry ({len(longs)})")
	ax.axhline(1.0, color="#cccccc", lw=0.8, linestyle=":")
	ax.set_title(title, fontsize=11)
	ax.set_ylabel("Normalized Value")
	ax.legend(fontsize=9)
	ax.grid(True, alpha=0.3)
	_style_date_axis(ax)


# ── Strategy utilities ─────────────────────────────────────────────────────────

def load_strategy_config(config_path: Path) -> dict:
	"""Load strategy config — delegates to kernel.config if available."""
	return json.loads(config_path.read_text())


def load_backtest_result(strategy_dir: Path, run_name: str | None) -> tuple[dict, Path]:
	if run_name is None:
		return load_latest_backtest(strategy_dir)

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
	symbol = config.get("stock_symbol") or config.get("benchmark", "SPY")
	provider = config.get("data_src", "yfinance")

	# Determine strategy dir to get kernel fetch_ohlcv
	strategy_name = config.get("strategy") or config.get("model_name", "renquant_103")
	strategy_dir = REPO_ROOT / "backtesting" / strategy_name
	if str(strategy_dir) not in sys.path:
		sys.path.insert(0, str(strategy_dir))

	from kernel.data import fetch_ohlcv
	price_df = fetch_ohlcv(symbol, start=period_start, end=period_end, provider=provider).copy()
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
	fig = backtest_dashboard(
		price_df=price_df, result=result, symbol=symbol,
		initial_cash=config.get("initial_cash", 100_000),
	)
	output_path = run_dir / "dashboard.png"
	fig.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	return output_path


def save_normalized_chart(run_dir: Path, price_df: pd.DataFrame, result: dict,
                          is_multi_stock: bool = False) -> Path | None:
	equity = parse_equity_series(result)
	benchmark = build_benchmark(price_df, equity)
	trades = parse_closed_trades(result)
	if equity is None or equity.empty or benchmark is None or benchmark.empty:
		return None

	if is_multi_stock and not trades.empty and "symbol" in trades.columns:
		fig, ax = plt.subplots(figsize=(16, 6))
		plot_normalized_performance(ax=ax, equity=equity, benchmark=benchmark,
		                            trades=pd.DataFrame(),
		                            title="Normalized Performance vs SPY Buy & Hold")
		unique_symbols = sorted(trades["symbol"].unique())
		colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_symbols), 1)))
		symbol_colors = dict(zip(unique_symbols, colors))
		norm_equity = equity / equity.iloc[0]
		for sym in unique_symbols:
			sym_trades = trades[trades["symbol"] == sym]
			entry_vals = norm_equity.reindex(sym_trades["entry_time"], method="ffill").dropna()
			if not entry_vals.empty:
				ax.scatter(entry_vals.index, entry_vals.values, marker="^",
				           color=symbol_colors[sym], s=60, zorder=5, label=f"{sym}")
			exit_vals = norm_equity.reindex(sym_trades["exit_time"], method="ffill").dropna()
			if not exit_vals.empty:
				ax.scatter(exit_vals.index, exit_vals.values, marker="v",
				           color=symbol_colors[sym], s=60, zorder=5)
		ax.legend(fontsize=7, loc="upper left", ncol=3)
	else:
		fig, ax = plt.subplots(figsize=(14, 5))
		plot_normalized_performance(ax=ax, equity=equity, benchmark=benchmark,
		                            trades=trades,
		                            title="Normalized Performance vs Buy & Hold")

	output_path = run_dir / "normalized-performance.png"
	fig.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	return output_path


def build_trade_table(result: dict) -> pd.DataFrame | None:
	trades = parse_closed_trades(result)
	if trades.empty:
		return None
	equity = parse_equity_series(result)
	rows = []
	cumulative_pnl = 0.0
	for _, trade in trades.iterrows():
		cumulative_pnl += trade["pnl"]
		hold_days = (trade["exit_time"] - trade["entry_time"]).days
		ret_pct = (trade["exit_price"] / trade["entry_price"] - 1) * 100 if trade["entry_price"] > 0 else 0
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
	parser.add_argument(
		"--n-trials", type=int, default=1,
		help="Number of strategies/configurations searched to find this one (feeds DSR selection-bias correction; default 1)",
	)
	args = parser.parse_args()

	strategy_dir = REPO_ROOT / "backtesting" / args.strategy
	if not strategy_dir.exists():
		print(f"Error: strategy directory not found: {strategy_dir}", file=sys.stderr)
		return 1

	config = load_strategy_config(strategy_dir / "strategy_config.json")
	# Inject strategy name so build_price_frame can find kernel/
	config.setdefault("strategy", args.strategy)
	result, result_path = load_backtest_result(strategy_dir, args.run)
	run_dir = result_path.parent
	price_df = build_price_frame(config, result)
	stats = parse_stats(result)
	# §5.13.4: every Sharpe ships with DSR (+ PBO when multi-seed available).
	equity_series = parse_equity_series(result)
	stats.update(compute_significance_metrics(equity_series, n_trials=args.n_trials))

	is_multi_stock = "watchlist" in config

	dashboard_path = save_dashboard(strategy_dir, run_dir, config, result)
	normalized_path = save_normalized_chart(run_dir, price_df, result, is_multi_stock=is_multi_stock)

	trade_table = build_trade_table(result)
	trade_table_path = None
	if trade_table is not None and not trade_table.empty:
		trade_table_path = run_dir / "trade-details.csv"
		trade_table.to_csv(trade_table_path, index=False)

	summary_payload = {
		"strategy": args.strategy, "run": run_dir.name,
		"result_path": str(result_path), "dashboard_path": str(dashboard_path),
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
	for line in format_stats_lines(stats):
		print(line)

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
