"""Visualization utilities for LEAN backtest results."""
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


# ── Result loading ─────────────────────────────────────────────────────────

def load_latest_backtest(strategy_dir: Path) -> tuple[dict, Path]:
    """Find and load the most recent LEAN backtest result JSON.

    Returns (result_dict, result_path).
    """
    backtests_dir = strategy_dir / "backtests"
    run_dirs = sorted(
        (d for d in backtests_dir.iterdir() if d.is_dir()),
        reverse=True,
    )
    for run_dir in run_dirs:
        candidates = [
            f
            for f in run_dir.glob("[0-9]*.json")
            if "-" not in f.stem
        ]
        if candidates:
            path = candidates[0]
            result = json.loads(path.read_text())
            if isinstance(result, dict):
                result["_result_path"] = str(path)
            return result, path
    raise FileNotFoundError(f"No backtest results found in {backtests_dir}")


# ── Parsers ────────────────────────────────────────────────────────────────

def parse_equity_series(result: dict) -> pd.Series | None:
    """Extract equity curve as a UTC-indexed Series. Returns None if empty."""
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
    """Extract a chart series as a UTC-indexed Series. Returns None if empty."""
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


def parse_decision_telemetry(result: dict) -> pd.DataFrame:
    """Extract score, thresholds, and action telemetry emitted by the LEAN strategy."""
    series_map = {
        "score": parse_chart_series(result, "Decision Telemetry", "Score"),
        "buy_threshold": parse_chart_series(result, "Decision Telemetry", "Buy Threshold"),
        "sell_threshold": parse_chart_series(result, "Decision Telemetry", "Sell Threshold"),
        "raw_action": parse_chart_series(result, "Decision Telemetry", "Raw Action"),
        "action": parse_chart_series(result, "Decision Telemetry", "Action"),
    }

    non_empty = {name: series for name, series in series_map.items() if series is not None and not series.empty}
    if not non_empty:
        return pd.DataFrame()

    telemetry = pd.concat(non_empty, axis=1).sort_index()
    return telemetry.ffill()


def parse_closed_trades(result: dict) -> pd.DataFrame:
    """Extract closed trades as a DataFrame with entry/exit pairs."""
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
            rows.append(
                {
                    "symbol": symbol or trade.get("symbolValue", ""),
                    "direction": direction,
                    "quantity": abs(float(trade.get("quantity", 0))),
                    "entry_time": pd.to_datetime(trade["entryTime"], utc=True),
                    "entry_price": float(trade["entryPrice"]),
                    "exit_time": pd.to_datetime(trade["exitTime"], utc=True),
                    "exit_price": float(trade["exitPrice"]),
                    "pnl": float(trade.get("profitLoss", 0)),
                    "fees": float(trade.get("totalFees", 0)),
                }
            )
        return pd.DataFrame(rows)

    result_path_text = result.get("_result_path")
    if not result_path_text:
        return pd.DataFrame()

    order_events_path = Path(result_path_text).with_name(f"{Path(result_path_text).stem}-order-events.json")
    if not order_events_path.exists():
        return pd.DataFrame()

    events = json.loads(order_events_path.read_text())
    filled_events = [
        event for event in events
        if event.get("status") == "filled" and float(event.get("fillQuantity", 0)) != 0
    ]
    if not filled_events:
        return pd.DataFrame()

    rows = []
    open_trade = None
    for event in filled_events:
        direction = str(event.get("direction", "")).lower()
        event_time = datetime.fromtimestamp(event["time"], tz=timezone.utc)
        quantity = abs(float(event.get("fillQuantity", 0)))
        fee = float(event.get("orderFeeAmount", 0) or 0)
        if direction == "buy":
            open_trade = {
                "symbol": event.get("symbolValue") or event.get("symbol", ""),
                "direction": "Long",
                "quantity": quantity,
                "entry_time": event_time,
                "entry_price": float(event.get("fillPrice", 0)),
                "fees": fee,
            }
        elif direction == "sell" and open_trade is not None:
            exit_price = float(event.get("fillPrice", 0))
            total_fees = open_trade["fees"] + fee
            pnl = (exit_price - open_trade["entry_price"]) * open_trade["quantity"] - total_fees
            rows.append(
                {
                    "symbol": open_trade["symbol"],
                    "direction": open_trade["direction"],
                    "quantity": open_trade["quantity"],
                    "entry_time": open_trade["entry_time"],
                    "entry_price": open_trade["entry_price"],
                    "exit_time": event_time,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "fees": total_fees,
                }
            )
            open_trade = None

    return pd.DataFrame(rows)


def parse_stats(result: dict) -> dict:
    """Extract key performance metrics from a LEAN result dict."""
    ts = result.get("totalPerformance", {}).get("tradeStatistics", {})
    ps = result.get("totalPerformance", {}).get("portfolioStatistics", {})
    rt = result.get("runtimeStatistics", {})
    cfg = result.get("algorithmConfiguration", {})
    state = result.get("state", {})

    def pct(v):
        try:
            return f"{float(v) * 100:.2f}%"
        except (ValueError, TypeError):
            return "—"

    def num(v):
        try:
            return f"{float(v):.4f}"
        except (ValueError, TypeError):
            return "—"

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
        **{
            key: rt[key]
            for key in [
                "Policy",
                "Wash Sale Days",
                "Min Hold Days",
                "Buy Decisions",
                "Sell Decisions",
                "Hold Decisions",
                "Executed Buys",
                "Executed Sells",
                "Blocked Wash Sales",
                "Blocked Min Hold",
            ]
            if key in rt
        },
    }


def format_stats_lines(stats: dict) -> list[str]:
    """Format performance stats for CLI output."""
    width = max(len(key) for key in stats)
    return [f"{key:<{width}} : {value}" for key, value in stats.items()]


# ── Plot helpers ───────────────────────────────────────────────────────────

def _style_date_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.figure.autofmt_xdate(rotation=30, ha="right")


# ── Plot functions ─────────────────────────────────────────────────────────

def plot_price_with_signals(ax, price_df: pd.DataFrame, title: str = "Price + Model Signals"):
    """Plot close price overlaid with buy/sell gate signal markers.

    price_df must have columns: close, buy_signal, sell_signal.
    """
    ax.plot(price_df.index, price_df["close"], color="#4a90d9", lw=1.2, label="Close")

    buys = price_df[price_df["buy_signal"].astype(bool)]
    sells = price_df[price_df["sell_signal"].astype(bool)]
    if not buys.empty:
        ax.scatter(buys.index, buys["close"], marker="^", color="#2ecc71", s=90, zorder=5, label=f"Buy signal ({len(buys)})")
    if not sells.empty:
        ax.scatter(sells.index, sells["close"], marker="v", color="#e74c3c", s=90, zorder=5, label=f"Sell signal ({len(sells)})")

    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _style_date_axis(ax)


def plot_trades_on_price(ax, trades: pd.DataFrame):
    """Overlay LEAN closed trade entry/exit points on an existing price axis."""
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


def plot_equity_curve(ax, equity: pd.Series, initial_cash: float = 100_000):
    """Plot portfolio equity curve with profit/loss shading."""
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


def plot_drawdown(ax, equity: pd.Series):
    """Plot rolling drawdown (%) derived from an equity curve."""
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


def plot_decision_telemetry(ax, telemetry: pd.DataFrame):
    """Plot model score/threshold telemetry with action markers."""
    score = telemetry.get("score")
    if score is not None:
        ax.plot(score.index, score.values, color="#1f4e79", lw=1.3, label="Score")

    buy_threshold = telemetry.get("buy_threshold")
    if buy_threshold is not None:
        ax.plot(
            buy_threshold.index,
            buy_threshold.values,
            color="#2ecc71",
            lw=1.0,
            linestyle="--",
            label="Buy threshold",
        )

    sell_threshold = telemetry.get("sell_threshold")
    if sell_threshold is not None:
        ax.plot(
            sell_threshold.index,
            sell_threshold.values,
            color="#e74c3c",
            lw=1.0,
            linestyle="--",
            label="Sell threshold",
        )

    action = telemetry.get("action")
    if action is not None:
        buys = action[action >= 1]
        sells = action[action <= -1]
        holds = action[action == 0]
        if not buys.empty:
            buy_y = score.reindex(buys.index) if score is not None else pd.Series(0.0, index=buys.index)
            ax.scatter(buys.index, buy_y.values, marker="^", color="#2ecc71", s=70, zorder=5, label="Final buy")
        if not sells.empty:
            sell_y = score.reindex(sells.index) if score is not None else pd.Series(0.0, index=sells.index)
            ax.scatter(sells.index, sell_y.values, marker="v", color="#e74c3c", s=70, zorder=5, label="Final sell")
        if score is not None and not holds.empty:
            hold_y = score.reindex(holds.index)
            ax.scatter(holds.index, hold_y.values, marker="o", color="#7f8c8d", s=20, alpha=0.35, zorder=4, label="Hold")

    ax.axhline(0, color="#cccccc", lw=0.8, linestyle=":")
    ax.set_title("Decision Telemetry", fontsize=11)
    ax.set_ylabel("Model Score")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(True, alpha=0.3)
    _style_date_axis(ax)


def plot_stats_table(ax, stats: dict):
    """Render a two-column performance statistics table."""
    ax.axis("off")
    rows = [(k, str(v)) for k, v in stats.items()]
    midpoint = int(np.ceil(len(rows) / 2))
    left_rows = rows[:midpoint]
    right_rows = rows[midpoint:]

    left_table = ax.table(
        cellText=left_rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        bbox=[0.00, 0.0, 0.48, 1.0],
    )
    right_table = ax.table(
        cellText=right_rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        bbox=[0.52, 0.0, 0.48, 1.0],
    )

    for table in (left_table, right_table):
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


# ── Full dashboard ─────────────────────────────────────────────────────────

def backtest_dashboard(
    price_df: pd.DataFrame,
    result: dict,
    symbol: str = "",
    initial_cash: float = 100_000,
) -> plt.Figure:
    """Render a 4-panel backtest dashboard.

    Panels:
      [0] Price + model gate signals (+ LEAN trades if available)
      [1] Equity curve (LEAN) or placeholder
      [2] Drawdown (LEAN) or placeholder
      [3] Performance statistics table

    Args:
        price_df: OHLCV DataFrame with buy_signal and sell_signal columns.
        result:   LEAN result dict from load_latest_backtest().
        symbol:   Ticker symbol for the chart title.
        initial_cash: Starting capital for equity curve baseline.
    """
    equity = parse_equity_series(result)
    telemetry = parse_decision_telemetry(result)
    trades = parse_closed_trades(result)
    stats = parse_stats(result)
    period = stats.get("Period", "")

    has_telemetry = not telemetry.empty and "score" in telemetry

    fig = plt.figure(figsize=(16, 14 if has_telemetry else 12))
    fig.suptitle(
        f"Backtest Analysis — {symbol}  ({period})",
        fontsize=13, fontweight="bold", y=0.99,
    )
    if has_telemetry:
        gs = gridspec.GridSpec(
            4,
            2,
            figure=fig,
            hspace=0.45,
            wspace=0.30,
            height_ratios=[1.35, 0.95, 1.0, 1.1],
        )
    else:
        gs = gridspec.GridSpec(
            3,
            2,
            figure=fig,
            hspace=0.45,
            wspace=0.30,
            height_ratios=[1.4, 1.0, 1.1],
        )

    # Row 0: price + signals (full width)
    ax_price = fig.add_subplot(gs[0, :])
    plot_price_with_signals(ax_price, price_df)
    if not trades.empty:
        plot_trades_on_price(ax_price, trades)

    if has_telemetry:
        ax_telemetry = fig.add_subplot(gs[1, :])
        plot_decision_telemetry(ax_telemetry, telemetry)
        equity_row = 2
        stats_row = 3
    else:
        equity_row = 1
        stats_row = 2

    # Equity row
    ax_eq = fig.add_subplot(gs[equity_row, 0])
    if equity is not None:
        plot_equity_curve(ax_eq, equity, initial_cash=initial_cash)
    else:
        _no_data_panel(ax_eq, "Equity Curve", "No equity data\n(0 LEAN trades)")

    ax_dd = fig.add_subplot(gs[equity_row, 1])
    if equity is not None:
        plot_drawdown(ax_dd, equity)
    else:
        _no_data_panel(ax_dd, "Drawdown", "No drawdown data\n(0 LEAN trades)")

    ax_stats = fig.add_subplot(gs[stats_row, :])
    plot_stats_table(ax_stats, stats)

    return fig


# ── Normalized performance chart ──────────────────────────────────────

def plot_normalized_performance(
    ax,
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
    title: str = "Normalized Performance",
):
    """Plot strategy vs benchmark normalized to 1.0, with trade entry markers.

    Args:
        ax: Matplotlib axes.
        equity: Portfolio equity series.
        benchmark: Optional benchmark equity series (e.g. buy-and-hold).
        trades: LEAN closed-trades DataFrame (from ``parse_closed_trades``).
            Must have ``entry_time``, ``entry_price``, ``direction`` columns.
    """
    norm_eq = equity / equity.iloc[0]
    ax.plot(norm_eq.index, norm_eq.values, color="#4a90d9", lw=1.5, label="Strategy")

    if benchmark is not None:
        norm_bm = benchmark / benchmark.iloc[0]
        ax.plot(norm_bm.index, norm_bm.values, color="#aaaaaa", lw=1.2,
                linestyle="--", label="Benchmark")

    if trades is not None and not trades.empty:
        # Normalize entry prices at the closest equity value
        directions = trades["direction"].fillna("").astype(str).str.lower()
        longs = trades[directions.str.contains("long")]
        shorts = trades[~directions.str.contains("long")]

        if not longs.empty:
            long_idx = longs["entry_time"]
            long_vals = [norm_eq.asof(t) for t in long_idx]
            ax.scatter(long_idx, long_vals, marker="^", color="#2ecc71",
                       s=100, zorder=5, edgecolors="white", lw=0.6,
                       label=f"Long entry ({len(longs)})")

        if not shorts.empty:
            short_idx = shorts["entry_time"]
            short_vals = [norm_eq.asof(t) for t in short_idx]
            ax.scatter(short_idx, short_vals, marker="v", color="#e74c3c",
                       s=100, zorder=5, edgecolors="white", lw=0.6,
                       label=f"Short entry ({len(shorts)})")

    ax.axhline(1.0, color="#cccccc", lw=0.8, linestyle=":")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Normalized Value")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _style_date_axis(ax)
