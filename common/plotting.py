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
        summary_stems = {
            f.stem[: f.stem.rfind("-summary")]
            for f in run_dir.glob("*-summary.json")
        }
        candidates = [
            f for f in run_dir.glob("[0-9]*.json") if f.stem not in summary_stems
        ]
        if candidates:
            path = candidates[0]
            return json.loads(path.read_text()), path
    raise FileNotFoundError(f"No backtest results found in {backtests_dir}")


# ── Parsers ────────────────────────────────────────────────────────────────

def parse_equity_series(result: dict) -> pd.Series | None:
    """Extract equity curve as a UTC-indexed Series. Returns None if empty."""
    try:
        values = result["charts"]["Strategy Equity"]["series"]["Equity"]["values"]
        if not values:
            return None
        idx = [datetime.fromtimestamp(v["x"], tz=timezone.utc) for v in values]
        return pd.Series([v["y"] for v in values], index=idx, name="equity")
    except (KeyError, TypeError):
        return None


def parse_closed_trades(result: dict) -> pd.DataFrame:
    """Extract closed trades as a DataFrame with entry/exit pairs."""
    trades = result.get("totalPerformance", {}).get("closedTrades", [])
    if not trades:
        return pd.DataFrame()
    rows = [
        {
            "symbol": t["symbol"]["value"],
            "direction": t["direction"],
            "quantity": float(t["quantity"]),
            "entry_time": pd.to_datetime(t["entryTime"], utc=True),
            "entry_price": float(t["entryPrice"]),
            "exit_time": pd.to_datetime(t["exitTime"], utc=True),
            "exit_price": float(t["exitPrice"]),
            "pnl": float(t["profitLoss"]),
            "fees": float(t["totalFees"]),
        }
        for t in trades
    ]
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
        "Total Trades": int(ts.get("totalNumberOfTrades", 0)),
        "Win Rate": pct(ts.get("winRate", 0)),
        "Loss Rate": pct(ts.get("lossRate", 0)),
        "Profit Factor": num(ts.get("profitFactor", 0)),
        "Sharpe Ratio": num(ts.get("sharpeRatio", 0)),
        "Sortino Ratio": num(ts.get("sortinoRatio", 0)),
        "Net Profit": rt.get("Net Profit", "—"),
        "Total Return": rt.get("Return", "—"),
        "Ann. Return": pct(ps.get("compoundingAnnualReturn", 0)),
        "Max Drawdown": pct(ps.get("drawdown", 0)),
        "Ann. Std Dev": pct(ps.get("annualStandardDeviation", 0)),
        "Total Fees": rt.get("Fees", "—"),
        "Alpha": num(ps.get("alpha", 0)),
        "Beta": num(ps.get("beta", 0)),
    }


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
    ax.scatter(buys.index, buys["close"], marker="^", color="#2ecc71", s=90, zorder=5, label=f"Buy signal ({len(buys)})")
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


def plot_stats_table(ax, stats: dict):
    """Render a two-column performance statistics table."""
    ax.axis("off")
    rows = [(k, str(v)) for k, v in stats.items()]
    table = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
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
    trades = parse_closed_trades(result)
    stats = parse_stats(result)
    period = stats.get("Period", "")

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Backtest Analysis — {symbol}  ({period})",
        fontsize=13, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.30,
                           height_ratios=[1.4, 1, 1])

    # Row 0: price + signals (full width)
    ax_price = fig.add_subplot(gs[0, :])
    plot_price_with_signals(ax_price, price_df)
    if not trades.empty:
        plot_trades_on_price(ax_price, trades)

    # Row 1: equity curve
    ax_eq = fig.add_subplot(gs[1, 0])
    if equity is not None:
        plot_equity_curve(ax_eq, equity, initial_cash=initial_cash)
    else:
        _no_data_panel(ax_eq, "Equity Curve", "No equity data\n(0 LEAN trades)")

    # Row 1: drawdown
    ax_dd = fig.add_subplot(gs[1, 1])
    if equity is not None:
        plot_drawdown(ax_dd, equity)
    else:
        _no_data_panel(ax_dd, "Drawdown", "No drawdown data\n(0 LEAN trades)")

    # Row 2: stats table (full width)
    ax_stats = fig.add_subplot(gs[2, :])
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
        longs = trades[trades["direction"].str.lower().str.contains("long")]
        shorts = trades[~trades["direction"].str.lower().str.contains("long")]

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
