#!/usr/bin/env python
"""Deep per-trade comparison between two sim configs.

Usage::

    python scripts/compare_sims.py \
        --config-a strategy_config.h60_103.json \
        --config-b strategy_config.golden.json \
        --label-a "60d model" --label-b "Golden (10d)"

Outputs:
  - Summary stats side by side
  - Per-ticker P&L comparison
  - Hold-duration distribution
  - Win-rate by exit reason
  - Ticker overlap: which tickers each model traded more
  - Regime breakdown of trades
  - Entry timing: which months/quarters each model was most active
  - Saves trade logs to /tmp/sim_trades_*.json for further analysis
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.WARNING,   # suppress sim noise
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("compare-sims")
logging.getLogger("run-sim-104").setLevel(logging.INFO)

STRATEGY  = "renquant_104"
SIM_START = "2024-01-02"
SIM_END   = "2026-03-28"


def _run_sim(config_name: str, strategy_dir: Path, ohlcv: dict,
             spy_df: pd.DataFrame, etf_map: dict, cash: float):
    """Load config and run sim; return SimResult."""
    from sim.runner import run_backtest   # noqa: PLC0415
    cfg = json.loads((strategy_dir / config_name).read_text())
    cfg["_strategy_dir"]  = str(strategy_dir)
    cfg["initial_cash"]   = cash
    cfg["backtest_start"] = SIM_START
    cfg["backtest_end"]   = SIM_END
    print(f"  Running {config_name} …", flush=True)
    return run_backtest(config=cfg, strategy_dir=strategy_dir, ohlcv=ohlcv,
                        spy_df=spy_df, sector_etf_map=etf_map, snapshot=False)


def _matched_trades(result) -> pd.DataFrame:
    """Build a DataFrame of completed round-trips (buy → sell pairs)."""
    buys:  dict[str, list[dict]] = defaultdict(list)
    trips: list[dict] = []
    for t in result.trade_log:
        if t["action"] == "buy":
            buys[t["ticker"]].append(t)
        elif t["action"] == "sell" and buys.get(t["ticker"]):
            b = buys[t["ticker"]].pop(0)
            hold = (pd.Timestamp(t["date"]) - pd.Timestamp(b["date"])).days
            pnl  = t.get("pnl_pct", 0.0) or 0.0
            trips.append({
                "ticker":       t["ticker"],
                "buy_date":     b["date"],
                "sell_date":    t["date"],
                "hold_days":    hold,
                "pnl_pct":      pnl,
                "buy_regime":   b.get("regime", "?"),
                "exit_reason":  t.get("reason", "?"),
                "buy_price":    b.get("price", 0.0),
                "sell_price":   t.get("price", 0.0),
                "shares":       b.get("shares", 0),
            })
    return pd.DataFrame(trips)


def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-a", default="strategy_config.h60_103.json")
    p.add_argument("--config-b", default="strategy_config.golden.json")
    p.add_argument("--label-a",  default="Config-A")
    p.add_argument("--label-b",  default="Config-B (golden)")
    p.add_argument("--cash",     type=float, default=100_000)
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / STRATEGY
    sys.path.insert(0, str(strategy_dir))

    from kernel.data import fetch_ohlcv  # noqa: PLC0415

    print("Loading market data …")
    spy_df  = fetch_ohlcv("SPY")
    etf_map = json.loads((strategy_dir / args.config_a).read_text()).get(
        "sector_etf_map", {})
    watchlist_a = json.loads((strategy_dir / args.config_a).read_text()).get(
        "watchlist", [])
    watchlist_b = json.loads((strategy_dir / args.config_b).read_text()).get(
        "watchlist", [])
    all_syms = sorted(set(watchlist_a) | set(watchlist_b) | set(etf_map.values()))
    ohlcv: dict = {"SPY": spy_df}
    for sym in all_syms:
        try:
            ohlcv[sym] = fetch_ohlcv(sym)
        except Exception:
            pass

    res_a = _run_sim(args.config_a, strategy_dir, ohlcv, spy_df, etf_map, args.cash)
    res_b = _run_sim(args.config_b, strategy_dir, ohlcv, spy_df, etf_map, args.cash)

    # Save raw trade logs
    Path("/tmp/sim_trades_a.json").write_text(json.dumps(res_a.trade_log, indent=1))
    Path("/tmp/sim_trades_b.json").write_text(json.dumps(res_b.trade_log, indent=1))
    print(f"\nTrade logs saved → /tmp/sim_trades_a.json  /tmp/sim_trades_b.json")

    df_a = _matched_trades(res_a)
    df_b = _matched_trades(res_b)

    # ── 1. TOP-LINE ──────────────────────────────────────────────────────────
    _print_section("1. TOP-LINE SUMMARY")
    fmt = "{:<35} {:>12} {:>12}"
    print(fmt.format("Metric", args.label_a[:12], args.label_b[:12]))
    print("-" * 60)
    def row(label, va, vb): print(fmt.format(label, va, vb))
    row("APY",             f"{res_a.apy*100:+.1f}%",  f"{res_b.apy*100:+.1f}%")
    row("Total return",    f"{res_a.total_return*100:+.1f}%", f"{res_b.total_return*100:+.1f}%")
    row("Win rate",        f"{res_a.win_rate:.0%}",   f"{res_b.win_rate:.0%}")
    row("Buys",            str(len(res_a.buys)),       str(len(res_b.buys)))
    row("Sells",           str(len(res_a.sells)),      str(len(res_b.sells)))
    row("Avg hold (days)", f"{res_a.avg_hold:.0f}",   f"{res_b.avg_hold:.0f}")
    row("Avg P&L/trade",   f"{res_a.avg_pnl*100:+.1f}%", f"{res_b.avg_pnl*100:+.1f}%")
    row("Total tax",       f"${res_a.total_tax:,.0f}", f"${res_b.total_tax:,.0f}")
    apy_delta = (res_a.apy - res_b.apy) * 100
    print()
    print(f"  APY delta ({args.label_a} vs {args.label_b}): {apy_delta:+.1f} pp")
    verdict = "✓ PROMOTE candidate" if apy_delta >= 0 else "✗ REJECT"
    print(f"  Verdict: {verdict}")

    if len(df_a) == 0 or len(df_b) == 0:
        print("\n[No completed round-trips — cannot do per-trade analysis]")
        return

    # ── 2. HOLD DURATION ────────────────────────────────────────────────────
    _print_section("2. HOLD DURATION DISTRIBUTION (days)")
    for label, df in [(args.label_a, df_a), (args.label_b, df_b)]:
        q = df["hold_days"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        print(f"  {label:<30}  "
              f"p10={q[0.1]:.0f}  p25={q[0.25]:.0f}  "
              f"median={q[0.5]:.0f}  p75={q[0.75]:.0f}  "
              f"p90={q[0.9]:.0f}  mean={df['hold_days'].mean():.0f}")

    # ── 3. EXIT REASON BREAKDOWN ─────────────────────────────────────────────
    _print_section("3. EXIT REASON BREAKDOWN")
    all_reasons = sorted(
        set(df_a["exit_reason"].unique()) | set(df_b["exit_reason"].unique()))
    fmt3 = "{:<30} {:>10} {:>10}"
    print(fmt3.format("Exit reason", args.label_a[:10], args.label_b[:10]))
    print("-" * 52)
    for r in all_reasons:
        ca = (df_a["exit_reason"] == r).sum()
        cb = (df_b["exit_reason"] == r).sum()
        print(fmt3.format(r[:29], str(ca), str(cb)))

    # ── 4. WIN RATE BY EXIT REASON ───────────────────────────────────────────
    _print_section("4. WIN RATE BY EXIT REASON")
    fmt4 = "{:<30} {:>12} {:>12}"
    print(fmt4.format("Exit reason", args.label_a[:12], args.label_b[:12]))
    print("-" * 56)
    for r in all_reasons:
        wa = df_a[df_a["exit_reason"]==r]["pnl_pct"].gt(0).mean() if (df_a["exit_reason"]==r).any() else float("nan")
        wb = df_b[df_b["exit_reason"]==r]["pnl_pct"].gt(0).mean() if (df_b["exit_reason"]==r).any() else float("nan")
        print(fmt4.format(r[:29],
                          f"{wa:.0%}" if not np.isnan(wa) else "—",
                          f"{wb:.0%}" if not np.isnan(wb) else "—"))

    # ── 5. PER-TICKER TRADE FREQUENCY ───────────────────────────────────────
    _print_section("5. TOP 20 MOST-TRADED TICKERS (each model)")
    tc_a = df_a["ticker"].value_counts().head(20)
    tc_b = df_b["ticker"].value_counts().head(20)
    fmt5 = "{:<8} {:>6} {:>8}  {:<8} {:>6} {:>8}"
    print(fmt5.format("Ticker-A", "Cnt-A", "WR-A", "Ticker-B", "Cnt-B", "WR-B"))
    print("-" * 48)
    for i in range(min(20, max(len(tc_a), len(tc_b)))):
        ta = tc_a.index[i] if i < len(tc_a) else ""
        ca = tc_a.iloc[i]  if i < len(tc_a) else 0
        wa = df_a[df_a["ticker"]==ta]["pnl_pct"].gt(0).mean() if ta else float("nan")
        tb = tc_b.index[i] if i < len(tc_b) else ""
        cb = tc_b.iloc[i]  if i < len(tc_b) else 0
        wb = df_b[df_b["ticker"]==tb]["pnl_pct"].gt(0).mean() if tb else float("nan")
        print(fmt5.format(
            ta, ca, f"{wa:.0%}" if ta and not np.isnan(wa) else "—",
            tb, cb, f"{wb:.0%}" if tb and not np.isnan(wb) else "—"))

    # ── 6. TICKERS UNIQUE TO EACH MODEL ──────────────────────────────────────
    _print_section("6. TICKERS ONLY TRADED BY ONE MODEL")
    set_a = set(df_a["ticker"].unique())
    set_b = set(df_b["ticker"].unique())
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    print(f"  Only in {args.label_a}: {only_a}")
    print(f"  Only in {args.label_b}: {only_b}")
    print(f"  Common tickers: {len(set_a & set_b)}")

    # ── 7. PER-TICKER P&L COMPARISON ─────────────────────────────────────────
    _print_section("7. PER-TICKER AVG P&L (tickers in both models)")
    common_tickers = sorted(set_a & set_b)
    pnl_a = df_a.groupby("ticker")["pnl_pct"].mean()
    pnl_b = df_b.groupby("ticker")["pnl_pct"].mean()
    diff_pnl = {t: pnl_a.get(t, 0) - pnl_b.get(t, 0) for t in common_tickers}
    worst = sorted(diff_pnl.items(), key=lambda x: x[1])[:10]
    best  = sorted(diff_pnl.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  Top 10 A OUTPERFORMS B (A avg_pnl - B avg_pnl):")
    for t, d in best:
        print(f"    {t:<8}  A={pnl_a.get(t,0)*100:+.1f}%  B={pnl_b.get(t,0)*100:+.1f}%  delta={d*100:+.1f}%")
    print(f"\n  Top 10 B OUTPERFORMS A:")
    for t, d in worst:
        print(f"    {t:<8}  A={pnl_a.get(t,0)*100:+.1f}%  B={pnl_b.get(t,0)*100:+.1f}%  delta={d*100:+.1f}%")

    # ── 8. REGIME BREAKDOWN ──────────────────────────────────────────────────
    _print_section("8. TRADE COUNT BY REGIME AT BUY")
    all_regimes = sorted(
        set(df_a["buy_regime"].unique()) | set(df_b["buy_regime"].unique()))
    fmt8 = "{:<20} {:>10} {:>10}"
    print(fmt8.format("Regime", args.label_a[:10], args.label_b[:10]))
    print("-" * 42)
    for r in all_regimes:
        ca = (df_a["buy_regime"] == r).sum()
        cb = (df_b["buy_regime"] == r).sum()
        print(fmt8.format(r[:19], str(ca), str(cb)))

    # ── 9. MONTHLY ACTIVITY ──────────────────────────────────────────────────
    _print_section("9. BUY ACTIVITY BY QUARTER")
    df_a["quarter"] = pd.to_datetime(df_a["buy_date"]).dt.to_period("Q").astype(str)
    df_b["quarter"] = pd.to_datetime(df_b["buy_date"]).dt.to_period("Q").astype(str)
    all_q = sorted(set(df_a["quarter"].unique()) | set(df_b["quarter"].unique()))
    fmt9 = "{:<10} {:>10} {:>10}"
    print(fmt9.format("Quarter", args.label_a[:10], args.label_b[:10]))
    print("-" * 32)
    for q in all_q:
        ca = (df_a["quarter"] == q).sum()
        cb = (df_b["quarter"] == q).sum()
        bar_a = "█" * min(ca, 40)
        bar_b = "█" * min(cb, 40)
        print(fmt9.format(q, str(ca), str(cb)))

    # ── 10. P&L DISTRIBUTION ─────────────────────────────────────────────────
    _print_section("10. P&L DISTRIBUTION PER TRADE")
    for label, df in [(args.label_a, df_a), (args.label_b, df_b)]:
        pnl = df["pnl_pct"] * 100
        print(f"  {label}")
        print(f"    mean={pnl.mean():+.1f}%  std={pnl.std():.1f}%  "
              f"min={pnl.min():+.1f}%  max={pnl.max():+.1f}%")
        bins = [-100, -20, -10, -5, 0, 5, 10, 20, 100]
        hist = pd.cut(pnl, bins=bins).value_counts().sort_index()
        for interval, count in hist.items():
            print(f"    {str(interval):<20} {count:>5}  {'█'*min(count//2,40)}")
        print()


if __name__ == "__main__":
    main()
