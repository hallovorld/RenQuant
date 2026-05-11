#!/usr/bin/env python
"""Re-run baseline + extract trade/equity distributions for grid-sweep design.

Outputs to stdout AND writes a JSON sidecar at data/logs/baseline_distributions.json
for later use. Captures:

  * Per-trade P&L distribution by exit_reason
  * Per-trade max-favorable-excursion (peak gain reached before exit)
  * Per-trade max-adverse-excursion (worst drawdown from entry)
  * Daily portfolio return percentiles
  * Daily portfolio drawdown profile (equity curve peak→trough)
  * Position-level single-day moves (what % drops actually trigger SDL?)
  * Trailing-stop trigger profile (peak-gain when armed, drop when fired)

Each section ends with a one-line recommendation of candidate values
informed by the distribution percentiles.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
STRAT = REPO / "backtesting" / "renquant_104"
sys.path.insert(0, str(STRAT))
sys.path.insert(0, str(REPO))

# Reuse the existing CLI but capture the SimResult
from kernel.data import fetch_ohlcv          # noqa: E402
from sim.runner import run_backtest           # noqa: E402

print("Running baseline sim (this takes ~25 min) …")
config_path = STRAT / "strategy_config.sim_baseline.json"
config = json.loads(config_path.read_text())
config["_strategy_dir"]         = str(STRAT)
config["_strategy_config_name"] = "strategy_config.sim_baseline.json"
config["initial_cash"]          = 100_000.0
config["backtest_start"]        = "2024-04-01"
config["backtest_end"]          = "2026-03-26"
config["persistence"]           = {"enabled": False}

benchmark = config.get("benchmark", "SPY")
spy_df = fetch_ohlcv(benchmark)
ohlcv = {benchmark: spy_df}
etf_map = config.get("sector_etf_map", {})
for sym in sorted(set(config.get("watchlist", [])) | set(etf_map.values())):
    try:
        ohlcv[sym] = fetch_ohlcv(sym)
    except Exception as exc:
        print(f"  {sym}: {exc}", file=sys.stderr)

res = run_backtest(
    config=config, strategy_dir=STRAT, ohlcv=ohlcv, spy_df=spy_df,
    sector_etf_map=etf_map, snapshot=False,
)

print()
print("=" * 70)
print("BASELINE DISTRIBUTION ANALYSIS — 2024-04-01 → 2026-03-26")
print("=" * 70)

dump: dict = {}

# ── 1. Per-trade P&L distribution ─────────────────────────────────────
sells = pd.DataFrame(res.sells) if res.sells else pd.DataFrame()
print(f"\n[1] Closed positions: {len(sells)}")
if len(sells) > 0 and "pnl_pct" in sells.columns:
    print(f"    Mean P&L:    {sells['pnl_pct'].mean():+.2%}")
    print(f"    Median P&L:  {sells['pnl_pct'].median():+.2%}")
    print(f"    Std P&L:     {sells['pnl_pct'].std():.2%}")
    print(f"    Percentiles of trade-level P&L:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        v = sells["pnl_pct"].quantile(p / 100.0)
        print(f"       p{p:2d}: {v:+.2%}")
    dump["trade_pnl_pct"] = {
        "n": int(len(sells)),
        "mean": float(sells["pnl_pct"].mean()),
        "median": float(sells["pnl_pct"].median()),
        "std": float(sells["pnl_pct"].std()),
        "percentiles": {
            f"p{p}": float(sells["pnl_pct"].quantile(p/100))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        },
    }

# ── 2. P&L by exit reason ──────────────────────────────────────────────
print(f"\n[2] Per-trade P&L by exit_reason:")
if len(sells) > 0 and "exit_reason" in sells.columns:
    grp = sells.groupby("exit_reason")["pnl_pct"]
    by_reason = {}
    for reason, vals in grp:
        d = {
            "n": int(len(vals)),
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "p10": float(vals.quantile(0.10)),
            "p25": float(vals.quantile(0.25)),
            "p75": float(vals.quantile(0.75)),
            "p90": float(vals.quantile(0.90)),
        }
        by_reason[reason] = d
        print(f"    {reason:20s} n={d['n']:>3}  mean={d['mean']:+.2%}  "
              f"median={d['median']:+.2%}  "
              f"p10={d['p10']:+.2%}  p90={d['p90']:+.2%}")
    dump["trade_pnl_by_reason"] = by_reason

# ── 3. Equity-curve daily returns ─────────────────────────────────────
eq = res.equity_df
print(f"\n[3] Equity curve: {len(eq)} bars")
if "portfolio" in eq.columns:
    pv = eq["portfolio"].astype(float).dropna()
    daily_ret = pv.pct_change().dropna()
    print(f"    Daily-return percentiles:")
    for p in [0.5, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5]:
        v = daily_ret.quantile(p / 100.0)
        print(f"       p{p:>5}: {v:+.2%}")
    print(f"    Worst single day:   {daily_ret.min():+.2%} on {daily_ret.idxmin()}")
    print(f"    Best  single day:   {daily_ret.max():+.2%} on {daily_ret.idxmax()}")
    print(f"    Daily σ:            {daily_ret.std():.2%}")
    print(f"    Annualised σ:       {daily_ret.std() * np.sqrt(252):.2%}")
    # Count days below specific thresholds
    print(f"\n    Day-count below thresholds:")
    for thr in [-0.10, -0.08, -0.06, -0.05, -0.04, -0.03, -0.02]:
        n = int((daily_ret < thr).sum())
        pct = 100 * n / len(daily_ret)
        print(f"       daily return < {thr:+.1%}: {n:>3} days ({pct:.1f}% of days)")
    dump["daily_returns"] = {
        "n": int(len(daily_ret)),
        "worst": float(daily_ret.min()),
        "best": float(daily_ret.max()),
        "daily_sigma": float(daily_ret.std()),
        "ann_sigma": float(daily_ret.std() * np.sqrt(252)),
        "percentiles": {
            f"p{p}": float(daily_ret.quantile(p/100))
            for p in [0.5, 1, 5, 10, 25, 50, 75, 90, 95, 99]
        },
        "days_below": {
            f"{thr:+.1%}": int((daily_ret < thr).sum())
            for thr in [-0.10, -0.08, -0.06, -0.05, -0.04, -0.03, -0.02]
        },
    }

    # Drawdown trajectory
    peak = pv.cummax()
    dd = (pv - peak) / peak
    print(f"\n    Drawdown trajectory:")
    print(f"       MaxDD: {dd.min():+.2%} at {dd.idxmin()}")
    peak_idx = pv.loc[:dd.idxmin()].idxmax()
    print(f"       Peak before MaxDD: {peak_idx} (${peak.loc[dd.idxmin()]:,.0f})")
    print(f"       Trough at MaxDD:    {dd.idxmin()} (${pv.loc[dd.idxmin()]:,.0f})")
    print(f"       Loss in $: ${peak.loc[dd.idxmin()] - pv.loc[dd.idxmin()]:,.0f}")
    print(f"       DD duration: {(pd.Timestamp(dd.idxmin()) - pd.Timestamp(peak_idx)).days}d")
    print(f"    Days spent at DD ≥ threshold:")
    for thr in [-0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40]:
        n = int((dd < thr).sum())
        pct = 100 * n / len(dd)
        print(f"       DD < {thr:+.1%}: {n:>3} days ({pct:.1f}%)")
    dump["drawdown"] = {
        "max_dd": float(dd.min()),
        "max_dd_date": str(dd.idxmin()),
        "peak_date": str(peak_idx),
        "duration_days": int((pd.Timestamp(dd.idxmin()) - pd.Timestamp(peak_idx)).days),
        "days_below_threshold": {
            f"{thr:+.1%}": int((dd < thr).sum())
            for thr in [-0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40]
        },
    }

# ── 4. SDL-triggered trade analysis ───────────────────────────────────
if len(sells) > 0 and "exit_reason" in sells.columns:
    sdl = sells[sells["exit_reason"] == "single_day_loss"]
    print(f"\n[4] SDL-triggered exits ({len(sdl)}):")
    if len(sdl) > 0 and "pnl_pct" in sdl.columns:
        print(f"    P&L on SDL exits — median: {sdl['pnl_pct'].median():+.2%}")
        print(f"    P&L percentiles:")
        for p in [10, 25, 50, 75, 90]:
            v = sdl["pnl_pct"].quantile(p / 100.0)
            print(f"       p{p:2d}: {v:+.2%}")
        # If hold_days available, infer single-day move from total P&L / hold
        if "hold_days" in sdl.columns:
            print(f"    Hold-days on SDL exits: median {sdl['hold_days'].median():.0f}d")

# ── 5. Trailing-stop trigger profile ──────────────────────────────────
if len(sells) > 0 and "exit_reason" in sells.columns:
    trail = sells[sells["exit_reason"] == "trailing_stop"]
    print(f"\n[5] Trailing-stop exits ({len(trail)}):")
    if len(trail) > 0 and "pnl_pct" in trail.columns:
        print(f"    Median P&L on trailing-stop: {trail['pnl_pct'].median():+.2%}")
        print(f"    P&L percentiles:")
        for p in [10, 25, 50, 75, 90]:
            v = trail["pnl_pct"].quantile(p / 100.0)
            print(f"       p{p:2d}: {v:+.2%}")

# ── 6. stop_loss trigger profile ──────────────────────────────────────
if len(sells) > 0 and "exit_reason" in sells.columns:
    sl = sells[sells["exit_reason"] == "stop_loss"]
    print(f"\n[6] Stop-loss exits ({len(sl)}):")
    if len(sl) > 0 and "pnl_pct" in sl.columns:
        print(f"    Median P&L on stop_loss: {sl['pnl_pct'].median():+.2%}")
        for p in [10, 25, 50, 75, 90]:
            v = sl["pnl_pct"].quantile(p / 100.0)
            print(f"       p{p:2d}: {v:+.2%}")

# ── 7. Per-position max-gain-reached (informs take_profit / trailing) ─
print(f"\n[7] Max gain reached before close (by exit reason):")
# This needs more info than trade_log usually has; check the available columns
if len(sells) > 0:
    print(f"    sells columns available: {list(sells.columns)}")

# Save the dump
out = REPO / "data" / "logs" / "baseline_distributions.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(dump, indent=2, default=str))
print(f"\nWrote sidecar → {out}")
