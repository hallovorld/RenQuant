#!/usr/bin/env python
"""Analyze per-trade detail dumped by holdout_backtest.py to find APY/Sharpe
root cause.

Reads:
  data/holdout_results/<train_end>.json            (aggregate)
  data/holdout_results/<train_end>.trades.parquet  (per-trade detail)

Produces a multi-section report:
  1. Distribution: pnl_pct histogram, win/loss split
  2. Exit-reason attribution (which exits drive P&L)
  3. Hold-time: winners vs losers
  4. Tax efficiency: gross vs after-tax
  5. Per-ticker P&L (top contributors + worst)
  6. Entry-side predictive validity (rank_score / mu / sigma at entry vs realized P&L)
  7. Cumulative P&L time series — when did the strategy bleed

Usage::
    python scripts/analyze_holdout_trades.py
    python scripts/analyze_holdout_trades.py --train-end 2024-12-31
    python scripts/analyze_holdout_trades.py --trades data/holdout_results/2024-12-31.trades.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def _safe_pct(num: float, denom: float) -> float:
    return (num / denom * 100) if denom else float("nan")


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def analyze(trades_path: Path, json_path: Path | None) -> None:
    df = pd.read_parquet(trades_path)
    print(f"Loaded {len(df)} trade events from {trades_path.name}")

    aggregate = {}
    if json_path and json_path.exists():
        aggregate = json.loads(json_path.read_text())
        print(f"Aggregate metrics from {json_path.name}:")
        for k in ("apy_holdout", "sharpe_holdout", "sortino_holdout",
                  "max_dd_holdout", "ann_vol_holdout", "win_rate_holdout",
                  "total_return_holdout", "n_buys", "n_sells"):
            if k in aggregate:
                print(f"  {k:<25} {aggregate[k]}")

    buys = df[df["action"] == "buy"].copy()
    sells = df[df["action"] == "sell"].copy()
    print(f"\n  buys = {len(buys)}, sells = {len(sells)}")

    # ── Section 1: pnl_pct distribution ────────────────────────────────────
    section("1. P&L per Closed Trade (sells only)")
    if "pnl_pct" not in sells.columns or sells["pnl_pct"].dropna().empty:
        print("  no pnl_pct on sells — sim didn't tag exits with P&L")
    else:
        s = sells["pnl_pct"].dropna()
        wins = s[s > 0]
        losses = s[s <= 0]
        print(f"  count             {len(s)}")
        print(f"  wins              {len(wins)}  ({_safe_pct(len(wins), len(s)):.1f}%)")
        print(f"  losses            {len(losses)}  ({_safe_pct(len(losses), len(s)):.1f}%)")
        print(f"  mean pnl_pct      {s.mean():+.4%}")
        print(f"  median pnl_pct    {s.median():+.4%}")
        print(f"  std pnl_pct       {s.std():.4%}")
        print(f"  worst             {s.min():+.4%}")
        print(f"  best              {s.max():+.4%}")
        if len(wins):
            print(f"  avg win           {wins.mean():+.4%}  median {wins.median():+.4%}")
        if len(losses):
            print(f"  avg loss          {losses.mean():+.4%}  median {losses.median():+.4%}")
        print(f"  win/loss ratio    "
              f"{abs(wins.mean()/losses.mean()) if len(wins) and len(losses) and losses.mean() != 0 else float('nan'):.2f}")
        # Histogram (text)
        print("\n  pnl_pct histogram (in %):")
        bins = [-1.0, -0.20, -0.10, -0.05, -0.02, 0.0,
                0.02, 0.05, 0.10, 0.20, 1.0]
        h, _ = np.histogram(s, bins=bins)
        for i, c in enumerate(h):
            lo, hi = bins[i] * 100, bins[i + 1] * 100
            bar = "█" * int(c * 50 / max(h.max(), 1))
            print(f"    [{lo:>+6.1f}% .. {hi:>+6.1f}%] {c:>4d} {bar}")

    # ── Section 2: exit-reason attribution ─────────────────────────────────
    section("2. Exit-Reason Attribution (gross P&L by reason)")
    if "exit_reason" in sells.columns and "pnl_pct" in sells.columns:
        # Position-size weighted P&L per reason. Use invest if available; else equal.
        invest_col = "invest" if "invest" in buys.columns else None
        # Match buys to sells by ticker + chronology; approximate position $ by avg buy.
        sells_by = sells.groupby("exit_reason")
        print(f"  {'exit_reason':<22}{'n':>6}{'mean_pnl%':>14}"
              f"{'sum_pnl%':>14}{'med_hold_d':>14}")
        print("  " + "-" * 68)
        for reason, g in sells_by:
            n = len(g)
            mean_p = g["pnl_pct"].mean()
            sum_p = g["pnl_pct"].sum()
            hold = g["hold_days"].median() if "hold_days" in g.columns else float("nan")
            print(f"  {reason:<22}{n:>6}{mean_p:>+13.4%}{sum_p:>+13.4%}{hold:>14.1f}")

    # ── Section 3: hold-time, winners vs losers ────────────────────────────
    section("3. Hold-Time — Winners vs Losers")
    if "hold_days" in sells.columns and "pnl_pct" in sells.columns:
        s2 = sells.dropna(subset=["pnl_pct", "hold_days"])
        wins = s2[s2["pnl_pct"] > 0]
        losses = s2[s2["pnl_pct"] <= 0]
        if len(wins):
            print(f"  Winners ({len(wins)}):  hold mean={wins['hold_days'].mean():.1f}d "
                  f"median={wins['hold_days'].median():.1f}d "
                  f"max={wins['hold_days'].max():.0f}d")
        if len(losses):
            print(f"  Losers  ({len(losses)}):  hold mean={losses['hold_days'].mean():.1f}d "
                  f"median={losses['hold_days'].median():.1f}d "
                  f"max={losses['hold_days'].max():.0f}d")

    # ── Section 4: tax efficiency ─────────────────────────────────────────
    section("4. Tax Efficiency")
    if "tax" in sells.columns:
        gross_pos_pnl = sells.loc[sells["pnl_pct"] > 0, "pnl_pct"].sum()  # in pct units
        total_tax_dollars = sells["tax"].sum()
        print(f"  total_tax_dollars      ${total_tax_dollars:,.2f}")
        print(f"  median_tax_per_sell    ${sells['tax'].median():,.2f}")
        # Trades with tax > 0 are realized gains
        taxed = sells[sells["tax"] > 0]
        untaxed = sells[sells["tax"] <= 0]
        print(f"  trades with tax > 0    {len(taxed)}  "
              f"({_safe_pct(len(taxed), len(sells)):.1f}%)")
        print(f"  trades w/ no tax       {len(untaxed)}  (losses or breakeven)")
        if len(taxed):
            print(f"  avg tax / taxed-trade  ${taxed['tax'].mean():,.2f}")
        # Hold-time at sell
        if "hold_days" in sells.columns and len(taxed):
            lt = taxed[taxed["hold_days"] >= 365]
            st = taxed[taxed["hold_days"] < 365]
            print(f"  long-term-rate sells   {len(lt)}  "
                  f"({_safe_pct(len(lt), len(taxed)):.1f}% of taxed)")
            print(f"  short-term-rate sells  {len(st)}  "
                  f"({_safe_pct(len(st), len(taxed)):.1f}% of taxed) ← short-term tax drag")

    # ── Section 5: per-ticker contribution ─────────────────────────────────
    section("5. Per-Ticker P&L Contribution")
    if "pnl_pct" in sells.columns and "invest" in buys.columns:
        # rough $ pnl using buys' invest matched to sells per ticker
        per_ticker = sells.groupby("ticker")["pnl_pct"].agg(["count", "sum", "mean"])
        per_ticker = per_ticker.sort_values("sum", ascending=False)
        print(f"  TOP 10 contributors:")
        print(per_ticker.head(10).to_string())
        print(f"\n  WORST 10 contributors:")
        print(per_ticker.tail(10).to_string())

    # ── Section 6: predictive validity at entry ────────────────────────────
    section("6. Entry-Side Predictive Validity")
    # buys carry rank_score, mu, sigma; sells carry pnl_pct.
    # Match by ticker chronology — naive: pair k-th buy with k-th sell of same ticker.
    if all(c in buys.columns for c in ("rank_score", "mu", "sigma")) \
            and "pnl_pct" in sells.columns:
        from collections import defaultdict
        per_t_buys: dict = defaultdict(list)
        per_t_sells: dict = defaultdict(list)
        for _, r in buys.iterrows():
            per_t_buys[r["ticker"]].append(r)
        for _, r in sells.iterrows():
            per_t_sells[r["ticker"]].append(r)

        rows = []
        for t, bs in per_t_buys.items():
            ss = per_t_sells.get(t, [])
            for i, sale in enumerate(ss):
                if i < len(bs):
                    b = bs[i]
                    rows.append({
                        "ticker":     t,
                        "rank_entry": b.get("rank_score"),
                        "mu_entry":   b.get("mu"),
                        "sigma_entry": b.get("sigma"),
                        "pnl_pct":    sale.get("pnl_pct"),
                        "hold_days":  sale.get("hold_days"),
                    })
        paired = pd.DataFrame(rows).dropna(subset=["pnl_pct", "rank_entry"])
        if len(paired) >= 5:
            from scipy.stats import spearmanr
            r_rank, _ = spearmanr(paired["rank_entry"], paired["pnl_pct"])
            r_mu, _ = spearmanr(paired["mu_entry"].fillna(0), paired["pnl_pct"])
            r_sigma, _ = spearmanr(paired["sigma_entry"].fillna(0), paired["pnl_pct"])
            print(f"  paired trades = {len(paired)}")
            print(f"  Spearman(rank_entry, pnl_pct)  = {r_rank:+.4f}  "
                  f"(>0 means model's high-rank picks won → predictive)")
            print(f"  Spearman(mu_entry,   pnl_pct)  = {r_mu:+.4f}")
            print(f"  Spearman(sigma_entry, pnl_pct) = {r_sigma:+.4f}")
            # Quintile bucket
            try:
                paired["rank_q"] = pd.qcut(
                    paired["rank_entry"], 5, labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"],
                    duplicates="drop",
                )
                print("\n  pnl by rank_score quintile (Q5_high = model's most-bullish):")
                print(paired.groupby("rank_q")["pnl_pct"].agg(["count", "mean", "median"])
                      .to_string())
            except ValueError:
                pass
        else:
            print(f"  too few paired trades ({len(paired)}) for predictive analysis")

    # ── Section 7: SPY buy-and-hold benchmark ──────────────────────────────
    section("7. vs SPY Buy-and-Hold (same window)")
    if aggregate:
        sim_start = aggregate.get("sim_start")
        sim_end = aggregate.get("sim_end")
        if sim_start and sim_end:
            spy_apy, spy_sharpe, spy_dd, spy_total = _compute_spy_benchmark(
                sim_start, sim_end,
            )
            strat_apy = aggregate.get("apy_holdout", float("nan"))
            strat_sharpe = aggregate.get("sharpe_holdout", float("nan"))
            strat_dd = aggregate.get("max_dd_holdout", float("nan"))
            strat_total = aggregate.get("total_return_holdout", float("nan"))

            print(f"  Window: {sim_start} → {sim_end}")
            print()
            print(f"  {'metric':<25}{'strategy':>14}{'SPY B&H':>14}{'Δ (strat - SPY)':>20}")
            print("  " + "-" * 73)
            def _row(label, s, b, scale="%"):
                if s != s or b != b:
                    sig_s = "—" if s != s else f"{s:.2f}{scale}"
                    sig_b = "—" if b != b else f"{b:.2f}{scale}"
                    return f"  {label:<25}{sig_s:>14}{sig_b:>14}{'—':>20}"
                d = s - b
                marker = " ✓" if d > 0 else (" ✗" if d < 0 else "")
                return f"  {label:<25}{s:>+13.2f}{scale}{b:>+13.2f}{scale}{d:>+18.2f}{scale}{marker}"
            print(_row("APY", strat_apy, spy_apy * 100))
            print(_row("Sharpe", strat_sharpe, spy_sharpe, scale=""))
            print(_row("Max DD", strat_dd, spy_dd * 100))
            print(_row("Total return", strat_total, spy_total * 100))
            print()
            if strat_apy < spy_apy * 100:
                lag = spy_apy * 100 - strat_apy
                print(f"  ⚠️  Strategy LAGS SPY by {lag:+.2f}%/yr — passive benchmark wins.")
            else:
                lead = strat_apy - spy_apy * 100
                print(f"  Strategy LEADS SPY by {lead:+.2f}%/yr.")
        else:
            print("  No sim_start/sim_end in aggregate JSON — can't pull SPY benchmark.")
    else:
        print("  No aggregate JSON loaded — can't compare to SPY.")

    section("DONE")


def _compute_spy_benchmark(start_iso: str, end_iso: str) -> tuple:
    """Pull SPY OHLCV over [start, end], compute APY/Sharpe/MaxDD/total return.

    Returns (apy, sharpe, max_dd, total_return) — all decimals (0.10 = 10%).
    """
    import sys as _sys
    REPO = Path(__file__).resolve().parent.parent
    if str(REPO / "backtesting" / "renquant_104") not in _sys.path:
        _sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))
    try:
        from kernel.data import fetch_ohlcv  # noqa: PLC0415
        from kernel.risk_metrics import sharpe_ratio, max_drawdown  # noqa: PLC0415
        df = fetch_ohlcv("SPY")
        df = df.loc[start_iso:end_iso]
        if df.empty or len(df) < 5:
            return float("nan"), float("nan"), float("nan"), float("nan")
        close = df["close"].astype(float)
        rets = close.pct_change().dropna()
        n_days = len(rets)
        total = float(close.iloc[-1] / close.iloc[0] - 1.0)
        years = n_days / 252.0
        apy = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
        sh = sharpe_ratio(rets)
        dd = max_drawdown(close)
        return apy, sh, dd, total
    except Exception as exc:
        import logging
        logging.getLogger("analyze").warning("SPY benchmark fetch failed: %s", exc)
        return float("nan"), float("nan"), float("nan"), float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--trades", default=None,
                   help="Override trades parquet path.")
    p.add_argument("--json", default=None,
                   help="Override aggregate JSON path.")
    args = p.parse_args()

    if args.trades:
        trades_path = Path(args.trades)
    else:
        trades_path = REPO_ROOT / "data" / "holdout_results" / f"{args.train_end}.trades.parquet"
    if args.json:
        json_path = Path(args.json)
    else:
        json_path = REPO_ROOT / "data" / "holdout_results" / f"{args.train_end}.json"

    if not trades_path.exists():
        print(f"ERROR: trades parquet not found at {trades_path}")
        print("Run holdout_backtest.py first (or check the --train-end).")
        sys.exit(1)

    analyze(trades_path, json_path if json_path.exists() else None)


if __name__ == "__main__":
    main()
