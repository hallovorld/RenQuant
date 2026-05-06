#!/usr/bin/env python
"""Simple long-K rotating sim of Qlib Linear + alpha158 winner.

Tests the question: does +0.038 test_median IC translate to real APY?

Method (faithful to Qlib's standard backtest pattern, simpler than
cvxportfolio's MultiPeriodOptimization which adds transaction cost,
risk-aversion, etc.):

1. Train sklearn LinearRegression(fit_intercept=False) on TRAIN split
   features → predict labels.
2. For each TEST date, rank the per-ticker prediction scores.
3. Long the top K (default 10), equal-weighted.
4. Hold for `holding_days` then rebalance.
5. Per-period return = mean of held tickers' fwd_<horizon>d_excess.
6. SPY benchmark = same dates' SPY actual return; alpha = strat - SPY.

Reference: Qlib's `qlib.contrib.evaluate.long_short_backtest()`.
We do long-only here (long-short is more aggressive).

Usage::

    # Default: long top 10, fwd_5d horizon, daily rebalance
    python scripts/qlib_linear_sim.py

    # Longer horizon (matches winner config from E30)
    python scripts/qlib_linear_sim.py --label fwd_20d_excess --top-k 10

Output: artifacts/qlib_linear_sim_<label>.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("qlib-linear-sim")


def load_spy_returns(ohlcv_dir: Path,
                     start: pd.Timestamp,
                     end: pd.Timestamp) -> pd.Series:
    """Read SPY daily returns over [start, end]."""
    p = ohlcv_dir / "SPY" / "1d.parquet"
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    rets = df["close"].pct_change()
    return rets.loc[(rets.index >= start) & (rets.index <= end)]


def compute_period_return(panel: pd.DataFrame, ohlcv_dir: Path,
                           top_tickers: list[str],
                           date: pd.Timestamp,
                           horizon_days: int) -> float:
    """Compute the realized return of an equal-weight portfolio of
    `top_tickers` held from `date` to `date + horizon_days`.

    We re-read raw OHLCV instead of using the dataset's pre-computed
    label so we can evaluate any horizon (not just fwd_5d/20d/60d)."""
    rets = []
    for t in top_tickers:
        p = ohlcv_dir / t / "1d.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=["close"])
            df.index = pd.to_datetime(df.index)
        except Exception:
            continue
        idx_dates = df.index[df.index >= date]
        if len(idx_dates) < horizon_days + 1:
            continue
        try:
            entry_close = float(df["close"].loc[idx_dates[0]])
            exit_close  = float(df["close"].loc[idx_dates[horizon_days]])
            r = exit_close / entry_close - 1.0
            if np.isfinite(r):
                rets.append(r)
        except (KeyError, IndexError):
            continue
    return float(np.mean(rets)) if rets else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset",
                   default=str(REPO_ROOT / "data" / "alpha158_qlib_dataset.parquet"))
    p.add_argument("--ohlcv-dir", default=str(REPO_ROOT / "data" / "ohlcv"))
    p.add_argument("--label", default="fwd_5d_excess",
                   choices=["fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"])
    p.add_argument("--top-k", type=int, default=10,
                   help="Long top-K ranked tickers per rebalance")
    p.add_argument("--rebalance-days", type=int, default=5,
                   help="Days between rebalances. Should match label horizon "
                        "for cleanest mapping.")
    p.add_argument("--cost-bps", type=float, default=0.0,
                   help="One-way transaction cost in basis points (1 bp = 0.01%%). "
                        "Round-trip cost = 2 × this. Typical retail: 5-10 bp.")
    p.add_argument("--output",
                   default=str(REPO_ROOT / "artifacts" / "qlib_linear_sim.json"))
    args = p.parse_args()

    log.info("Loading %s", args.dataset)
    panel = pd.read_parquet(args.dataset)
    excluded = {"ticker", "date", "split_label",
                "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excluded]
    log.info("Features: %d", len(feat_cols))

    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.dropna(subset=[args.label])
    train = panel[panel["split_label"] == "train"]
    test  = panel[panel["split_label"] == "test"]
    log.info("Train: %d  Test: %d  N test dates: %d",
             len(train), len(test), test["date"].nunique())

    log.info("Fitting LinearRegression on %d × %d …",
             len(train), len(feat_cols))
    model = LinearRegression(fit_intercept=False, copy_X=False)
    model.fit(train[feat_cols].values, train[args.label].values)

    test = test.copy()
    test["score"] = model.predict(test[feat_cols].values)

    # Map label → horizon in trading days
    horizon = {"fwd_5d_excess": 5, "fwd_20d_excess": 20, "fwd_60d_excess": 60}[args.label]
    log.info("Strategy: long top %d, rebalance every %d days, horizon=%d",
             args.top_k, args.rebalance_days, horizon)

    # Iterate test dates, take every rebalance_days-th
    test_dates = sorted(test["date"].unique())
    rebalance_dates = test_dates[::args.rebalance_days]
    log.info("Rebalance: %d dates over [%s, %s]",
             len(rebalance_dates), rebalance_dates[0], rebalance_dates[-1])

    ohlcv_dir = Path(args.ohlcv_dir)
    period_rets: list[float] = []
    period_dates: list[pd.Timestamp] = []
    period_turnovers: list[float] = []
    selected_tickers_log: list[list[str]] = []

    cost_one_way = args.cost_bps / 10000.0  # bps → fraction
    prev_tickers: set[str] = set()
    for d in rebalance_dates:
        day_data = test[test["date"] == d]
        if len(day_data) < args.top_k:
            continue
        # Top K by score
        topk_rows = day_data.nlargest(args.top_k, "score")
        topk_tickers = topk_rows["ticker"].tolist()
        selected_tickers_log.append(topk_tickers)
        # Turnover = fraction of portfolio rebalanced (each ticker = 1/K weight)
        new_set = set(topk_tickers)
        if not prev_tickers:
            # Initial buy — full turnover
            turnover = 1.0
        else:
            # Turnover = sum of |Δw|/2 for each name; equal-weight 1/K each
            sold_fraction = len(prev_tickers - new_set) / args.top_k
            bought_fraction = len(new_set - prev_tickers) / args.top_k
            turnover = (sold_fraction + bought_fraction) / 2.0
        prev_tickers = new_set
        period_turnovers.append(turnover)
        # Realized return over `rebalance_days`
        gross = compute_period_return(panel, ohlcv_dir,
                                       topk_tickers, d, args.rebalance_days)
        # Subtract transaction cost: turnover × 2 (round-trip) × cost_one_way
        cost = turnover * 2 * cost_one_way
        net = gross - cost
        period_rets.append(net)
        period_dates.append(d)

    rets_arr = np.asarray(period_rets)
    n_periods = len(rets_arr)
    log.info("══ %d rebalance periods executed ══", n_periods)

    # SPY benchmark (matched periods)
    spy_full = load_spy_returns(ohlcv_dir,
                                 period_dates[0], period_dates[-1] + pd.Timedelta(days=horizon * 2))
    spy_period_rets = []
    for d in period_dates:
        idx_dates = spy_full.index[spy_full.index >= d]
        if len(idx_dates) < args.rebalance_days:
            spy_period_rets.append(0.0)
            continue
        # Compound rebalance_days SPY returns
        period_window = spy_full.loc[idx_dates[0]: idx_dates[args.rebalance_days - 1]]
        cum = (1.0 + period_window).prod() - 1.0
        spy_period_rets.append(float(cum))
    spy_arr = np.asarray(spy_period_rets)

    # Annualized stats
    periods_per_year = 252 / args.rebalance_days
    strat_apy = float((1.0 + np.mean(rets_arr)) ** periods_per_year - 1.0)
    spy_apy   = float((1.0 + np.mean(spy_arr)) ** periods_per_year - 1.0)
    strat_sharpe = float((np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(periods_per_year))
                          if np.std(rets_arr) > 1e-12 else 0.0)
    spy_sharpe = float((np.mean(spy_arr) / np.std(spy_arr) * np.sqrt(periods_per_year))
                        if np.std(spy_arr) > 1e-12 else 0.0)
    alpha_pts = (strat_apy - spy_apy) * 100

    avg_turnover = float(np.mean(period_turnovers)) if period_turnovers else 0.0
    log.info("══ SIM RESULTS (label=%s, top_k=%d, rebalance=%dd, cost=%.1f bp) ══",
             args.label, args.top_k, args.rebalance_days, args.cost_bps)
    log.info("  N periods      = %d", n_periods)
    log.info("  Avg turnover   = %.1f%% per rebalance", avg_turnover * 100)
    log.info("  Strategy APY   = %+.2f%% (net of cost)", strat_apy * 100)
    log.info("  Strategy Sharpe= %+.3f", strat_sharpe)
    log.info("  SPY APY        = %+.2f%%", spy_apy * 100)
    log.info("  SPY Sharpe     = %+.3f", spy_sharpe)
    log.info("  Alpha vs SPY   = %+.2f pts", alpha_pts)
    log.info("  Total cum. ret = %+.2f%%", (np.prod(1 + rets_arr) - 1) * 100)
    log.info("  SPY cum.   ret = %+.2f%%", (np.prod(1 + spy_arr) - 1) * 100)

    summary = {
        "label": args.label,
        "horizon_days": horizon,
        "top_k": args.top_k,
        "rebalance_days": args.rebalance_days,
        "cost_bps": args.cost_bps,
        "avg_turnover": avg_turnover,
        "n_periods": n_periods,
        "strat_apy_pct": strat_apy * 100,
        "strat_sharpe": strat_sharpe,
        "spy_apy_pct": spy_apy * 100,
        "spy_sharpe": spy_sharpe,
        "alpha_vs_spy_pts": alpha_pts,
        "total_cum_return_pct": float((np.prod(1 + rets_arr) - 1) * 100),
        "spy_cum_return_pct": float((np.prod(1 + spy_arr) - 1) * 100),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2, default=str))
    log.info("Summary: %s", args.output)


if __name__ == "__main__":
    main()
