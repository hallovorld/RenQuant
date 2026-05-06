#!/usr/bin/env python
"""Walk-forward validation of alpha158 + Linear winner.

Per CLAUDE.md §5.2 sanity sequence + E27 walk-forward lesson: single-cut
sim is path-dependent. We test if the +15.8 pt alpha (sim, 10bp cost,
top-10) holds across 3 independent OOS cuts with retrain.

Method:
1. For each cut date C, train Linear on data with date ≤ C - embargo (60d).
2. Sim 6 months [C, C+6mo] using top-K rotation + transaction cost.
3. Compute APY, Sharpe, alpha vs SPY for each cut.
4. Report mean ± σ across cuts → is +15.8 pt alpha real?

Compares directly to wl103 production walk-forward 3-cut (mean alpha
−8.6 pts) since cuts use same date format and label.

Usage::
    python scripts/qlib_linear_walk_forward.py
    python scripts/qlib_linear_walk_forward.py --top-k 10 --cost-bps 10
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
log = logging.getLogger("qlib-walk-forward")


def load_spy_returns(ohlcv_dir: Path) -> pd.Series:
    p = ohlcv_dir / "SPY" / "1d.parquet"
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df["close"].pct_change()


def compute_period_return(ohlcv_dir: Path, top_tickers: list[str],
                           date: pd.Timestamp, hold_days: int) -> float:
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
        if len(idx_dates) < hold_days + 1:
            continue
        try:
            entry = float(df["close"].loc[idx_dates[0]])
            exit  = float(df["close"].loc[idx_dates[hold_days]])
            r = exit / entry - 1.0
            if np.isfinite(r):
                rets.append(r)
        except (KeyError, IndexError):
            continue
    return float(np.mean(rets)) if rets else 0.0


def run_one_cut(panel: pd.DataFrame, feat_cols: list[str], label: str,
                 cut_date: pd.Timestamp, oos_months: int,
                 top_k: int, rebalance_days: int, cost_bps: float,
                 ohlcv_dir: Path, embargo_days: int = 60) -> dict:
    """Train Linear on data ≤ cut - embargo, sim from cut to cut+oos_months."""
    sim_start = cut_date
    sim_end = cut_date + pd.Timedelta(days=oos_months * 30)
    train_end = cut_date - pd.Timedelta(days=embargo_days)

    train = panel[panel["date"] <= train_end].dropna(subset=[label])
    sim   = panel[(panel["date"] >= sim_start) & (panel["date"] <= sim_end)].copy()

    log.info("CUT %s: train=%d  sim=%d  train_end=%s  sim=[%s, %s]",
             cut_date.date(), len(train), len(sim), train_end.date(),
             sim_start.date(), sim_end.date())

    if len(train) < 1000 or len(sim) < 100:
        log.warning("CUT %s: insufficient data (train=%d sim=%d) — skipping",
                    cut_date.date(), len(train), len(sim))
        return {}

    model = LinearRegression(fit_intercept=False, copy_X=False)
    model.fit(train[feat_cols].values, train[label].values)
    sim["score"] = model.predict(sim[feat_cols].values)

    cost_one_way = cost_bps / 10000.0
    sim_dates = sorted(sim["date"].unique())
    rebalance_dates = sim_dates[::rebalance_days]

    period_rets, period_dates, period_turnovers = [], [], []
    prev_tickers: set[str] = set()
    for d in rebalance_dates:
        day_data = sim[sim["date"] == d]
        if len(day_data) < top_k:
            continue
        topk = day_data.nlargest(top_k, "score")["ticker"].tolist()
        new_set = set(topk)
        if not prev_tickers:
            turnover = 1.0
        else:
            sold = len(prev_tickers - new_set) / top_k
            bought = len(new_set - prev_tickers) / top_k
            turnover = (sold + bought) / 2.0
        prev_tickers = new_set
        period_turnovers.append(turnover)
        gross = compute_period_return(ohlcv_dir, topk, d, rebalance_days)
        cost = turnover * 2 * cost_one_way
        period_rets.append(gross - cost)
        period_dates.append(d)

    if not period_rets:
        return {}

    rets_arr = np.asarray(period_rets)
    spy_full = load_spy_returns(ohlcv_dir)
    spy_period_rets = []
    for d in period_dates:
        idx_dates = spy_full.index[spy_full.index >= d]
        if len(idx_dates) < rebalance_days:
            spy_period_rets.append(0.0)
            continue
        cum = float((1 + spy_full.loc[idx_dates[0]: idx_dates[rebalance_days - 1]]).prod() - 1)
        spy_period_rets.append(cum)
    spy_arr = np.asarray(spy_period_rets)

    periods_per_year = 252 / rebalance_days
    strat_apy = float((1 + np.mean(rets_arr)) ** periods_per_year - 1.0)
    spy_apy   = float((1 + np.mean(spy_arr)) ** periods_per_year - 1.0)
    strat_sharpe = float(np.mean(rets_arr) / (np.std(rets_arr) + 1e-12)
                          * np.sqrt(periods_per_year))
    spy_sharpe = float(np.mean(spy_arr) / (np.std(spy_arr) + 1e-12)
                        * np.sqrt(periods_per_year))

    return {
        "cut": str(cut_date.date()),
        "n_periods": len(period_rets),
        "avg_turnover": float(np.mean(period_turnovers)),
        "strat_apy_pct": strat_apy * 100,
        "spy_apy_pct": spy_apy * 100,
        "alpha_vs_spy_pts": (strat_apy - spy_apy) * 100,
        "strat_sharpe": strat_sharpe,
        "spy_sharpe": spy_sharpe,
        "strat_cum_pct": float((np.prod(1 + rets_arr) - 1) * 100),
        "spy_cum_pct": float((np.prod(1 + spy_arr) - 1) * 100),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset",
                   default=str(REPO_ROOT / "data" / "alpha158_qlib_dataset.parquet"))
    p.add_argument("--ohlcv-dir", default=str(REPO_ROOT / "data" / "ohlcv"))
    p.add_argument("--label", default="fwd_5d_excess")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--rebalance-days", type=int, default=5)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--cuts", nargs="+",
                   default=["2024-05-01", "2024-11-01", "2025-05-04"],
                   help="Same cuts as walk_forward_holdout.py for production comparison")
    p.add_argument("--oos-months", type=int, default=6)
    p.add_argument("--output",
                   default=str(REPO_ROOT / "artifacts" / "qlib_linear_walk_forward.json"))
    args = p.parse_args()

    log.info("Loading %s", args.dataset)
    panel = pd.read_parquet(args.dataset)
    excluded = {"ticker", "date", "split_label",
                "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excluded]
    panel["date"] = pd.to_datetime(panel["date"])
    log.info("Features=%d  total rows=%d", len(feat_cols), len(panel))

    cut_dates = [pd.Timestamp(c) for c in args.cuts]
    log.info("Walk-forward: %d cuts × %d-mo OOS, top_k=%d, rebalance=%dd, cost=%.0f bp",
             len(cut_dates), args.oos_months, args.top_k,
             args.rebalance_days, args.cost_bps)

    results = []
    for cut in cut_dates:
        r = run_one_cut(panel, feat_cols, args.label, cut, args.oos_months,
                        args.top_k, args.rebalance_days, args.cost_bps,
                        Path(args.ohlcv_dir))
        if r:
            results.append(r)
            log.info(
                "CUT %s — APY=%+.1f%% Sharpe=%+.2f alpha=%+.1f pts (turnover=%.0f%% N=%d)",
                r["cut"], r["strat_apy_pct"], r["strat_sharpe"],
                r["alpha_vs_spy_pts"], r["avg_turnover"] * 100, r["n_periods"],
            )

    log.info("══ WALK-FORWARD SUMMARY ══")
    log.info("  cuts done:  %d", len(results))
    apys = [r["strat_apy_pct"] for r in results]
    sharpes = [r["strat_sharpe"] for r in results]
    alphas = [r["alpha_vs_spy_pts"] for r in results]
    log.info("  APY        : mean=%+.1f%%  std=%+.1f  per-cut=%s",
             np.mean(apys), np.std(apys),
             [f"{a:+.1f}" for a in apys])
    log.info("  Sharpe     : mean=%+.2f   std=%+.2f  per-cut=%s",
             np.mean(sharpes), np.std(sharpes),
             [f"{s:+.2f}" for s in sharpes])
    log.info("  Alpha pts  : mean=%+.1f   std=%+.1f  per-cut=%s",
             np.mean(alphas), np.std(alphas),
             [f"{a:+.1f}" for a in alphas])
    pos_alpha = sum(1 for a in alphas if a > 0)
    log.info("  N alpha > 0: %d / %d", pos_alpha, len(alphas))
    log.info("  ─── reference ───")
    log.info("  wl103 walk-forward (production XGB): mean alpha −8.6 pts")
    log.info("  Single-cut sim (alpha158 Linear, same cost): +15.8 pts")

    summary = {
        "label": args.label,
        "top_k": args.top_k,
        "rebalance_days": args.rebalance_days,
        "cost_bps": args.cost_bps,
        "n_features": len(feat_cols),
        "cuts": args.cuts,
        "oos_months": args.oos_months,
        "results": results,
        "consistency": {
            "n_cuts": len(results),
            "apy_mean": float(np.mean(apys)),
            "apy_std": float(np.std(apys)),
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_std": float(np.std(sharpes)),
            "alpha_mean": float(np.mean(alphas)),
            "alpha_std": float(np.std(alphas)),
            "n_pos_alpha": pos_alpha,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2, default=str))
    log.info("Summary: %s", args.output)


if __name__ == "__main__":
    main()
