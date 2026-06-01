#!/usr/bin/env python
"""Regime-stratified paired-returns evaluator.

Decomposes the cross-window paired daily Δ analysis by OBJECTIVE
market-regime labels derived from SPY (not the strategy's internal
regime detector, which we've shown is too sticky — labels 95% of days
BULL_CALM regardless of underlying regime).

Stratification dimensions (all data-derived from SPY):
  1. Trend strength = rolling 60d SPY Sharpe = (mean_ret/vol) × √252
     - Thresholds: LOW < 0.5 < MED < 1.5 < HIGH (percentile-based optional)
  2. Volatility regime = 20d realized vol vs 252d history percentile
     - Thresholds: CALM < p33 < NORMAL < p66 < SPIKED

A strategy that wins under e.g. LOW-Sharpe + CALM-vol regimes but
loses in HIGH-Sharpe regimes is a "regime-conditional winner" —
candidate for conditional deployment via regime gating, not blanket
promote.

Reference: Asness-Moskowitz-Pedersen 2013 "Value and Momentum
Everywhere" *J. Finance* 68(3):929 — factor returns are
regime-dependent; conditional analysis reveals the structure.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from renquant_common.metrics.hac_se import hac_t_stat  # noqa: E402
from renquant_common.metrics.block_bootstrap import stationary_bootstrap_ci  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("eval-regime")


def load_spy(spy_path: Path) -> pd.DataFrame:
    spy = pd.read_parquet(spy_path)
    spy.index = pd.to_datetime(spy.index)
    spy = spy.sort_index()
    return spy


def compute_regime_labels(spy: pd.DataFrame,
                          trend_window: int = 60,
                          vol_window: int = 20,
                          vol_hist_window: int = 252) -> pd.DataFrame:
    """Per-day SPY-derived regime labels.

    Columns added:
      - spy_ret_60d_sharpe : trend strength (annualized)
      - spy_vol_20d        : 20-day annualized realized vol
      - spy_vol_pct        : percentile of 20d vol over 252d history
      - trend_label        : 'LOW' / 'MED' / 'HIGH'
      - vol_label          : 'CALM' / 'NORMAL' / 'SPIKED'
      - regime             : combined "TREND_VOL" label
    """
    df = spy.copy()
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df["roll_mean"] = df["ret"].rolling(trend_window, min_periods=trend_window // 2).mean()
    df["roll_vol"]  = df["ret"].rolling(trend_window, min_periods=trend_window // 2).std(ddof=1)
    df["spy_ret_60d_sharpe"] = (df["roll_mean"] / df["roll_vol"]) * math.sqrt(252.0)
    df["spy_vol_20d"] = df["ret"].rolling(vol_window, min_periods=vol_window // 2).std(ddof=1) * math.sqrt(252.0)
    # Vol percentile (rolling) over 252-day history
    def _pct_rank(s):
        if len(s) < 30 or pd.isna(s.iloc[-1]):
            return np.nan
        last = s.iloc[-1]
        return float((s < last).sum()) / float(len(s) - 1)
    df["spy_vol_pct"] = df["spy_vol_20d"].rolling(vol_hist_window, min_periods=30).apply(_pct_rank, raw=False)
    # Labels
    df["trend_label"] = pd.cut(
        df["spy_ret_60d_sharpe"],
        bins=[-np.inf, 0.5, 1.5, np.inf],
        labels=["LOW", "MED", "HIGH"],
    )
    df["vol_label"] = pd.cut(
        df["spy_vol_pct"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["CALM", "NORMAL", "SPIKED"],
    )
    df["regime"] = df["trend_label"].astype(str) + "_" + df["vol_label"].astype(str)
    return df[["spy_ret_60d_sharpe", "spy_vol_20d", "spy_vol_pct",
               "trend_label", "vol_label", "regime"]]


def daily_paired_delta(baseline_dir: Path, candidate_dir: Path) -> pd.Series:
    """Stitch per-window paired daily Δ into a single date-indexed series."""
    bfiles = sorted(baseline_dir.glob("*.json"))
    parts = []
    for bf in bfiles:
        cf = candidate_dir / bf.name
        if not cf.exists():
            continue
        b = json.loads(bf.read_text())
        c = json.loads(cf.read_text())
        b_eq = pd.Series(b["equity"]).astype(float)
        c_eq = pd.Series(c["equity"]).astype(float)
        b_eq.index = pd.to_datetime(b_eq.index)
        c_eq.index = pd.to_datetime(c_eq.index)
        common = b_eq.index.intersection(c_eq.index)
        b_ret = np.log(b_eq.loc[common] / b_eq.loc[common].shift(1)).dropna()
        c_ret = np.log(c_eq.loc[common] / c_eq.loc[common].shift(1)).dropna()
        d = (c_ret - b_ret).dropna()
        parts.append(d)
    return pd.concat(parts).sort_index()


def stratified_analysis(delta: pd.Series, regimes: pd.DataFrame) -> pd.DataFrame:
    """Per-regime paired-Δ HAC t-stat + bootstrap CI + Cohen's d."""
    df = pd.DataFrame({"d": delta}).join(regimes, how="left")
    df = df.dropna(subset=["regime"])
    rows = []
    for regime, g in df.groupby("regime", observed=True):
        d = g["d"].values
        n = len(d)
        if n < 8:
            rows.append({"regime": regime, "n": n, "skipped": True})
            continue
        nw = hac_t_stat(d)
        # Bootstrap CI on mean Δ
        bs = stationary_bootstrap_ci(d, B=1000)
        sigma = float(np.std(d, ddof=1))
        cohens_d = nw["mean"] / sigma if sigma > 0 else 0.0
        rows.append({
            "regime":         regime,
            "n":              int(n),
            "mean_d_ann":     float(nw["mean"] * 252.0),
            "t_stat":         float(nw["t_stat"]),
            "p_value":        float(nw["p_value"]),
            "ci95_lo_ann":    float(bs["ci_lo"] * 252.0),
            "ci95_hi_ann":    float(bs["ci_hi"] * 252.0),
            "cohens_d":       float(cohens_d),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--candidate-dir", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--spy-path", default="data/ohlcv/SPY/1d.parquet")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    spy = load_spy(REPO / args.spy_path)
    regimes = compute_regime_labels(spy)
    log.info("Regimes computed for %d SPY days", len(regimes))
    delta = daily_paired_delta(Path(args.baseline_dir), Path(args.candidate_dir))
    log.info("Paired Δ series: %d daily observations", len(delta))
    name = args.name or f"{Path(args.candidate_dir).name} vs {Path(args.baseline_dir).name}"
    print(f"\n=== Regime-Stratified Paired-Returns — {name} ===")
    print(f"Pooled n_days = {len(delta)}")
    print()
    df = stratified_analysis(delta, regimes)
    print(f"{'Regime':18} {'n':>5} {'meanΔ_ann':>11} {'t':>6} {'p':>8} "
          f"{'CI95_lo':>9} {'CI95_hi':>9} {'d':>6}")
    print("-" * 80)
    for _, r in df.iterrows():
        # NaN is truthy in Python, so r.get('skipped') == NaN passes 'if';
        # check explicitly for True.
        if r.get("skipped") is True:
            print(f"{r['regime']:18} {r['n']:>5d}  (skipped, n<8)")
            continue
        print(f"{r['regime']:18} {r['n']:>5d} "
              f"{r['mean_d_ann']*100:>+10.2f}% {r['t_stat']:>+6.2f} "
              f"{r['p_value']:>8.4f} {r['ci95_lo_ann']*100:>+8.2f}% "
              f"{r['ci95_hi_ann']*100:>+8.2f}% {r['cohens_d']:>+6.2f}")
    print()
    # Identify conditional winners (t > 1.5 in some regime)
    winners = df[(df.get("t_stat", 0) > 1.5) & (df.get("ci95_lo_ann", 0) > 0)] if not df.empty else df
    if not winners.empty:
        print("CONDITIONAL-WIN REGIMES (t > 1.5 AND CI95_lo > 0):")
        for _, r in winners.iterrows():
            print(f"  {r['regime']}: t={r['t_stat']:+.2f}, "
                  f"meanΔ={r['mean_d_ann']*100:+.2f}%, "
                  f"CI[{r['ci95_lo_ann']*100:+.2f}%, {r['ci95_hi_ann']*100:+.2f}%]")
    else:
        print("No regime shows significant conditional win.")
    if args.json_out:
        Path(args.json_out).write_text(df.to_json(orient="records", indent=2))
        log.info(f"JSON report → {args.json_out}")


if __name__ == "__main__":
    main()
