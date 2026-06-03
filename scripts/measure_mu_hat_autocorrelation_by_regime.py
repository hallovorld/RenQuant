#!/usr/bin/env python
"""Measure μ̂ autocorrelation per regime — closes #125 §8 Step 2 / HIGH-4.

The parent memo of PR #125 originally used realized forward-label
autocorrelation (`label_autocorr_60`) as evidence about signal decay,
which codex correctly flagged as the wrong observable. Multi-period
optimization (cvxportfolio MultiPeriodOpt / Level 2) needs the
*forecast-state* persistence — ``corr(μ̂_t, μ̂_{t+k})`` — not realized
label autocorr.

This script computes the right observable on the historical sim
decision traces stored in ``data/sim_runs.db::score_distribution``,
per regime, for lag k ∈ {1, 5, 10, 20, 60} trading days.

**Outputs** (under ``doc/research/evidence/``):

- ``2026-06-03-mu-hat-autocorrelation-by-regime.json`` — the canonical
  evidence artifact:
  ```
  {
    "as_of_date": "2026-06-03",
    "data_source": "data/sim_runs.db::score_distribution",
    "regimes": {
      "BULL_CALM": {
        "n_rows": 2699, "n_dates": …,
        "mean_autocorr": {"1": …, "5": …, "10": …, "20": …, "60": …},
        "topk_overlap": {
          "5":  {"1": …, "5": …, …},
          "10": {…},
          "20": {…},
        },
        "half_life_days":  …          # smallest k with corr ≤ 0.5
      },
      "BULL_VOLATILE": {…},
      "CHOPPY":        {…},
      "BEAR":          {…}
    }
  }
  ```

The output is the input to the §8 Step 4 offline WF A/B replay's
Level-2 attractiveness question. A high autocorrelation across lags
means MultiPeriodOpt's planning-ahead value is small (the forecast
barely changes bar to bar); a fast decay means Level 2 has more to
optimize over.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("measure-mu-autocorr")

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO / "data" / "sim_runs.db"
DEFAULT_OUT = REPO / "doc" / "research" / "evidence" / "2026-06-03-mu-hat-autocorrelation-by-regime.json"

LAGS = (1, 5, 10, 20, 60)
TOPK_VALUES = (5, 10, 20)
MIN_ROWS_PER_REGIME = 50  # below this, mean autocorr is unreliable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", type=str, default=str(DEFAULT_DB),
                   help="Path to sim_runs.db (default: data/sim_runs.db)")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                   help="Output JSON path")
    p.add_argument("--min-ticker-dates", type=int, default=30,
                   help="Minimum dates a ticker must appear in a regime "
                        "for its per-ticker autocorr to count toward the mean")
    return p.parse_args()


def load_mu_panel(db_path: Path) -> pd.DataFrame:
    """Pull (date, ticker, mu, regime) from score_distribution.

    Returns a long-form DataFrame; mu values are the calibrated
    expected-return forecasts the QP/Hybrid/MPO consume as μ̂.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT date, ticker, mu, regime FROM score_distribution "
        "WHERE mu IS NOT NULL",
        conn,
        parse_dates=["date"],
    )
    conn.close()
    log.info("Loaded %d (date,ticker,mu,regime) rows from %s",
             len(df), db_path)
    return df


def per_ticker_autocorr(series: pd.Series, lag: int) -> Optional[float]:
    """Pearson autocorr of a single ticker's μ̂ series at the given lag.

    ``series`` is indexed by date (gaps allowed; we use shift not lag
    in calendar days). Returns NaN if fewer than ``lag + 5`` valid
    pairs survive.
    """
    if len(series) <= lag + 5:
        return None
    s = series.sort_index()
    paired = pd.concat([s, s.shift(lag)], axis=1).dropna()
    if len(paired) < 5:
        return None
    a = paired.iloc[:, 0].values
    b = paired.iloc[:, 1].values
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def regime_mean_autocorr(
    regime_df: pd.DataFrame,
    lag: int,
    min_ticker_dates: int,
) -> Optional[float]:
    """Mean per-ticker autocorr at the given lag, taken across tickers
    with at least ``min_ticker_dates`` observations in this regime.
    """
    out = []
    for ticker, sub in regime_df.groupby("ticker"):
        if len(sub) < min_ticker_dates:
            continue
        series = sub.set_index("date")["mu"]
        ac = per_ticker_autocorr(series, lag)
        if ac is not None and np.isfinite(ac):
            out.append(ac)
    if not out:
        return None
    return float(np.mean(out))


def topk_overlap_rate(regime_df: pd.DataFrame, K: int, lag: int) -> Optional[float]:
    """Fraction-of-K of names that survive K-day-by-K-day re-ranking.

    At each date, take the top-K tickers by μ̂; advance lag days; take
    the top-K again. The overlap fraction is |intersection| / K.
    Returns the mean across (date, date+lag) pairs with valid K.
    """
    by_date = regime_df.groupby("date")["mu"].apply(
        lambda x: regime_df.loc[x.index].set_index("ticker")["mu"]
    )
    # Re-pivot for cleaner access
    pivot = regime_df.pivot_table(
        index="date", columns="ticker", values="mu", aggfunc="mean"
    )
    dates = pivot.index.sort_values()
    if len(dates) <= lag:
        return None
    overlaps = []
    for i in range(len(dates) - lag):
        t1 = dates[i]
        t2 = dates[i + lag]
        row1 = pivot.loc[t1].dropna()
        row2 = pivot.loc[t2].dropna()
        if len(row1) < K or len(row2) < K:
            continue
        top1 = set(row1.nlargest(K).index)
        top2 = set(row2.nlargest(K).index)
        overlaps.append(len(top1 & top2) / K)
    if not overlaps:
        return None
    return float(np.mean(overlaps))


def half_life(mean_autocorr: dict) -> Optional[int]:
    """Smallest lag k for which mean autocorr ≤ 0.5.

    Returns None if autocorr never crosses 0.5 within the measured lag
    range, or if lag 1 is already below 0.5.
    """
    sorted_items = sorted(
        ((int(k), v) for k, v in mean_autocorr.items() if v is not None),
        key=lambda kv: kv[0],
    )
    if not sorted_items:
        return None
    for k, v in sorted_items:
        if v <= 0.5:
            return k
    return None


def measure_regime(regime: str, regime_df: pd.DataFrame, min_ticker_dates: int) -> dict:
    """Build the per-regime evidence block."""
    n_rows = len(regime_df)
    n_dates = regime_df["date"].nunique()
    n_tickers = regime_df["ticker"].nunique()

    if n_rows < MIN_ROWS_PER_REGIME:
        log.warning(
            "Regime %s only has %d rows (< %d); autocorr estimates "
            "below will be unreliable. Reporting them anyway with the "
            "'undersampled' flag.",
            regime, n_rows, MIN_ROWS_PER_REGIME,
        )

    mean_autocorr = {
        str(lag): regime_mean_autocorr(regime_df, lag, min_ticker_dates)
        for lag in LAGS
    }
    topk_overlap = {
        str(K): {
            str(lag): topk_overlap_rate(regime_df, K, lag)
            for lag in LAGS
        }
        for K in TOPK_VALUES
    }
    hl = half_life({k: v for k, v in mean_autocorr.items()})

    return {
        "n_rows": int(n_rows),
        "n_dates": int(n_dates),
        "n_tickers": int(n_tickers),
        "undersampled": bool(n_rows < MIN_ROWS_PER_REGIME),
        "mean_autocorr": mean_autocorr,
        "topk_overlap": topk_overlap,
        "half_life_days": hl,
    }


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    out_path = Path(args.out)

    df = load_mu_panel(db_path)
    if df.empty:
        raise SystemExit(
            f"No (date,ticker,mu,regime) rows in {db_path}::score_distribution"
        )

    payload = {
        "as_of_date": "2026-06-03",
        "data_source": f"{db_path.relative_to(REPO)}::score_distribution",
        "n_total_rows": int(len(df)),
        "date_range": [
            df["date"].min().date().isoformat(),
            df["date"].max().date().isoformat(),
        ],
        "n_unique_dates": int(df["date"].nunique()),
        "n_unique_tickers": int(df["ticker"].nunique()),
        "lags_measured_trading_days": list(LAGS),
        "topk_values_measured": list(TOPK_VALUES),
        "min_ticker_dates_for_autocorr_inclusion": int(args.min_ticker_dates),
        "regimes": {},
        "interpretation_notes": [
            "mean_autocorr is the mean Pearson corr(μ̂_t, μ̂_{t+k}) taken "
            "across tickers that appear on at least min_ticker_dates dates "
            "in the regime. Values close to 1.0 mean the forecast barely "
            "changes from bar to bar (Level-2 MultiPeriodOpt has little "
            "to optimize); values near 0 mean fast forecast decay "
            "(Level-2 has more to optimize).",
            "topk_overlap measures rank-set persistence — useful for "
            "checking whether the TOP K names cycle rapidly even if the "
            "marginal μ̂ value is stable.",
            "half_life_days is the smallest lag at which mean autocorr "
            "first crosses 0.5; null means the autocorr never crosses 0.5 "
            "within the measured lag range.",
            "BULL_CALM dominates the row count (≈78% of bars) per the "
            "current 2024-2025 market regime mix. BEAR and CHOPPY may "
            "be undersampled; the 'undersampled' flag highlights regimes "
            "with < 50 rows where autocorr estimates are unreliable.",
        ],
    }

    for regime in ("BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"):
        regime_df = df[df["regime"] == regime].copy()
        if regime_df.empty:
            payload["regimes"][regime] = {
                "n_rows": 0,
                "skipped": True,
                "reason": "no rows in this regime",
            }
            continue
        payload["regimes"][regime] = measure_regime(
            regime, regime_df, args.min_ticker_dates,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    log.info("Wrote evidence to %s", out_path)

    # Console summary so the user can sanity-check without opening the JSON
    print("\n=== μ̂ autocorrelation per regime (trading-day lags) ===")
    print(f"{'Regime':<14} {'n_rows':>7} " + " ".join(
        f"{'L'+str(lag):>7}" for lag in LAGS
    ) + f"  {'half_life':>10}")
    for regime, block in payload["regimes"].items():
        if block.get("skipped"):
            print(f"{regime:<14} (skipped — no rows)")
            continue
        ac = block["mean_autocorr"]
        row = f"{regime:<14} {block['n_rows']:>7} "
        for lag in LAGS:
            v = ac.get(str(lag))
            row += f"{'-' if v is None else f'{v:+.3f}':>7} "
        hl = block["half_life_days"]
        row += f"  {'(none)' if hl is None else str(hl):>10}"
        if block["undersampled"]:
            row += "  ⚠ undersampled"
        print(row)


if __name__ == "__main__":
    main()
