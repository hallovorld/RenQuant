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
per regime, for lag k ∈ {1, 5, 10, 20, 60} **global trading days**.

**Trading-day lag, not observation lag** (#128 review fix). The first
version of this script used ``pd.Series.shift(lag)`` on each ticker's
sparse in-regime series — that shifts by the *k-th later observation*,
which on this DB averaged a 37-trading-day gap at nominal L=20. The
fix is to build a single global trading-date index (sorted unique
dates across the FULL panel, all regimes) and reindex each ticker's
regime-restricted series against it before shifting. Now L=5 means
exactly 5 trading days, period.

**Regime stratification** (#128 review): we stratify by regime AT t
(the bar where the decision is made). The persistence we care about
is "the forecast made at t in regime R, how stable is it k days
later regardless of the regime at t+k". A "regime stays R from t to
t+k" alternative would shrink the sample and answers a different
question.

**Outputs** (under ``doc/research/evidence/``):

- ``2026-06-03-mu-hat-autocorrelation-by-regime.json`` — the canonical
  evidence artifact. Includes per-regime ``n_eligible_tickers``,
  ``n_valid_pairs_by_lag`` so readers can see exactly why a null
  autocorr is null.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
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
MIN_PAIRS_PER_TICKER_LAG = 5  # below this, per-ticker corr is unreliable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", type=str, default=str(DEFAULT_DB),
                   help="Path to sim_runs.db (default: data/sim_runs.db)")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                   help="Output JSON path")
    p.add_argument(
        "--min-ticker-dates", type=int, default=30,
        help="Minimum dates a ticker must appear in a regime "
             "for its per-ticker autocorr to count toward the mean",
    )
    return p.parse_args()


def load_mu_panel(db_path: Path) -> pd.DataFrame:
    """Pull (date, ticker, mu, regime) from score_distribution.

    Fails fast with a clear message if the DB does not exist (#128
    review fix: previously raised sqlite3.OperationalError).
    """
    if not db_path.exists():
        raise SystemExit(
            f"DB not found: {db_path}\n"
            "Note: data/ is gitignored. Provide --db <path> for an "
            "external decision-trace DB, or run a sim cycle to populate "
            "the default location."
        )
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


def per_ticker_autocorr_trading_day_lag(
    series: pd.Series,
    global_trading_dates: pd.DatetimeIndex,
    lag: int,
) -> tuple[Optional[float], int]:
    """Pearson corr(μ̂_t, μ̂_{t+k}) at trading-day lag ``lag`` for one ticker.

    ``series`` is the ticker's (date, μ̂) observations within one regime
    (may be sparse).  We reindex against the *global* trading-day
    index so ``.shift(lag)`` advances by exactly ``lag`` trading days,
    not by ``lag`` later observations.

    Returns ``(corr, n_pairs)`` — ``corr`` is None when fewer than
    :data:`MIN_PAIRS_PER_TICKER_LAG` valid (t, t+k) pairs survive or
    when either side has zero variance.
    """
    s_global = series.reindex(global_trading_dates).sort_index()
    paired = pd.concat([s_global, s_global.shift(lag)], axis=1).dropna()
    n = len(paired)
    if n < MIN_PAIRS_PER_TICKER_LAG:
        return (None, n)
    a = paired.iloc[:, 0].values
    b = paired.iloc[:, 1].values
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return (None, n)
    return (float(np.corrcoef(a, b)[0, 1]), n)


def regime_mean_autocorr(
    regime_df: pd.DataFrame,
    global_trading_dates: pd.DatetimeIndex,
    lag: int,
    min_ticker_dates: int,
) -> tuple[Optional[float], int, int]:
    """Mean per-ticker autocorr at trading-day lag ``lag``.

    Returns ``(mean_corr, n_eligible_tickers, n_total_valid_pairs)``.
    A ticker is *eligible* when it has ≥ ``min_ticker_dates``
    observations in this regime AND returns a finite corr at this lag.
    """
    corrs = []
    total_pairs = 0
    for ticker, sub in regime_df.groupby("ticker"):
        if len(sub) < min_ticker_dates:
            continue
        series = sub.set_index("date")["mu"]
        c, npairs = per_ticker_autocorr_trading_day_lag(
            series, global_trading_dates, lag,
        )
        if c is not None and np.isfinite(c):
            corrs.append(c)
            total_pairs += npairs
    if not corrs:
        return (None, 0, total_pairs)
    return (float(np.mean(corrs)), len(corrs), total_pairs)


def topk_overlap_rate_trading_day_lag(
    regime_df: pd.DataFrame,
    global_trading_dates: pd.DatetimeIndex,
    K: int,
    lag: int,
) -> tuple[Optional[float], int]:
    """Fraction of top-K names that survive K-day-by-K-day re-ranking.

    Pivots on the global trading-day index so the lag is in trading
    days. Returns ``(mean_overlap, n_valid_pairs)``.
    """
    pivot = regime_df.pivot_table(
        index="date", columns="ticker", values="mu", aggfunc="mean",
    )
    pivot = pivot.reindex(global_trading_dates).sort_index()
    overlaps = []
    for i in range(len(global_trading_dates) - lag):
        t1 = global_trading_dates[i]
        t2 = global_trading_dates[i + lag]
        row1 = pivot.loc[t1].dropna()
        row2 = pivot.loc[t2].dropna()
        if len(row1) < K or len(row2) < K:
            continue
        top1 = set(row1.nlargest(K).index)
        top2 = set(row2.nlargest(K).index)
        overlaps.append(len(top1 & top2) / K)
    if not overlaps:
        return (None, 0)
    return (float(np.mean(overlaps)), len(overlaps))


def half_life(mean_autocorr: dict) -> Optional[int]:
    """Smallest trading-day lag k for which mean autocorr ≤ 0.5.

    Returns None if autocorr never crosses 0.5 in the measured range,
    or if all measured values are null.
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


def measure_regime(
    regime: str,
    regime_df: pd.DataFrame,
    global_trading_dates: pd.DatetimeIndex,
    min_ticker_dates: int,
) -> dict:
    """Build the per-regime evidence block on a global trading-day index."""
    n_rows = len(regime_df)
    n_dates = regime_df["date"].nunique()
    n_tickers = regime_df["ticker"].nunique()

    mean_autocorr: dict[str, Optional[float]] = {}
    n_eligible_by_lag: dict[str, int] = {}
    n_pairs_by_lag: dict[str, int] = {}
    for lag in LAGS:
        c, n_eligible, n_pairs = regime_mean_autocorr(
            regime_df, global_trading_dates, lag, min_ticker_dates,
        )
        mean_autocorr[str(lag)] = c
        n_eligible_by_lag[str(lag)] = n_eligible
        n_pairs_by_lag[str(lag)] = n_pairs

    topk_overlap: dict[str, dict[str, Optional[float]]] = {}
    topk_pairs_by_lag: dict[str, dict[str, int]] = {}
    for K in TOPK_VALUES:
        topk_overlap[str(K)] = {}
        topk_pairs_by_lag[str(K)] = {}
        for lag in LAGS:
            ov, np_ = topk_overlap_rate_trading_day_lag(
                regime_df, global_trading_dates, K, lag,
            )
            topk_overlap[str(K)][str(lag)] = ov
            topk_pairs_by_lag[str(K)][str(lag)] = np_

    hl = half_life(mean_autocorr)

    # n_eligible_tickers: max across lags (the ticker count that
    # surfaced at the most-permissive lag — usually L=1).
    max_eligible = max(n_eligible_by_lag.values()) if n_eligible_by_lag else 0

    # ``undersampled`` (#128 review fix) — driven by eligibility, not n_rows.
    undersampled = max_eligible < 3 or all(
        v is None for v in mean_autocorr.values()
    )

    return {
        "n_rows": int(n_rows),
        "n_dates": int(n_dates),
        "n_tickers": int(n_tickers),
        "n_eligible_tickers_max": int(max_eligible),
        "n_eligible_tickers_by_lag": n_eligible_by_lag,
        "n_valid_pairs_by_lag": n_pairs_by_lag,
        "topk_valid_pairs_by_lag": topk_pairs_by_lag,
        "undersampled": bool(undersampled),
        "mean_autocorr": mean_autocorr,
        "topk_overlap": topk_overlap,
        "half_life_days": hl,
    }


def _data_source_label(db_path: Path) -> str:
    """Best-effort relative-path label; fall back to absolute (#128 fix)."""
    try:
        return f"{db_path.relative_to(REPO)}::score_distribution"
    except ValueError:
        return f"{os.fspath(db_path)}::score_distribution"


def main() -> None:
    args = parse_args()
    db_path = Path(args.db).resolve()
    out_path = Path(args.out)

    df = load_mu_panel(db_path)
    if df.empty:
        raise SystemExit(
            f"No (date,ticker,mu,regime) rows in {db_path}::score_distribution"
        )

    # The global trading-day index — sorted unique dates across the
    # FULL panel, regardless of regime. Used for all .shift(lag) calls
    # so lag is in global trading days, not in-regime observations.
    global_trading_dates = pd.DatetimeIndex(sorted(df["date"].unique()))

    payload = {
        "as_of_date": "2026-06-03",
        "data_source": _data_source_label(db_path),
        "n_total_rows": int(len(df)),
        "n_global_trading_dates": int(len(global_trading_dates)),
        "date_range": [
            df["date"].min().date().isoformat(),
            df["date"].max().date().isoformat(),
        ],
        "n_unique_tickers": int(df["ticker"].nunique()),
        "lags_measured_trading_days": list(LAGS),
        "topk_values_measured": list(TOPK_VALUES),
        "min_ticker_dates_for_autocorr_inclusion": int(args.min_ticker_dates),
        "regime_stratification": (
            "by regime at t only. The persistence question is: "
            "given we are in regime R at t, how stable is μ̂ k trading "
            "days later regardless of the regime at t+k. A 'regime "
            "stays R from t to t+k' alternative would shrink the sample "
            "and answer a different question."
        ),
        "regimes": {},
        "interpretation_notes": [
            "Lag is in GLOBAL TRADING DAYS (corrected after #128 review). "
            "An earlier version of this script used pd.Series.shift on "
            "each ticker's sparse in-regime series, which on this DB "
            "averaged 37 trading-day gaps at nominal L=20.",
            "mean_autocorr is the mean Pearson corr(μ̂_t, μ̂_{t+k}) taken "
            "across tickers eligible at this lag. Values close to 1 mean "
            "the forecast barely changes (Level-2 MPO has little to "
            "optimize); near 0 means fast forecast decay (Level-2 has "
            "more to optimize).",
            "topk_overlap measures rank-set persistence — useful for "
            "checking whether the top-K names cycle rapidly even when "
            "marginal μ̂ values are stable.",
            "n_eligible_tickers_by_lag tells you why a null autocorr is "
            "null (zero eligible tickers ⇒ no per-ticker correlation "
            "could be computed). n_valid_pairs_by_lag is the total "
            "(t, t+k) pair count summed across eligible tickers.",
            "'undersampled' (#128 review fix) is True when fewer than "
            "3 tickers are eligible at the most-permissive lag, OR all "
            "lags yield null autocorr. n_rows alone is not the right "
            "criterion.",
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
            regime, regime_df, global_trading_dates, args.min_ticker_dates,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    log.info("Wrote evidence to %s", out_path)

    # Console summary
    print("\n=== μ̂ autocorrelation per regime (GLOBAL trading-day lags) ===")
    print(f"{'Regime':<14} {'n_rows':>7} {'n_elig':>7} " + " ".join(
        f"{'L'+str(lag):>7}" for lag in LAGS
    ) + f"  {'half_life':>10}")
    for regime, block in payload["regimes"].items():
        if block.get("skipped"):
            print(f"{regime:<14} (skipped — no rows)")
            continue
        ac = block["mean_autocorr"]
        row = f"{regime:<14} {block['n_rows']:>7} {block['n_eligible_tickers_max']:>7} "
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
