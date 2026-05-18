#!/usr/bin/env python3
"""IC eval for news sentiment features (Roadmap C5 step 3).

Joins per-ticker-per-date sentiment features (from
``score_news_finbert.py``) to the alpha158_fund panel and computes
cross-sectional IC against fwd_5d / fwd_20d / fwd_60d excess returns.

Sanity battery (CLAUDE.md §5.2):
  1. Walk-forward IC vs Linear/Ridge baseline (currently
     just the raw feature IC; baseline comparison is N/A for
     a single-feature drop-in).
  2. Time-shift placebo: shift sentiment timestamps by ±30 days
     (article t → label t-30) — IC must collapse to ~0.
  3. Shuffled-label placebo: within each date, randomly permute
     sentiment across tickers — IC must be ~0.
  4. A/A test: split news in half by date parity, compute IC on
     each half — verify magnitude consistency.

A feature passes if:
  • raw |IC| > 2× max(placebo |IC|), AND
  • Walk-forward sign consistency ≥ 4/5 cuts, AND
  • magnitude is reasonable (|IC| ∈ [0.005, 0.05])

Writes ``artifacts/ic_eval_news_sentiment.json`` with full battery
results, NOT a "promote" verdict (operator review per CLAUDE.md
§5.13.4a Tier 3 gate).

Reference: Tetlock 2007 *JF*; Garcia 2013 *JF*; Ke-Kelly-Xiu 2019.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PANEL_PATH = REPO / "data" / "alpha158_291_fundamental_dataset.parquet"
SENT_DIR = REPO / "data" / "news_sentiment_alpaca"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ic_eval_news_sentiment")


def _load_panel(panel_path: Path) -> pd.DataFrame:
    """Read only the columns we need (cheap)."""
    cols = ["ticker", "date",
            "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"]
    df = pd.read_parquet(panel_path, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_sentiment(sent_dir: Path) -> pd.DataFrame:
    parts = []
    for f in sorted(sent_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def _xs_ic(merged: pd.DataFrame, feat: str, label: str) -> tuple[float, int]:
    """Cross-sectional IC averaged across dates (FF mean).

    Returns (mean_ic, n_valid_dates).
    """
    ics = []
    for d, g in merged.groupby("date"):
        if len(g) < 5:
            continue  # too few cross-sectional obs
        x = g[feat].values
        y = g[label].values
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            continue
        # Spearman rank corr is more robust than Pearson here
        from scipy.stats import spearmanr
        r, _ = spearmanr(x, y)
        if not np.isnan(r):
            ics.append(r)
    if not ics:
        return float("nan"), 0
    return float(np.mean(ics)), len(ics)


def _walk_forward_ic(merged: pd.DataFrame, feat: str, label: str,
                     n_cuts: int = 5) -> list[float]:
    """Split into n_cuts contiguous time slices; compute IC on each."""
    dates = sorted(merged["date"].unique())
    if len(dates) < n_cuts * 5:
        return []
    cut_size = len(dates) // n_cuts
    ics = []
    for i in range(n_cuts):
        slice_dates = dates[i * cut_size:(i + 1) * cut_size]
        sub = merged[merged["date"].isin(slice_dates)]
        ic, _ = _xs_ic(sub, feat, label)
        ics.append(ic)
    return ics


def _time_shift_placebo(panel: pd.DataFrame, sent: pd.DataFrame,
                        feat: str, label: str, shift_days: int = 30) -> float:
    """Shift sentiment dates BACKWARD by shift_days, then re-join + IC.

    Backward-shifted sentiment is "future" relative to the label →
    IC should collapse to ~0 if the original IC was due to causation.
    """
    s = sent.copy()
    s["date"] = s["date"] - pd.Timedelta(days=shift_days)
    merged = panel.merge(s[["ticker", "date", feat]],
                          on=["ticker", "date"], how="inner")
    ic, _ = _xs_ic(merged, feat, label)
    return ic


def _shuffle_label_placebo(merged: pd.DataFrame, feat: str, label: str,
                            seed: int = 42, n_runs: int = 5) -> list[float]:
    """Within each date, shuffle the feature column across tickers.

    Repeats n_runs times with different seeds → spread of placebo IC.
    True signal IC should be >> max placebo IC.
    """
    ics = []
    rng = np.random.RandomState(seed)
    for run in range(n_runs):
        m = merged.copy()
        m["_shuffled"] = m.groupby("date")[feat].transform(
            lambda x: rng.permutation(x.values))
        ic, _ = _xs_ic(m, "_shuffled", label)
        ics.append(ic)
    return ics


def _aa_split(merged: pd.DataFrame, feat: str, label: str) -> tuple[float, float]:
    """Split dates by parity → IC on each half should agree in sign + magnitude."""
    dates = sorted(merged["date"].unique())
    even = [d for i, d in enumerate(dates) if i % 2 == 0]
    odd  = [d for i, d in enumerate(dates) if i % 2 == 1]
    ic_e, _ = _xs_ic(merged[merged["date"].isin(even)], feat, label)
    ic_o, _ = _xs_ic(merged[merged["date"].isin(odd)], feat, label)
    return ic_e, ic_o


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", default=str(PANEL_PATH),
                   help="alpha158 panel parquet")
    p.add_argument("--sent-dir", default=str(SENT_DIR),
                   help="news_sentiment_alpaca/")
    p.add_argument("--features", nargs="*",
                   default=["mean_sentiment", "sentiment_dispersion",
                            "n_articles", "sentiment_pos_share",
                            "sentiment_neg_share"])
    p.add_argument("--labels", nargs="*",
                   default=["fwd_5d_excess", "fwd_20d_excess",
                            "fwd_60d_excess"])
    p.add_argument("--out", default="artifacts/ic_eval_news_sentiment.json")
    args = p.parse_args()

    log.info("loading panel %s ...", args.panel)
    panel = _load_panel(Path(args.panel))
    log.info("  panel: %s rows %s tickers [%s, %s]",
             f"{len(panel):,}", panel["ticker"].nunique(),
             panel["date"].min().date(), panel["date"].max().date())

    log.info("loading sentiment %s/ ...", args.sent_dir)
    sent = _load_sentiment(Path(args.sent_dir))
    log.info("  sentiment: %s rows %s tickers [%s, %s]",
             f"{len(sent):,}", sent["symbol"].nunique(),
             sent["date"].min().date(), sent["date"].max().date())
    # rename for join
    sent = sent.rename(columns={"symbol": "ticker"})

    merged = panel.merge(sent, on=["ticker", "date"], how="inner")
    log.info("  merged: %s rows  feature-coverage=%.1f%% of panel",
             f"{len(merged):,}", 100 * len(merged) / len(panel))
    if merged.empty:
        log.error("no overlap between panel + sentiment dates — bail")
        return 1

    results: dict = {
        "panel": args.panel,
        "sent_dir": args.sent_dir,
        "merged_rows": int(len(merged)),
        "panel_date_max": str(panel["date"].max().date()),
        "sent_date_max": str(sent["date"].max().date()),
        "features": {},
    }

    for feat in args.features:
        if feat not in merged.columns:
            log.warning("  feature %s missing in sentiment data; skip", feat)
            continue
        feat_block: dict = {}
        for lab in args.labels:
            if lab not in merged.columns:
                continue
            raw_ic, n_dates = _xs_ic(merged, feat, lab)
            wf = _walk_forward_ic(merged, feat, lab, n_cuts=5)
            ts_30 = _time_shift_placebo(panel, sent, feat, lab, shift_days=30)
            ts_neg30 = _time_shift_placebo(panel, sent, feat, lab, shift_days=-30)
            shuf = _shuffle_label_placebo(merged, feat, lab, n_runs=5)
            aa_e, aa_o = _aa_split(merged, feat, lab)
            sign_consistent = sum(1 for x in wf if x is not None and not np.isnan(x)
                                  and np.sign(x) == np.sign(raw_ic))
            feat_block[lab] = {
                "raw_ic": raw_ic,
                "n_dates": n_dates,
                "walk_forward_ic": [None if np.isnan(x) else x for x in wf],
                "wf_sign_consistent": sign_consistent,
                "placebo_time_shift_+30d": ts_30,
                "placebo_time_shift_-30d": ts_neg30,
                "placebo_shuffle_max_abs": float(np.nanmax(np.abs(shuf)) if shuf else float("nan")),
                "placebo_shuffle_mean": float(np.nanmean(shuf) if shuf else float("nan")),
                "aa_split_even": aa_e,
                "aa_split_odd": aa_o,
            }
            log.info("  %-22s × %-18s  raw_ic=%+.4f  n=%4d  "
                     "wf=[%s]  shuf_max=%+.4f",
                     feat, lab, raw_ic, n_dates,
                     " ".join(f"{x:+.3f}" if not np.isnan(x) else "  nan"
                              for x in wf),
                     float(np.nanmax(np.abs(shuf)) if shuf else float("nan")))
        results["features"][feat] = feat_block

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2, default=str))
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
