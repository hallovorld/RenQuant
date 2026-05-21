#!/usr/bin/env python
"""Evaluate a cutoff-trained PROD XGB on dates STRICTLY POST cutoff.

Inputs:
  - cutoff-trained artifact (produced by retrain_prod_truly_oos.py),
    with `_truly_oos_train_cutoff` stamped in metadata
  - canonical panel (label fwd_60d_excess)

Output:
  artifacts/prod/truly_oos_eval/eval_truly_oos.json:
    {
      "train_cutoff":  "2024-07-01",
      "eval_dates":    [...],
      "ic_per_date":   [...],
      "ic_mean":       float,
      "ic_std":        float,
      "n_pos_days":    int,
      "top10_alpha":   float,
      "long_short":    float,
      "per_regime":    {regime: {n, ic_mean, top10_alpha}},
    }

Consumed by tests/test_prod_signal_truly_oos.py — those tests will go
from SKIPPED → PASSED/FAILED after this script writes the JSON.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("eval-truly-oos")


def detect_regime(panel_dates: pd.Series) -> pd.Series:
    """Per-date regime via SPY 60d-return + 60d-vol (same as eval_prod_vs_shadow)."""
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError:
        return pd.Series(index=panel_dates.unique(), data="UNKNOWN")
    start = panel_dates.min() - pd.Timedelta(days=120)
    end   = panel_dates.max() + pd.Timedelta(days=2)
    spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] for c in spy.columns]
    if spy.empty:
        return pd.Series(index=panel_dates.unique(), data="UNKNOWN")
    s = pd.DataFrame({"close": spy["Close"]})
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s["ret_60d"] = s["close"].pct_change(60)
    s["vol_60d_ann"] = s["close"].pct_change().rolling(60).std() * np.sqrt(252)
    def lbl(r):
        if pd.isna(r["ret_60d"]) or pd.isna(r["vol_60d_ann"]): return "UNKNOWN"
        if r["ret_60d"] < -0.08: return "BEAR"
        if r["ret_60d"] > 0.10 and r["vol_60d_ann"] < 0.18: return "BULL_STRONG"
        if r["ret_60d"] > 0 and r["vol_60d_ann"] < 0.15: return "BULL_CALM"
        if r["ret_60d"] > 0: return "BULL_VOLATILE"
        return "CHOPPY"
    return s.apply(lbl, axis=1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir",
                   default="backtesting/renquant_104/artifacts/prod/truly_oos_eval")
    p.add_argument("--panel", default="data/alpha158_291_fundamental_dataset.parquet")
    p.add_argument("--label", default="fwd_60d_excess")
    p.add_argument("--out",
                   default="backtesting/renquant_104/artifacts/prod/truly_oos_eval/eval_truly_oos.json")
    args = p.parse_args()

    art_dir = REPO / args.artifact_dir
    art_path = art_dir / "panel-ltr.json"
    if not art_path.exists():
        log.error("Artifact not found: %s", art_path)
        return 2

    art = json.loads(art_path.read_text())
    cutoff_str = art.get("_truly_oos_train_cutoff") or art.get("training_notes", "")
    # Robust extract: any YYYY-MM-DD string under train_cutoff
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})", cutoff_str)
    if not m:
        log.error("Could not extract train_cutoff from artifact")
        return 2
    cutoff = pd.Timestamp(m.group(1))
    log.info("Artifact train_cutoff = %s", cutoff.date())

    # Load panel
    log.info("Loading panel: %s", args.panel)
    panel = pd.read_parquet(REPO / args.panel)
    panel["date"] = pd.to_datetime(panel["date"])
    feat_cols = art["feature_cols"]
    feat_means = np.asarray(art["feature_means"], dtype=np.float64)
    feat_stds  = np.asarray(art["feature_stds"], dtype=np.float64)

    # Eval window: dates STRICTLY > cutoff with valid label (panel_max - 60d)
    last_label_date = panel.dropna(subset=[args.label])["date"].max()
    eval_panel = panel[(panel["date"] > cutoff) & (panel["date"] <= last_label_date)]
    eval_panel = eval_panel.dropna(subset=[args.label])
    log.info("Eval window: %s → %s (n_dates=%d, n_rows=%d)",
             eval_panel["date"].min().date(), eval_panel["date"].max().date(),
             eval_panel["date"].nunique(), len(eval_panel))

    # Load XGB booster
    import xgboost as xgb  # noqa: PLC0415
    booster = xgb.Booster()
    booster.load_model(bytearray(art["booster_raw_json"].encode("utf-8")))

    # Score whole eval frame
    log.info("Scoring %d rows ...", len(eval_panel))
    X = eval_panel[feat_cols].fillna(0.0).values.astype(np.float64)
    safe = np.where(feat_stds > 0, feat_stds, 1.0)
    Xn = ((X - feat_means) / safe).clip(-5, 5)
    d = xgb.DMatrix(Xn, feature_names=feat_cols)
    eval_panel = eval_panel.copy()
    eval_panel["score"] = booster.predict(d)

    # Regime per date
    log.info("Detecting regimes ...")
    regimes = detect_regime(eval_panel["date"])

    # Per-date IC + top10 + bot10
    from scipy.stats import spearmanr  # noqa: PLC0415
    rows = []
    for d, grp in eval_panel.groupby("date"):
        if len(grp) < 5:
            continue
        ic, _ = spearmanr(grp["score"], grp[args.label])
        if not np.isfinite(ic):
            continue
        top10 = grp.nlargest(10, "score")[args.label].mean()
        bot10 = grp.nsmallest(10, "score")[args.label].mean()
        u_mean = grp[args.label].mean()
        rows.append({
            "date": str(d.date()),
            "n": len(grp),
            "ic": float(ic),
            "top10_alpha": float(top10 - u_mean),
            "bot10_alpha": float(bot10 - u_mean),
            "long_short": float(top10 - bot10),
            "regime": str(regimes.get(d, "UNKNOWN")),
        })

    if not rows:
        log.error("No dates with valid scoring")
        return 2

    df = pd.DataFrame(rows)
    log.info("=" * 70)
    log.info("TRULY OOS RESULTS — train cutoff %s", cutoff.date())
    log.info("=" * 70)
    log.info("Dates evaluated: %d (%s → %s)",
             len(df), df["date"].min(), df["date"].max())
    log.info("Mean IC: %+.4f  (median=%+.4f, std=%.4f)",
             df["ic"].mean(), df["ic"].median(), df["ic"].std())
    log.info("Pos-IC days: %d/%d (%.0f%%)",
             int((df["ic"] > 0).sum()), len(df), 100*(df["ic"] > 0).mean())
    log.info("Top-10 alpha mean: %+.4f", df["top10_alpha"].mean())
    log.info("Bot-10 alpha mean: %+.4f", df["bot10_alpha"].mean())
    log.info("Long-short mean: %+.4f", df["long_short"].mean())
    log.info("")
    log.info("Per-regime IC + top10_alpha:")
    per_regime = {}
    for r, g in df.groupby("regime"):
        per_regime[r] = {
            "n": int(len(g)),
            "ic_mean": float(g["ic"].mean()),
            "top10_alpha": float(g["top10_alpha"].mean()),
            "long_short": float(g["long_short"].mean()),
        }
        log.info("  %-14s n=%4d  ic=%+.4f  top10_α=%+.4f  long-short=%+.4f",
                 r, len(g), g["ic"].mean(),
                 g["top10_alpha"].mean(), g["long_short"].mean())

    out = {
        "train_cutoff":   cutoff.date().isoformat(),
        "eval_dates":     df["date"].tolist(),
        "ic_per_date":    df["ic"].tolist(),
        "ic_mean":        float(df["ic"].mean()),
        "ic_std":         float(df["ic"].std()),
        "ic_median":      float(df["ic"].median()),
        "n_pos_days":     int((df["ic"] > 0).sum()),
        "top10_alpha":    float(df["top10_alpha"].mean()),
        "bot10_alpha":    float(df["bot10_alpha"].mean()),
        "long_short":     float(df["long_short"].mean()),
        "per_regime":     per_regime,
        "produced_at":    datetime.now(timezone.utc).isoformat(),
    }
    # Write to test-expected path
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote → %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
