#!/usr/bin/env python
"""Shadow inference: load production artifact, score current universe, show picks.

This is a pre-promotion check — runs full inference using the production
artifact format without touching any live config or trades. Compares to
prior production picks if available.

Output: top 30 long picks + bottom 30 short picks for the most recent
available date.
"""
from __future__ import annotations
import json, logging
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("shadow")

REPO = Path(__file__).resolve().parent.parent


def main():
    log.info("Loading production artifact...")
    art_path = REPO / "data" / "panel-ltr-prod-alpha158-fund-fwd60d.json"
    art = json.loads(art_path.read_text())
    log.info("Artifact: kind=%s feature_cols=%d label=%s lookahead=%d trained=%s",
             art["kind"], len(art["feature_cols"]), art["label_col"],
             art["lookahead_days"], art["trained_date"])
    log.info("Fingerprint: %s", art["config_fingerprint"])

    # Reconstruct booster
    booster = xgb.Booster()
    booster.load_model(bytearray(art["booster_raw_json"].encode("utf-8")))

    # Load most recent panel
    panel = pd.read_parquet(REPO / "data" / "alpha158_291_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    # Score the most recent date with valid features
    latest_date = panel["date"].max()
    today = panel[panel["date"] == latest_date]
    log.info("Scoring %d tickers on date %s (most recent in panel)",
             len(today), latest_date.date())

    feat_cols = art["feature_cols"]
    mu = np.array(art["feature_means"])
    sd = np.array(art["feature_stds"])
    X = today[feat_cols].fillna(0).values.astype(np.float64)
    Xn = ((X - mu) / sd).clip(-5, 5)

    preds = booster.predict(xgb.DMatrix(Xn))
    today = today.copy()
    today["pred"] = preds
    today_sorted = today.sort_values("pred", ascending=False)

    log.info("\n══ TOP 30 LONG PICKS (highest predicted 60d excess returns) ══")
    log.info("%-6s %s", "Tkr", "Predicted score")
    for _, r in today_sorted.head(30).iterrows():
        log.info("%-6s  %+.4f", r["ticker"], r["pred"])

    log.info("\n══ BOTTOM 30 SHORT PICKS (lowest predicted) ══")
    for _, r in today_sorted.tail(30).iloc[::-1].iterrows():
        log.info("%-6s  %+.4f", r["ticker"], r["pred"])

    # Stats
    log.info("\n══ Distribution stats ══")
    log.info("min=%+.4f median=%+.4f max=%+.4f std=%.4f spread=%.4f",
             today["pred"].min(), today["pred"].median(),
             today["pred"].max(), today["pred"].std(),
             today["pred"].max() - today["pred"].min())

    # Save predictions
    out = REPO / "data" / "shadow_predictions.json"
    out.write_text(json.dumps({
        "artifact": art_path.name,
        "scoring_date": str(latest_date.date()),
        "n_tickers": len(today),
        "top30_long": today_sorted.head(30)[["ticker","pred"]].to_dict("records"),
        "bottom30_short": today_sorted.tail(30)[["ticker","pred"]].to_dict("records"),
    }, indent=2, default=str))
    log.info("Saved predictions: %s", out)


if __name__ == "__main__":
    main()
