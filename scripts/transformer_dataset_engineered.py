#!/usr/bin/env python
"""Phase 6 of Transformer data prep: engineered-feature dataset.

Parallel to scripts/transformer_dataset_builder.py (raw OHLCV) — this
builds the feature path B for A/B comparison: 11 technical indicators
derived from raw OHLCV via kernel.indicators.compute_all (rsi, adx,
cci, bbp, williams_r, macd_hist, obv_slope, trend, trend_long,
rel_mom_20d, rel_mom_60d).

Note: production XGB uses 21 features which include cross-sectional
panel z-scores (size_z, mom_12_1_z, beta_60d_z, ...) that require
running the full panel pipeline (~12 min × 292 tickers, plus
fundamentals + sector momentum + macro). For first iteration of the
Transformer A/B, technical-indicator subset is the principled
comparison: same raw input (OHLCV), different transformations applied
before the model sees them.

Subsequent iterations can add the cross-sectional z-features as a
"path C" if technical-engineered shows positive lift over raw.

Usage:
    python scripts/transformer_dataset_engineered.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("transformer-engineered")

# 11 technical indicators directly computable from OHLCV — subset of
# production XGB's 21 features (the half that doesn't need panel-wide
# Z-scoring or sector classification).
TECH_FEATURES = [
    "rsi", "adx", "cci", "bbp", "williams_r",
    "macd_hist", "obv_slope",
    "trend", "trend_long",
    "rel_mom_20d", "rel_mom_60d",
]


def _build_features_for_ticker(t: str, ohlcv_dir: Path,
                                 spy_df: pd.DataFrame, spec: dict) -> pd.DataFrame | None:
    """Mirror inference path: compute_all(ticker_OHLCV) - SPY-relative."""
    from kernel.indicators import build_feature_frame  # noqa: PLC0415
    p = ohlcv_dir / t / "1d.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception as exc:
        log.warning("  %s: read failed — %s", t, exc)
        return None
    if df.empty:
        return None
    feats = build_feature_frame(df, spy_df, spec, 20)
    if feats is None or feats.empty:
        return None
    # Keep only the 11 technical features we want; drop rest (some
    # build_feature_frame outputs include scalars / unused cols)
    keep = [c for c in TECH_FEATURES if c in feats.columns]
    if not keep:
        return None
    return feats[keep]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory",
                    default=str(REPO_ROOT / "data" / "transformer_universe_inventory.json"))
    p.add_argument("--integrity-report",
                    default=str(REPO_ROOT / "data" / "transformer_data_integrity_report.json"))
    p.add_argument("--labels",
                    default=str(REPO_ROOT / "data" / "transformer_panel_labels.parquet"))
    p.add_argument("--ohlcv-dir",
                    default=str(REPO_ROOT / "data" / "ohlcv"))
    p.add_argument("--output",
                    default=str(REPO_ROOT / "data" / "transformer_dataset_engineered.parquet"))
    p.add_argument("--normalize-window", type=int, default=252)
    p.add_argument("--train-end",       type=str, default="2022-12-31")
    p.add_argument("--val-end",         type=str, default="2023-12-31")
    p.add_argument("--embargo-days",    type=int, default=60)
    args = p.parse_args()

    inv   = json.loads(Path(args.inventory).read_text())
    integ = json.loads(Path(args.integrity_report).read_text())
    universe = set(inv["tier_A_tickers"]) | set(inv["tier_B_tickers"])
    failed = set()
    for tier in ("A", "B"):
        for r in integ["per_ticker"][tier]:
            if not r["ok"]:
                failed.add(r["ticker"])
    universe = sorted(universe - failed)
    log.info("Building engineered dataset for %d tickers", len(universe))

    # Load SPY for SPY-relative feature derivation
    spy_path = Path(args.ohlcv_dir) / "SPY" / "1d.parquet"
    spy_df = pd.read_parquet(spy_path)

    # Read indicator spec from production strategy config
    cfg = json.loads((REPO_ROOT / "backtesting" / "renquant_104"
                      / "strategy_config.json").read_text())
    indicator_spec = cfg.get("indicator_spec", {})
    if not indicator_spec:
        log.error("No indicator_spec in strategy_config.json")
        sys.exit(1)
    log.info("Using production indicator_spec (%d entries)", len(indicator_spec))

    log.info("Phase 6a: Computing 11 technical indicators per ticker…")
    rows: list[pd.DataFrame] = []
    for i, t in enumerate(universe):
        if i % 50 == 0 and i > 0:
            log.info("  ... %d/%d computed", i, len(universe))
        feats = _build_features_for_ticker(t, Path(args.ohlcv_dir), spy_df, indicator_spec)
        if feats is None:
            continue
        feats = feats.reset_index().rename(columns={"index": "date"})
        feats.insert(0, "ticker", t)
        rows.append(feats)
    if not rows:
        log.error("No tickers produced engineered features")
        sys.exit(1)
    panel = pd.concat(rows, ignore_index=True)
    log.info("After feature compute: %d rows × %d cols", len(panel), len(panel.columns))

    # Normalize each feature with rolling per-ticker z-score (same as Path A's OHLCV)
    log.info("Phase 6b: Per-ticker rolling z-score (window=%d)…", args.normalize_window)
    feat_cols = [c for c in panel.columns if c in TECH_FEATURES]
    for c in feat_cols:
        panel[c] = panel.groupby("ticker")[c].transform(
            lambda s: (s - s.rolling(args.normalize_window,
                                       min_periods=args.normalize_window // 2).mean())
                       / s.rolling(args.normalize_window,
                                     min_periods=args.normalize_window // 2).std().replace(0, np.nan)
        )
    panel = panel.dropna(subset=feat_cols)
    log.info("After per-ticker zscore: %d rows", len(panel))

    # Cross-sectional z-score
    log.info("Phase 6c: Cross-sectional z-score per date…")
    for c in feat_cols:
        date_mean = panel.groupby("date")[c].transform("mean")
        date_std  = panel.groupby("date")[c].transform("std")
        panel[c] = (panel[c] - date_mean) / date_std.replace(0, np.nan)
    panel = panel.dropna(subset=feat_cols)

    # Clip ±5σ
    log.info("Phase 6d: Clip ±5σ…")
    for c in feat_cols:
        panel[c] = panel[c].clip(-5.0, 5.0)

    # Join labels
    log.info("Phase 6e: Join multi-horizon labels…")
    labels = pd.read_parquet(args.labels)
    labels["date"] = pd.to_datetime(labels["date"])
    panel = panel.merge(labels, on=["ticker", "date"], how="inner")
    log.info("After label join: %d rows", len(panel))

    # Walk-forward split (same as Path A)
    log.info("Phase 6f: Walk-forward split…")
    train_end_ts    = pd.Timestamp(args.train_end)
    val_end_ts      = pd.Timestamp(args.val_end)
    train_emb_start = train_end_ts - pd.Timedelta(days=args.embargo_days)
    val_emb_start   = val_end_ts   - pd.Timedelta(days=args.embargo_days)
    panel["split_label"] = "test"
    panel.loc[panel["date"] <= train_emb_start, "split_label"] = "train"
    panel.loc[
        (panel["date"] > train_emb_start) & (panel["date"] <= train_end_ts),
        "split_label",
    ] = "embargo_train_val"
    panel.loc[
        (panel["date"] > train_end_ts) & (panel["date"] <= val_emb_start),
        "split_label",
    ] = "val"
    panel.loc[
        (panel["date"] > val_emb_start) & (panel["date"] <= val_end_ts),
        "split_label",
    ] = "embargo_val_test"

    splits = panel["split_label"].value_counts()
    log.info("══ Engineered dataset summary ══")
    for k, v in splits.items():
        log.info("  %-22s %d rows", k, v)
    log.info("Total: %d rows  Tickers: %d  Features: %d",
              len(panel), panel["ticker"].nunique(), len(feat_cols))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    log.info("══ engineered dataset written %s ══", out_path)


if __name__ == "__main__":
    main()
