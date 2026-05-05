#!/usr/bin/env python
"""Phases 4+5 of Transformer data prep: build the training dataset.

Per-bar features = raw OHLCV (open, high, low, close, volume) — let
the Transformer learn its own representation rather than reusing XGB's
21 hand-engineered features (per user direction 2026-05-05). Industry
references for raw-OHLCV Transformer training: Patch-TST (Nie et al.
2023 ICLR), Informer (Zhou et al. 2021 AAAI).

For each (ticker, date) pair in the clean Tier-A∪B universe, we
construct:

  X[t] = sequence of last `seq_len` bars  (default 60 trading days)
         with 5 channels per bar (O, H, L, C, V)
  y[t] = (fwd_5d_excess, fwd_20d_excess, fwd_60d_excess)

Normalization (per CLAUDE.md §1c — single Task per concern):

  1. Per-ticker rolling z-score: each channel's value at time t is
     normalized by mean and std over the trailing 252-day window
     for THAT ticker. Preserves cross-sectional ranking but kills
     baseline drift across multi-year regimes.
  2. Cross-sectional z-score per date: AFTER per-ticker scaling, also
     z-score across tickers per date so the cross-sectional signal
     dominates.
  3. Clip to ±5σ — kills outlier bars (real or data-bug) that would
     wreck attention.

Walk-forward split (Phase 5):

  Train: 2014-01-01 → 2022-12-31  (8 years)
  Val:   2023-01-01 → 2023-12-31  (1 year)
  Test:  2024-01-01 → end_of_cache (~2 years, includes recent OOS)

  60-day embargo between train/val and val/test boundaries — prevents
  label leakage from labels at the edge.

Output: data/transformer_dataset.parquet — long format, one row per
(ticker, date, feature_index) so it's a flat schema. Training loop
will reshape into [batch, seq_len, channels] tensors at load time.

Usage:
    python scripts/transformer_dataset_builder.py
    python scripts/transformer_dataset_builder.py --seq-len 30 --train-end 2022-06-30
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("transformer-dataset")

CHANNELS = ["open", "high", "low", "close", "volume"]


def _per_ticker_zscore(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Rolling-window z-score of each channel, per ticker.

    Trailing-only mean/std (no future leak). First `window` rows of
    each ticker have insufficient history → NaN, will be filtered.
    """
    out = df.copy()
    for c in CHANNELS:
        if c not in df.columns:
            continue
        roll_mean = df[c].rolling(window, min_periods=window // 2).mean()
        roll_std  = df[c].rolling(window, min_periods=window // 2).std()
        out[c] = (df[c] - roll_mean) / roll_std.replace(0, np.nan)
    return out


def _cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per date, across tickers, per channel.
    Applied AFTER per-ticker normalization so the cross-sectional
    signal dominates.
    """
    out = panel.copy()
    for c in CHANNELS:
        if c not in out.columns:
            continue
        date_mean = out.groupby("date")[c].transform("mean")
        date_std  = out.groupby("date")[c].transform("std")
        out[c] = (out[c] - date_mean) / date_std.replace(0, np.nan)
    return out


def _clip_outliers(df: pd.DataFrame, sigma: float = 5.0) -> pd.DataFrame:
    """Clip each channel to ±sigma. Already z-scored so sigma=5 ≈ 5σ."""
    out = df.copy()
    for c in CHANNELS:
        if c not in df.columns:
            continue
        out[c] = out[c].clip(lower=-sigma, upper=sigma)
    return out


def _walk_forward_split(panel: pd.DataFrame,
                         train_end: str = "2022-12-31",
                         val_end:   str = "2023-12-31",
                         embargo_days: int = 60) -> pd.DataFrame:
    """Tag each row with split_label ∈ {train, embargo, val, embargo, test}.
    The embargo bands prevent label leakage from horizon=60d forward returns.
    """
    panel = panel.copy()
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts   = pd.Timestamp(val_end)
    train_emb_start = train_end_ts - pd.Timedelta(days=embargo_days)
    val_emb_start   = val_end_ts   - pd.Timedelta(days=embargo_days)

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
    return panel


def build_dataset(ohlcv_dir: Path, universe: list[str], labels_path: Path,
                   seq_len: int, normalize_window: int,
                   train_end: str, val_end: str, embargo_days: int) -> pd.DataFrame:
    """Compose all phases: load OHLCV → normalize → join labels → split."""

    # 1. Load + normalize OHLCV per ticker
    log.info("Phase 4a: Loading + per-ticker rolling z-score (window=%d)…",
              normalize_window)
    rows: list[pd.DataFrame] = []
    for i, t in enumerate(universe):
        if i % 50 == 0 and i > 0:
            log.info("  ... %d/%d normalized", i, len(universe))
        p = ohlcv_dir / t / "1d.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=CHANNELS)
        except Exception as exc:
            log.warning("  %s: read failed — %s", t, exc)
            continue
        if df.empty:
            continue
        df = _per_ticker_zscore(df, window=normalize_window)
        df = df.dropna(subset=CHANNELS)
        if df.empty:
            continue
        df = df.reset_index().rename(columns={"index": "date"})
        df.insert(0, "ticker", t)
        rows.append(df)

    if not rows:
        log.error("No tickers with data after per-ticker z-score")
        sys.exit(1)
    panel = pd.concat(rows, ignore_index=True)
    log.info("After per-ticker zscore: %d rows", len(panel))

    # 2. Cross-sectional z-score per date
    log.info("Phase 4b: Cross-sectional z-score per date…")
    panel = _cross_sectional_zscore(panel)
    panel = panel.dropna(subset=CHANNELS)

    # 3. Clip outliers
    log.info("Phase 4c: Clip outliers to ±5σ…")
    panel = _clip_outliers(panel, sigma=5.0)

    # 4. Join labels
    log.info("Phase 5: Join multi-horizon labels…")
    labels = pd.read_parquet(labels_path)
    labels["date"] = pd.to_datetime(labels["date"])
    panel = panel.merge(labels, on=["ticker", "date"], how="inner")
    log.info("  After label join: %d rows", len(panel))

    # 5. Walk-forward split
    log.info("Phase 5: Walk-forward split (train_end=%s, val_end=%s, embargo=%dd)",
              train_end, val_end, embargo_days)
    panel = _walk_forward_split(panel, train_end, val_end, embargo_days)

    # NB: We are NOT yet building the [seq_len × channels] sequences. That's
    # a runtime concern at training-time (cheaper to pivot at load time
    # than to materialize all sequences here). The dataset stores per-bar
    # features + labels + split_label; the training DataLoader assembles
    # rolling windows per (ticker, date) when needed.

    return panel


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
                    default=str(REPO_ROOT / "data" / "transformer_dataset.parquet"))
    p.add_argument("--seq-len",          type=int,   default=60,
                    help="Length of historical sequence per training sample")
    p.add_argument("--normalize-window", type=int,   default=252,
                    help="Rolling window for per-ticker z-score")
    p.add_argument("--train-end",        type=str,   default="2022-12-31")
    p.add_argument("--val-end",          type=str,   default="2023-12-31")
    p.add_argument("--embargo-days",     type=int,   default=60,
                    help="Embargo between train/val/test (≥ max horizon)")
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
    log.info("Building dataset for %d tickers", len(universe))

    panel = build_dataset(
        ohlcv_dir=Path(args.ohlcv_dir),
        universe=universe,
        labels_path=Path(args.labels),
        seq_len=args.seq_len,
        normalize_window=args.normalize_window,
        train_end=args.train_end,
        val_end=args.val_end,
        embargo_days=args.embargo_days,
    )

    # Summary
    splits = panel["split_label"].value_counts()
    log.info("══ Dataset summary ══")
    for k, v in splits.items():
        log.info("  %-22s %d rows", k, v)
    log.info("Total: %d rows", len(panel))
    log.info("Date range: %s → %s", panel["date"].min(), panel["date"].max())
    log.info("Tickers: %d", panel["ticker"].nunique())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    log.info("══ dataset written %s ══", out_path)


if __name__ == "__main__":
    main()
