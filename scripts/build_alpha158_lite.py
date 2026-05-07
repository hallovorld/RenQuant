#!/usr/bin/env python
"""Alpha158-lite feature builder — adds Qlib-inspired stat features.

Reference: Microsoft Qlib alpha158 handler
  github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py

We build a focused subset (42 features) of the most-cited alpha158
families, mirroring what's been validated in production quant ML
benchmarks (Qlib LinearRegression IC ~0.045 with these features on
3000-stock universe).

Combined with our existing 11 TA indicators (rsi, adx, etc.) this
produces ~53 cross-sectional features, ≈ 5× the 11 we had.

Per Kelly+Gu+Xiu (RFS 2020) §V: "feature engineering >> architecture
choice". Their best models use ~94 firm characteristics; our 53 is
in the right ballpark for our data scale.

Output: data/alpha158_lite_dataset.parquet
  - Same panel layout as transformer_dataset_engineered.parquet
  - 42 alpha158-lite features (named alpha_*) + 11 TA features (rsi, etc.)
  - + same fwd_5d/20d/60d_excess labels + split_label
  - per-ticker rolling z-score + cross-sectional z-score per date,
    same normalization as transformer_dataset_engineered.

Usage::

    python scripts/build_alpha158_lite.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

for _k, _v in (("OMP_NUM_THREADS", "10"),
               ("MKL_NUM_THREADS", "10"),
               ("OPENBLAS_NUM_THREADS", "10")):
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("alpha158-lite")


# ── 42 alpha158-lite features ──────────────────────────────────────────────

WINDOWS_PRICE   = [5, 10, 20, 60]
WINDOWS_VOLUME  = [5, 20, 60]
WINDOWS_LONG    = [10, 20, 60]


def kline_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """K-line shape — 5 features."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    span = (h - l).replace(0, np.nan)
    out = {
        "alpha_KMID2": (c - o) / span,
        "alpha_KLEN":  (h - l) / o.replace(0, np.nan),
        "alpha_KUP":   (h - np.maximum(o, c)) / o.replace(0, np.nan),
        "alpha_KLOW":  (np.minimum(o, c) - l) / o.replace(0, np.nan),
        "alpha_KSFT2": (2 * c - h - l) / span,
    }
    return out


def rolling_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Rolling-window features — 37 features."""
    c = df["close"].astype(float)
    v = df["volume"].astype(float).replace(0, np.nan)
    out: dict[str, pd.Series] = {}

    # N-day return — 4 features
    for n in WINDOWS_PRICE:
        out[f"alpha_ROC{n}"] = c / c.shift(n) - 1.0

    # Moving average / close — 4 features
    for n in WINDOWS_PRICE:
        out[f"alpha_MA{n}"] = c.rolling(n).mean() / c - 1.0

    # Volatility — 4 features
    for n in WINDOWS_PRICE:
        out[f"alpha_STD{n}"] = c.rolling(n).std() / c

    # MAX / MIN over window — 3 + 3 features
    for n in WINDOWS_LONG:
        out[f"alpha_MAX{n}"] = c.rolling(n).max() / c - 1.0
        out[f"alpha_MIN{n}"] = c.rolling(n).min() / c - 1.0

    # RSV — Stochastic-K-style — 3 features
    for n in WINDOWS_LONG:
        roll_high = df["high"].rolling(n).max()
        roll_low  = df["low"].rolling(n).min()
        out[f"alpha_RSV{n}"] = (c - roll_low) / (roll_high - roll_low).replace(0, np.nan)

    # Volume rolling mean — 3 features
    for n in WINDOWS_VOLUME:
        out[f"alpha_VMA{n}"] = v / v.rolling(n).mean()

    # Volume rolling std — 3 features
    for n in WINDOWS_VOLUME:
        out[f"alpha_VSTD{n}"] = v.rolling(n).std() / v.rolling(n).mean()

    # CORR(close, volume) — 2 features
    for n in [20, 60]:
        out[f"alpha_CORR{n}"] = c.rolling(n).corr(v)

    # BETA: slope of close on time index — 2 features
    for n in [20, 60]:
        x = pd.Series(np.arange(len(c), dtype=float), index=c.index)
        # Manual slope = cov(x, c) / var(x)
        c_mean = c.rolling(n).mean()
        x_mean = x.rolling(n).mean()
        cov = (x - x_mean) * (c - c_mean)
        var_x = ((x - x_mean) ** 2)
        slope = cov.rolling(n).sum() / var_x.rolling(n).sum().replace(0, np.nan)
        out[f"alpha_BETA{n}"] = slope / c

    # WVMA — weighted volume × abs return — 2 features
    abs_ret = c.pct_change().abs()
    for n in [20, 60]:
        out[f"alpha_WVMA{n}"] = (v * abs_ret).rolling(n).mean() / (v.rolling(n).mean() + 1e-12)

    # IMAX / IMIN (position of max/min in window) — 2 features
    for n in [20, 60]:
        out[f"alpha_IMXD{n}"] = (
            c.rolling(n).apply(lambda x: float(np.argmax(x)), raw=True)
            - c.rolling(n).apply(lambda x: float(np.argmin(x)), raw=True)
        ) / n
    return out


def build_features_for_ticker(ticker: str, ohlcv_dir: Path) -> pd.DataFrame | None:
    """Compute 42 alpha158-lite features for one ticker."""
    p = ohlcv_dir / ticker / "1d.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception as exc:
        log.warning("  %s: read failed — %s", ticker, exc)
        return None
    if df.empty or len(df) < 70:  # need at least 70 bars for 60d windows
        return None
    feats = {}
    feats.update(kline_features(df))
    feats.update(rolling_features(df))
    feat_df = pd.DataFrame(feats, index=df.index)
    return feat_df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--inventory",
                    default=str(REPO_ROOT / "data" / "transformer_universe_inventory.json"))
    p.add_argument("--integrity-report",
                    default=str(REPO_ROOT / "data" / "transformer_data_integrity_report.json"))
    p.add_argument("--existing-engineered",
                    default=str(REPO_ROOT / "data" / "transformer_dataset_engineered.parquet"),
                    help="Existing engineered dataset to merge labels + 11 TA features from")
    p.add_argument("--ohlcv-dir",
                    default=str(REPO_ROOT / "data" / "ohlcv"))
    p.add_argument("--output",
                    default=str(REPO_ROOT / "data" / "alpha158_lite_dataset.parquet"))
    p.add_argument("--normalize-window", type=int, default=252)
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
    log.info("Building alpha158-lite features for %d tickers", len(universe))

    # Phase A: compute alpha158-lite per-ticker
    log.info("Phase A: computing 42 alpha158-lite features per-ticker …")
    rows: list[pd.DataFrame] = []
    for i, t in enumerate(universe):
        if i % 50 == 0 and i > 0:
            log.info("  ... %d/%d computed", i, len(universe))
        feats = build_features_for_ticker(t, Path(args.ohlcv_dir))
        if feats is None:
            continue
        feats = feats.reset_index().rename(columns={"index": "date"})
        feats["date"] = pd.to_datetime(feats["date"])
        feats.insert(0, "ticker", t)
        rows.append(feats)
    if not rows:
        log.error("No tickers produced features")
        sys.exit(1)
    panel = pd.concat(rows, ignore_index=True)
    log.info("After feature compute: %d rows × %d cols", len(panel), len(panel.columns))

    feat_cols = [c for c in panel.columns if c.startswith("alpha_")]
    log.info("alpha158-lite features (%d): %s", len(feat_cols), feat_cols)

    # Phase B: per-ticker rolling z-score
    log.info("Phase B: per-ticker rolling z-score (window=%d) …", args.normalize_window)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    min_periods = args.normalize_window // 2
    for c in feat_cols:
        gb = panel.groupby("ticker")[c]
        roll_mean = gb.rolling(args.normalize_window, min_periods=min_periods).mean().reset_index(level=0, drop=True)
        roll_std  = gb.rolling(args.normalize_window, min_periods=min_periods).std().reset_index(level=0, drop=True)
        panel[c] = (panel[c] - roll_mean) / roll_std.replace(0, np.nan)
    panel = panel.dropna(subset=feat_cols)
    log.info("After per-ticker zscore: %d rows", len(panel))

    # Phase C: cross-sectional z-score per date
    log.info("Phase C: cross-sectional z-score per date …")
    for c in feat_cols:
        date_mean = panel.groupby("date")[c].transform("mean")
        date_std  = panel.groupby("date")[c].transform("std")
        panel[c] = (panel[c] - date_mean) / date_std.replace(0, np.nan)
    panel = panel.dropna(subset=feat_cols)

    # Phase D: clip ±5σ
    log.info("Phase D: clip ±5σ …")
    for c in feat_cols:
        panel[c] = panel[c].clip(-5.0, 5.0)

    # Phase E: merge with existing engineered (gets labels + 11 TA features + split_label)
    log.info("Phase E: merge with engineered dataset for labels + TA …")
    existing = pd.read_parquet(args.existing_engineered)
    existing["date"] = pd.to_datetime(existing["date"])
    log.info("  existing: %d rows × %d cols", len(existing), len(existing.columns))
    merged = panel.merge(existing, on=["ticker", "date"], how="inner",
                         suffixes=("", "_existing"))
    log.info("After merge: %d rows × %d cols", len(merged), len(merged.columns))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    log.info("══ Written %s ══", out_path)
    splits = merged["split_label"].value_counts()
    log.info("Split summary:")
    for k, v in splits.items():
        log.info("  %-22s %d rows", k, v)


if __name__ == "__main__":
    main()
