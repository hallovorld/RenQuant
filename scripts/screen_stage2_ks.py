#!/usr/bin/env python
"""Track D Stage 2: KS distributional gate.

For each Stage-1-admitted candidate, compute per-feature distribution
stats from raw OHLCV (daily returns, vol, volume, momentum). Run a
KS test of each feature's empirical CDF against the wl103 reference
pool's CDF. Reject if median per-feature KS > threshold (default 0.20).

This Stage 2 uses OHLCV-derived signals (not the full 27-feature
production panel) for speed: testing 27 panel features would require
running PanelDataJob+PanelFeatureJob on 816 candidates (~30+ min).
The OHLCV stats — daily return, RV20, log-volume, mom_60d — are highly
correlated with the panel features and serve as a proxy gate.

Tighter than Witter 2025's 0.30 because we don't rely on L1+L2 to
rescue heterogeneity. Aim: candidates whose distributions match wl103
within KS=0.20 are likely to behave consistently with the existing
training panel.

Usage::

    python scripts/screen_stage2_ks.py
    python scripts/screen_stage2_ks.py --ks-max 0.15
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("screen-stage2-ks")


def _ticker_features(ticker: str, ohlcv_root: Path,
                     n_tail: int = 504) -> "dict | None":
    """Compute four per-ticker feature distributions from cached daily OHLCV.

    Returns a dict[str, np.ndarray] of feature → 1-D values (one per
    bar in the trailing `n_tail` window, ~2 yrs at 252 days/yr). NaNs
    dropped per feature.
    """
    p = ohlcv_root / ticker / "1d.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    if df.empty or "close" not in df.columns:
        return None
    df = df.tail(n_tail)
    if len(df) < 60:
        return None

    close = df["close"].astype(float).replace(0, np.nan).dropna()
    daily_ret = close.pct_change().dropna()
    if len(daily_ret) < 30:
        return None

    # 4 distributions:
    feats: dict[str, np.ndarray] = {}
    feats["daily_return"] = daily_ret.to_numpy()
    feats["realized_vol_20d"] = (
        daily_ret.rolling(20, min_periods=20).std().dropna().to_numpy()
    )
    if "volume" in df.columns:
        log_vol = np.log(df["volume"].replace(0, np.nan).dropna())
        feats["log_volume"] = log_vol.to_numpy()
    # 60-day momentum (close ratio)
    mom_60d = (close / close.shift(60) - 1.0).dropna().to_numpy()
    if len(mom_60d) > 0:
        feats["mom_60d"] = mom_60d
    return feats


def _build_reference_pool(ref_tickers: list[str], ohlcv_root: Path) -> dict[str, np.ndarray]:
    """Concatenate per-feature values across all reference tickers.

    Output: dict[feature_name, concatenated 1-D ndarray]. Used as the
    "wl103 reference distribution" for KS comparison.
    """
    bins: dict[str, list[np.ndarray]] = {}
    for tk in ref_tickers:
        ft = _ticker_features(tk, ohlcv_root)
        if ft is None:
            continue
        for name, arr in ft.items():
            bins.setdefault(name, []).append(arr)
    return {name: np.concatenate(arrs) for name, arrs in bins.items() if arrs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-results",
                   default=str(REPO_ROOT / "scripts" / "screen_stage1_results.json"))
    p.add_argument("--ohlcv-root",
                   default=str(REPO_ROOT / "data" / "ohlcv"))
    p.add_argument("--reference-config",
                   default=str(REPO_ROOT / "backtesting" / "renquant_104"
                               / "strategy_config.golden.json"),
                   help="Strategy config whose `watchlist` is the reference pool (wl103 by default).")
    p.add_argument("--out",
                   default=str(REPO_ROOT / "scripts" / "screen_stage2_results.json"))
    p.add_argument("--ks-max", type=float, default=0.20,
                   help="Reject candidate if median per-feature KS > this. "
                        "Tighter than Witter 2025 (0.30); aim for distributional "
                        "compatibility with wl103 pool.")
    args = p.parse_args()

    s1 = json.loads(Path(args.stage1_results).read_text())
    if s1.get("kind") != "track_d_stage1_mechanical":
        log.error("Stage 1 results JSON has unexpected kind: %s", s1.get("kind"))
        return 1
    candidates = [r["ticker"] for r in s1["admitted"]]
    log.info("Stage 2 inputs: %d candidates from Stage 1", len(candidates))

    ref_cfg = json.loads(Path(args.reference_config).read_text())
    ref_tickers = ref_cfg["watchlist"]
    log.info("Reference pool (wl103): %d tickers", len(ref_tickers))

    ohlcv_root = Path(args.ohlcv_root)
    ref_pool = _build_reference_pool(ref_tickers, ohlcv_root)
    log.info("Reference pool features: %s",
             {k: len(v) for k, v in ref_pool.items()})

    admitted: list[dict] = []
    rejected: list[dict] = []
    for tk in candidates:
        if tk in ref_tickers:
            # Already in production, skip (would compare against itself)
            continue
        ft = _ticker_features(tk, ohlcv_root)
        if ft is None:
            rejected.append({"ticker": tk, "reason": "no_features"})
            continue
        ks_per_feature: dict[str, float] = {}
        for name, candidate_arr in ft.items():
            ref_arr = ref_pool.get(name)
            if ref_arr is None or len(candidate_arr) < 30:
                continue
            ks_stat = sstats.ks_2samp(candidate_arr, ref_arr).statistic
            ks_per_feature[name] = float(ks_stat)
        if not ks_per_feature:
            rejected.append({"ticker": tk, "reason": "no_overlapping_features"})
            continue
        median_ks = float(np.median(list(ks_per_feature.values())))
        max_ks = float(np.max(list(ks_per_feature.values())))
        row = {
            "ticker":     tk,
            "median_ks":  median_ks,
            "max_ks":     max_ks,
            "ks_by_feature": ks_per_feature,
        }
        if median_ks > args.ks_max:
            rejected.append({**row, "reason": "above_ks_threshold"})
        else:
            admitted.append(row)

    log.info("Stage 2 result: %d admitted / %d rejected (input=%d, threshold ks≤%.2f)",
             len(admitted), len(rejected), len(candidates) - sum(1 for tk in candidates if tk in ref_tickers),
             args.ks_max)

    if admitted:
        median_ks_admitted = np.median([r["median_ks"] for r in admitted])
        log.info("Admitted median KS: %.4f", median_ks_admitted)

    out = {
        "kind": "track_d_stage2_ks_distributional",
        "ks_threshold":   args.ks_max,
        "admitted_count": len(admitted),
        "rejected_count": len(rejected),
        "admitted":       admitted,
        "rejected":       rejected,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
