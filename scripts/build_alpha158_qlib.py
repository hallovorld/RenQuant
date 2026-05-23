#!/usr/bin/env python
"""Faithful Qlib alpha158 implementation, adapted to RenQuant.

Source: github.com/microsoft/qlib  (commit @ 2026-05-06 clone)
  - qlib/contrib/data/loader.py:Alpha158DL.get_feature_config (Lines 73-302)
  - qlib/contrib/data/handler.py:Alpha158 + DEFAULT processors (Lines 37-152)
  - qlib/contrib/model/linear.py:LinearModel — MSE via sklearn LinearRegression

Read these files line-by-line on 2026-05-06; this implementation matches
the canonical Qlib formulas. Every deviation is documented inline.

## Features (148 total)

### KBAR (9)
KMID  = (close-open)/open
KLEN  = (high-low)/open
KMID2 = (close-open)/(high-low+ε)
KUP   = (high-Greater(open,close))/open
KUP2  = (high-Greater(open,close))/(high-low+ε)
KLOW  = (Less(open,close)-low)/open
KLOW2 = (Less(open,close)-low)/(high-low+ε)
KSFT  = (2*close-high-low)/open
KSFT2 = (2*close-high-low)/(high-low+ε)

### PRICE (4) — windows=[0]
OPEN0  = open/close
HIGH0  = high/close
LOW0   = low/close
VWAP0  = vwap/close   (vwap = (open+high+low+close)/4 since Alpaca lacks true VWAP)

### ROLLING (27 families × 5 windows = 135)
ROC[n]   = Ref($close, n) / $close              # past_close / today_close
MA[n]    = Mean($close, n) / $close
STD[n]   = Std($close, n) / $close
BETA[n]  = Slope($close, n) / $close
RSQR[n]  = Rsquare($close, n)
RESI[n]  = Resi($close, n) / $close
MAX[n]   = Max($high, n) / $close
MIN[n]   = Min($low, n) / $close                  # named LOW in Qlib but renamed MIN here
QTLU[n]  = Quantile($close, n, 0.8) / $close
QTLD[n]  = Quantile($close, n, 0.2) / $close
RANK[n]  = Rank($close, n)
RSV[n]   = ($close-Min($low,n))/(Max($high,n)-Min($low,n)+ε)
IMAX[n]  = IdxMax($high, n) / n
IMIN[n]  = IdxMin($low, n) / n
IMXD[n]  = (IdxMax($high,n) - IdxMin($low,n)) / n
CORR[n]  = Corr($close, Log($volume+1), n)         # Qlib uses LOG volume!
CORD[n]  = Corr($close/Ref($close,1), Log($volume/Ref($volume,1)+1), n)
CNTP[n]  = Mean($close > Ref($close,1), n)
CNTN[n]  = Mean($close < Ref($close,1), n)
CNTD[n]  = CNTP - CNTN
SUMP[n]  = Sum(Greater($close-Ref(close,1), 0), n) / Sum(Abs($close-Ref(close,1)), n)
SUMN[n]  = Sum(Greater(Ref(close,1)-$close, 0), n) / Sum(Abs($close-Ref(close,1)), n)
SUMD[n]  = SUMP - SUMN
VMA[n]   = Mean($volume, n) / $volume
VSTD[n]  = Std($volume, n) / $volume
WVMA[n]  = Std(Abs($close/Ref($close,1)-1)*$volume, n) / Mean(...)   # CV not just mean
VSUMP[n] = Sum(Greater($volume-Ref(vol,1), 0), n) / Sum(Abs($volume-Ref(vol,1)), n)
VSUMN[n] = Sum(Greater(Ref(vol,1)-$volume, 0), n) / Sum(Abs($volume-Ref(vol,1)), n)
VSUMD[n] = VSUMP - VSUMN

Note: Qlib's default uses windows = [5, 10, 20, 30, 60].

## Processor (faithful Qlib)
- INFER (features): ProcessInf + ZScoreNorm (PER FEATURE, GLOBAL) + Fillna(0)
- LEARN (label): DropnaLabel + CSZScoreNorm (cross-sectional z per date)

## Adaptations to RenQuant
- 290 tickers (vs Qlib csi500 = 500)
- Use fwd_5d_excess as primary label (RenQuant native horizon)
- Also save fwd_20d, fwd_60d for multi-horizon experiments
- vwap is approximated as (O+H+L+C)/4 since OHLCV doesn't have true VWAP
- Same train/val/test split as transformer_dataset_*.py for apples-to-apples
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_THREAD_COUNT = str(os.cpu_count() or 14)
for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, _THREAD_COUNT)

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("alpha158-qlib")

WINDOWS = [5, 10, 20, 30, 60]
EPS = 1e-12


# ── Operators (matching qlib/data/ops.py semantics) ────────────────────────

def _slope(s: pd.Series, n: int) -> pd.Series:
    """Rolling OLS slope of s on index (0..n-1). Vectorized."""
    # Slope = cov(x, y) / var(x) where x = 0..n-1
    x_mean = (n - 1) / 2.0
    var_x = sum((i - x_mean) ** 2 for i in range(n))  # constant
    def slope_fn(arr):
        if np.isnan(arr).any():
            return np.nan
        y_mean = arr.mean()
        return sum((i - x_mean) * (arr[i] - y_mean) for i in range(n)) / var_x
    return s.rolling(n).apply(slope_fn, raw=True)


def _rsquare(s: pd.Series, n: int) -> pd.Series:
    """Rolling R² of OLS s = a + b·t."""
    x_mean = (n - 1) / 2.0
    def rsq_fn(arr):
        if np.isnan(arr).any():
            return np.nan
        y_mean = arr.mean()
        ss_tot = ((arr - y_mean) ** 2).sum()
        if ss_tot < EPS:
            return np.nan
        slope = sum((i - x_mean) * (arr[i] - y_mean) for i in range(n)) / sum((i - x_mean) ** 2 for i in range(n))
        intercept = y_mean - slope * x_mean
        ss_res = sum((arr[i] - intercept - slope * i) ** 2 for i in range(n))
        return 1.0 - ss_res / ss_tot
    return s.rolling(n).apply(rsq_fn, raw=True)


def _resi(s: pd.Series, n: int) -> pd.Series:
    """Rolling residual at the end of OLS s = a + b·t."""
    x_mean = (n - 1) / 2.0
    def resi_fn(arr):
        if np.isnan(arr).any():
            return np.nan
        y_mean = arr.mean()
        slope = sum((i - x_mean) * (arr[i] - y_mean) for i in range(n)) / sum((i - x_mean) ** 2 for i in range(n))
        intercept = y_mean - slope * x_mean
        # Residual at last point i = n-1
        return arr[-1] - intercept - slope * (n - 1)
    return s.rolling(n).apply(resi_fn, raw=True)


def _idx_max_n(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).apply(lambda x: float(np.argmax(x)), raw=True)


def _idx_min_n(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).apply(lambda x: float(np.argmin(x)), raw=True)


def _greater(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise max."""
    return pd.concat([a, b], axis=1).max(axis=1)


def _less(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise min."""
    return pd.concat([a, b], axis=1).min(axis=1)


# ── Feature builder ────────────────────────────────────────────────────────

def kbar_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """9 K-bar features (lines 105-126 of qlib loader.py)."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    span = (h - l) + EPS
    g_oc = _greater(o, c)  # max(open, close)
    l_oc = _less(o, c)     # min(open, close)
    return {
        "KMID":  (c - o) / o,
        "KLEN":  (h - l) / o,
        "KMID2": (c - o) / span,
        "KUP":   (h - g_oc) / o,
        "KUP2":  (h - g_oc) / span,
        "KLOW":  (l_oc - l) / o,
        "KLOW2": (l_oc - l) / span,
        "KSFT":  (2 * c - h - l) / o,
        "KSFT2": (2 * c - h - l) / span,
    }


def price_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """4 price features at window=0 (lines 128-133)."""
    c = df["close"]
    vwap = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    return {
        "OPEN0":  df["open"] / c,
        "HIGH0":  df["high"] / c,
        "LOW0":   df["low"] / c,
        "VWAP0":  vwap / c,
    }


def rolling_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """27 rolling families × 5 windows = 135 features."""
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)

    # Pre-compute commonly used quantities
    c_lag1 = c.shift(1)
    c_diff = c - c_lag1
    abs_c_diff = c_diff.abs()
    log_v = np.log(v + 1)
    c_ret = c / c_lag1 - 1
    abs_c_ret = c_ret.abs()
    c_ret_norm = c / c_lag1
    v_ret_norm = v / v.shift(1)
    log_v_ret = np.log(v_ret_norm + 1)
    v_diff = v - v.shift(1)
    abs_v_diff = v_diff.abs()

    out: dict[str, pd.Series] = {}
    for n in WINDOWS:
        # ROC: past close / today close (Qlib's exact formula)
        out[f"ROC{n}"]   = c.shift(n) / c
        out[f"MA{n}"]    = c.rolling(n).mean() / c
        out[f"STD{n}"]   = c.rolling(n).std() / c
        out[f"BETA{n}"]  = _slope(c, n) / c
        out[f"RSQR{n}"]  = _rsquare(c, n)
        out[f"RESI{n}"]  = _resi(c, n) / c
        out[f"MAX{n}"]   = h.rolling(n).max() / c
        out[f"MIN{n}"]   = l.rolling(n).min() / c
        out[f"QTLU{n}"]  = c.rolling(n).quantile(0.8) / c
        out[f"QTLD{n}"]  = c.rolling(n).quantile(0.2) / c
        out[f"RANK{n}"]  = c.rolling(n).rank(pct=True)
        out[f"RSV{n}"]   = (c - l.rolling(n).min()) / (h.rolling(n).max() - l.rolling(n).min() + EPS)
        out[f"IMAX{n}"]  = _idx_max_n(h, n) / n
        out[f"IMIN{n}"]  = _idx_min_n(l, n) / n
        out[f"IMXD{n}"]  = (_idx_max_n(h, n) - _idx_min_n(l, n)) / n
        out[f"CORR{n}"]  = c.rolling(n).corr(log_v)
        out[f"CORD{n}"]  = c_ret_norm.rolling(n).corr(log_v_ret)
        out[f"CNTP{n}"]  = (c > c_lag1).astype(float).rolling(n).mean()
        out[f"CNTN{n}"]  = (c < c_lag1).astype(float).rolling(n).mean()
        out[f"CNTD{n}"]  = out[f"CNTP{n}"] - out[f"CNTN{n}"]
        # SUMP/SUMN: Sum of positive/negative returns / Sum of abs returns
        pos_ret = c_diff.clip(lower=0)
        neg_ret = (-c_diff).clip(lower=0)
        sum_abs = abs_c_diff.rolling(n).sum() + EPS
        out[f"SUMP{n}"] = pos_ret.rolling(n).sum() / sum_abs
        out[f"SUMN{n}"] = neg_ret.rolling(n).sum() / sum_abs
        out[f"SUMD{n}"] = out[f"SUMP{n}"] - out[f"SUMN{n}"]
        # Per §5.13.11: zero-volume days (halts/delistings) produce ~1e16
        # ratios when `v + EPS` is used as denominator (EPS=1e-12 → bid_avg/1e-12).
        # Use rolling-mean as denominator floor; final fallback to 1.0 prevents
        # inf when first rolling window is also all zero. Per §5.13.5: single
        # denominator-floor implementation shared by all VMA/VSTD windows.
        v_safe = v.where(np.isfinite(v) & (v > 0), v.rolling(20, min_periods=1).mean())
        v_safe = v_safe.where(np.isfinite(v_safe) & (v_safe > 0), 1.0)
        out[f"VMA{n}"]  = v.rolling(n).mean() / v_safe
        out[f"VSTD{n}"] = v.rolling(n).std() / v_safe
        # WVMA: coefficient of variation of (|return| × volume)
        wv = abs_c_ret * v
        out[f"WVMA{n}"] = wv.rolling(n).std() / (wv.rolling(n).mean() + EPS)
        # VSUMP/N/D
        pos_v = v_diff.clip(lower=0)
        neg_v = (-v_diff).clip(lower=0)
        sum_abs_v = abs_v_diff.rolling(n).sum() + EPS
        out[f"VSUMP{n}"] = pos_v.rolling(n).sum() / sum_abs_v
        out[f"VSUMN{n}"] = neg_v.rolling(n).sum() / sum_abs_v
        out[f"VSUMD{n}"] = out[f"VSUMP{n}"] - out[f"VSUMN{n}"]
    return out


def build_features_for_ticker(ticker: str, ohlcv_dir: Path) -> pd.DataFrame | None:
    p = ohlcv_dir / ticker / "1d.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception as exc:
        log.warning("  %s: read failed — %s", ticker, exc)
        return None
    if df.empty or len(df) < 70:
        return None
    feats: dict[str, pd.Series] = {}
    feats.update(kbar_features(df))
    feats.update(price_features(df))
    feats.update(rolling_features(df))
    feat_df = pd.DataFrame(feats, index=df.index)
    return feat_df


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--inventory",
                    default=str(REPO_ROOT / "data" / "transformer_universe_inventory.json"))
    p.add_argument("--integrity-report",
                    default=str(REPO_ROOT / "data" / "transformer_data_integrity_report.json"))
    p.add_argument("--existing-engineered",
                    default=str(REPO_ROOT / "data" / "transformer_dataset_engineered.parquet"),
                    help="Existing dataset to merge labels + split_label from")
    p.add_argument("--ohlcv-dir",
                    default=str(REPO_ROOT / "data" / "ohlcv"))
    p.add_argument("--output",
                    default=str(REPO_ROOT / "data" / "alpha158_qlib_dataset.parquet"))
    p.add_argument("--tickers", type=int, default=0,
                   help="Limit to first N tickers (smoke testing)")
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
    if args.tickers > 0:
        universe = universe[:args.tickers]
    log.info("Building Qlib alpha158 (148 features) for %d tickers", len(universe))

    rows: list[pd.DataFrame] = []
    for i, t in enumerate(universe):
        if i % 50 == 0 and i > 0:
            log.info("  ... %d/%d computed", i, len(universe))
        feats = build_features_for_ticker(t, Path(args.ohlcv_dir))
        if feats is None:
            continue
        feats = feats.reset_index().rename(columns={"index": "date", "Date": "date"})
        feats["date"] = pd.to_datetime(feats["date"])
        feats.insert(0, "ticker", t)
        rows.append(feats)
    if not rows:
        log.error("No tickers produced features")
        sys.exit(1)
    panel = pd.concat(rows, ignore_index=True)
    feat_cols = [c for c in panel.columns if c not in ("ticker", "date")]
    log.info("Raw panel: %d rows × %d feature cols", len(panel), len(feat_cols))

    # ── Qlib processors (faithful) ──
    # ProcessInf: replace ±inf with NaN
    panel[feat_cols] = panel[feat_cols].replace([np.inf, -np.inf], np.nan)

    # ── Labels: compute from OHLCV directly for all tickers ──────────────────
    # Previously used inner-merge with existing dataset which silently dropped
    # any ticker not in the old 291-ticker universe. Now we compute fwd returns
    # from each ticker's OHLCV and excess vs SPY, covering all N tickers.
    log.info("Phase: compute labels (fwd_5d/20d/60d excess vs SPY) from OHLCV …")
    ohlcv_dir = Path(args.ohlcv_dir)
    spy_path = ohlcv_dir / "SPY" / "1d.parquet"
    if not spy_path.exists():
        log.error("SPY OHLCV not found at %s — cannot compute excess returns", spy_path)
        sys.exit(1)
    spy_df   = pd.read_parquet(spy_path)
    spy_close = spy_df["close"].sort_index().rename("spy_close")

    label_rows: list[pd.DataFrame] = []
    for ticker in panel["ticker"].unique():
        try:
            t_df = pd.read_parquet(ohlcv_dir / ticker / "1d.parquet")
        except Exception:
            continue
        c = t_df["close"].sort_index()
        spy_aligned = spy_close.reindex(c.index, method="ffill")
        rec: dict[str, pd.Series] = {"ticker": pd.Series(ticker, index=c.index),
                                      "date": pd.Series(c.index, index=c.index)}
        for n in (5, 20, 60):
            fwd_ticker = c.shift(-n) / c - 1
            fwd_spy    = spy_aligned.shift(-n) / spy_aligned - 1
            rec[f"fwd_{n}d_excess"] = fwd_ticker - fwd_spy
        label_rows.append(pd.DataFrame(rec))
    labels_panel = pd.concat(label_rows, ignore_index=True)
    labels_panel["date"] = pd.to_datetime(labels_panel["date"])
    panel = panel.merge(labels_panel, on=["ticker", "date"], how="inner")
    log.info("After label merge: %d rows, %d tickers", len(panel), panel["ticker"].nunique())

    # ── split_label: date-based cutoffs from existing dataset ─────────────────
    # Use the existing dataset's date→split mapping so train/val/test periods
    # are consistent — but apply to ALL tickers regardless of old universe.
    log.info("Phase: assign split_label from date cutoffs in %s …", args.existing_engineered)
    existing = pd.read_parquet(args.existing_engineered)
    existing["date"] = pd.to_datetime(existing["date"])
    date_split = (existing[["date", "split_label"]]
                  .drop_duplicates("date")
                  .set_index("date")["split_label"])
    panel["split_label"] = panel["date"].map(date_split)
    # Dates beyond existing dataset → assign to test (most recent data)
    panel["split_label"] = panel["split_label"].fillna("test")
    panel = panel.dropna(subset=["fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"])
    log.info("After split: %d rows, %d tickers", len(panel), panel["ticker"].nunique())

    train_mask = panel["split_label"] == "train"
    log.info("Phase: ZScoreNorm per feature (train-only stats) …")
    # Per §5.13.12: defense in depth. Replace inf/NaN, then winsorize at
    # 0.1%/99.9% (train-only quantiles) BEFORE computing mean/std. A single
    # 1e16 outlier in VMA/VSTD historically poisoned the column's stats
    # and collapsed normal values to a single constant after z-score.
    panel[feat_cols] = panel[feat_cols].replace([np.inf, -np.inf], np.nan)
    for c in feat_cols:
        train_col = panel.loc[train_mask, c]
        q_lo, q_hi = train_col.quantile([0.001, 0.999])
        if np.isfinite(q_lo) and np.isfinite(q_hi) and q_hi > q_lo:
            panel[c] = panel[c].clip(q_lo, q_hi)
    # Save per-feature train-only means/stds as sidecar for inference-time
    # reuse (PanelLinearScorer.score_raw needs these to normalize raw
    # alpha158 features computed live).
    feature_stats: dict[str, dict[str, float]] = {}
    for c in feat_cols:
        col_train = panel.loc[train_mask, c]
        m = float(col_train.mean())
        s = float(col_train.std())
        feature_stats[c] = {"mean": m, "std": s}
        if s > 1e-9:
            panel[c] = (panel[c] - m) / s
        else:
            panel[c] = panel[c] - m
    stats_path = Path(args.output).with_suffix(".stats.json")
    stats_path.write_text(json.dumps({
        "feature_cols": feat_cols,
        "feature_means": [feature_stats[c]["mean"] for c in feat_cols],
        "feature_stds":  [feature_stats[c]["std"]  for c in feat_cols],
        "n_train_rows": int(train_mask.sum()),
        "clip_sigma": 5.0,
    }, indent=2, default=str))
    log.info("Saved train-only feature stats → %s", stats_path)
    # Fillna(0) — Qlib default
    panel[feat_cols] = panel[feat_cols].fillna(0.0)
    # Clip extreme values (not in Qlib but consistent with our prior datasets)
    for c in feat_cols:
        panel[c] = panel[c].clip(-5.0, 5.0)

    # Per Qlib LEARN_PROCESSORS: CSZScoreNorm on label (cross-sectional z per date)
    log.info("Phase: cross-sectional z-score on labels per date …")
    for lbl in ("fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"):
        date_mean = panel.groupby("date")[lbl].transform("mean")
        date_std  = panel.groupby("date")[lbl].transform("std")
        panel[lbl] = (panel[lbl] - date_mean) / (date_std + EPS)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    log.info("══ Written %s ══", out_path)
    log.info("Final: %d rows × %d cols (148 features + labels + meta)", len(panel), len(panel.columns))
    splits = panel["split_label"].value_counts()
    for k, vv in splits.items():
        log.info("  %-22s %d rows", k, vv)


if __name__ == "__main__":
    main()
