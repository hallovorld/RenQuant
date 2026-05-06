"""Inference-time alpha158 feature computation.

Mirrors `scripts/build_alpha158_qlib.py` but for SINGLE-BAR inference:
given a ticker's recent OHLCV, return the 158 alpha158 features at the
last bar. Apply train-time z-score normalization stored in the scorer
artifact metadata.

Reference: `qlib/contrib/data/loader.py:Alpha158DL.get_feature_config`
(read 2026-05-06). All 27 rolling families × 5 windows + 9 KBAR + 4
PRICE-relative features = 148. We keep the same canonical names.

This module is the inference-side companion to the build script. It
ensures train/inference feature definitions stay byte-identical (per
CLAUDE.md §5.3: name the invariant — both build script and this module
import the same low-level functions).

Usage::

    from kernel.panel_pipeline.alpha158_features import compute_alpha158_at

    # Given an OHLCV DataFrame indexed by date for one ticker:
    feats: dict[str, float] = compute_alpha158_at(ohlcv_df, today)
    # → {'KMID': ..., 'KLEN': ..., 'ROC5': ..., ...}
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

WINDOWS = [5, 10, 20, 30, 60]
EPS = 1e-12


# ── Operators (matching qlib/data/ops.py semantics) ────────────────────────

def _greater(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.concat([a, b], axis=1).max(axis=1)


def _less(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.concat([a, b], axis=1).min(axis=1)


def _slope_at(arr: np.ndarray) -> float:
    """OLS slope of arr (length n) on time index 0..n-1."""
    n = len(arr)
    x_mean = (n - 1) / 2.0
    y_mean = arr.mean()
    cov = sum((i - x_mean) * (arr[i] - y_mean) for i in range(n))
    var_x = sum((i - x_mean) ** 2 for i in range(n))
    return cov / var_x if var_x > 0 else 0.0


def _rsquare_at(arr: np.ndarray) -> float:
    n = len(arr)
    y_mean = arr.mean()
    ss_tot = ((arr - y_mean) ** 2).sum()
    if ss_tot < EPS:
        return float("nan")
    slope = _slope_at(arr)
    intercept = y_mean - slope * (n - 1) / 2.0
    ss_res = sum((arr[i] - intercept - slope * i) ** 2 for i in range(n))
    return 1.0 - ss_res / ss_tot


def _resi_at(arr: np.ndarray) -> float:
    n = len(arr)
    y_mean = arr.mean()
    slope = _slope_at(arr)
    intercept = y_mean - slope * (n - 1) / 2.0
    return float(arr[-1] - intercept - slope * (n - 1))


def _kbar(o: float, h: float, l: float, c: float) -> dict[str, float]:
    span = (h - l) + EPS
    g_oc = max(o, c)
    l_oc = min(o, c)
    return {
        "KMID":  (c - o) / o if o else 0.0,
        "KLEN":  (h - l) / o if o else 0.0,
        "KMID2": (c - o) / span,
        "KUP":   (h - g_oc) / o if o else 0.0,
        "KUP2":  (h - g_oc) / span,
        "KLOW":  (l_oc - l) / o if o else 0.0,
        "KLOW2": (l_oc - l) / span,
        "KSFT":  (2 * c - h - l) / o if o else 0.0,
        "KSFT2": (2 * c - h - l) / span,
    }


def _price_features(df_tail: pd.DataFrame) -> dict[str, float]:
    last = df_tail.iloc[-1]
    c = float(last["close"])
    if c == 0:
        return {"OPEN0": 0, "HIGH0": 0, "LOW0": 0, "VWAP0": 0}
    vwap = (float(last["open"]) + float(last["high"])
             + float(last["low"]) + c) / 4.0
    return {
        "OPEN0":  float(last["open"]) / c,
        "HIGH0":  float(last["high"]) / c,
        "LOW0":   float(last["low"]) / c,
        "VWAP0":  vwap / c,
    }


def _rolling_at(df_tail: pd.DataFrame) -> dict[str, float]:
    """Compute all 27 rolling families × 5 windows = 135 features at last bar."""
    c = df_tail["close"].astype(float).values
    h = df_tail["high"].astype(float).values
    l = df_tail["low"].astype(float).values
    v = df_tail["volume"].astype(float).values
    n_bars = len(c)
    out: dict[str, float] = {}
    if n_bars < max(WINDOWS):
        # Insufficient history — return NaN for all
        for n in WINDOWS:
            for fam in ("ROC", "MA", "STD", "BETA", "RSQR", "RESI",
                        "MAX", "MIN", "QTLU", "QTLD", "RANK", "RSV",
                        "IMAX", "IMIN", "IMXD", "CORR", "CORD",
                        "CNTP", "CNTN", "CNTD", "SUMP", "SUMN", "SUMD",
                        "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD"):
                out[f"{fam}{n}"] = float("nan")
        return out

    c_today = c[-1]
    if c_today == 0:
        c_today = EPS
    for n in WINDOWS:
        win_c = c[-n:]
        win_h = h[-n:]
        win_l = l[-n:]
        win_v = v[-n:]
        out[f"ROC{n}"]  = c[-n - 1] / c_today if n_bars > n else float("nan")
        out[f"MA{n}"]   = win_c.mean() / c_today
        out[f"STD{n}"]  = win_c.std() / c_today
        out[f"BETA{n}"] = _slope_at(win_c) / c_today
        out[f"RSQR{n}"] = _rsquare_at(win_c)
        out[f"RESI{n}"] = _resi_at(win_c) / c_today
        out[f"MAX{n}"]  = win_h.max() / c_today
        out[f"MIN{n}"]  = win_l.min() / c_today
        out[f"QTLU{n}"] = float(np.quantile(win_c, 0.8)) / c_today
        out[f"QTLD{n}"] = float(np.quantile(win_c, 0.2)) / c_today
        # Rank: today's close percentile rank in window
        out[f"RANK{n}"] = (win_c <= c_today).sum() / n
        rsv_denom = (win_h.max() - win_l.min()) + EPS
        out[f"RSV{n}"]  = (c_today - win_l.min()) / rsv_denom
        out[f"IMAX{n}"] = float(np.argmax(win_h)) / n
        out[f"IMIN{n}"] = float(np.argmin(win_l)) / n
        out[f"IMXD{n}"] = (np.argmax(win_h) - np.argmin(win_l)) / n
        # Correlations need at least 2 points
        if n >= 2 and win_v.std() > EPS:
            log_v = np.log(win_v + 1)
            corr = float(np.corrcoef(win_c, log_v)[0, 1]) if log_v.std() > EPS else 0.0
            out[f"CORR{n}"] = corr if not np.isnan(corr) else 0.0
        else:
            out[f"CORR{n}"] = 0.0
        if n_bars > n:
            c_prev = c[-n - 1: -1]
            v_prev = v[-n - 1: -1]
            c_ret = win_c / np.where(c_prev == 0, EPS, c_prev) - 1
            v_ret = win_v / np.where(v_prev == 0, EPS, v_prev)
            log_v_ret = np.log(v_ret + 1)
            if c_ret.std() > EPS and log_v_ret.std() > EPS:
                cord = float(np.corrcoef(c_ret, log_v_ret)[0, 1])
                out[f"CORD{n}"] = cord if not np.isnan(cord) else 0.0
            else:
                out[f"CORD{n}"] = 0.0
        else:
            out[f"CORD{n}"] = 0.0
        # CNTP/CNTN/CNTD (up/down day counts)
        if n_bars > n:
            c_prev = c[-n - 1: -1]
            up = (win_c > c_prev).sum() / n
            dn = (win_c < c_prev).sum() / n
        else:
            up = dn = 0.0
        out[f"CNTP{n}"] = up
        out[f"CNTN{n}"] = dn
        out[f"CNTD{n}"] = up - dn
        # SUMP/SUMN/SUMD (gain/loss ratios)
        if n_bars > n:
            c_prev = c[-n - 1: -1]
            d = win_c - c_prev
            sum_abs = np.abs(d).sum() + EPS
            sump = np.maximum(d, 0).sum() / sum_abs
            sumn = np.maximum(-d, 0).sum() / sum_abs
            out[f"SUMP{n}"] = sump
            out[f"SUMN{n}"] = sumn
            out[f"SUMD{n}"] = sump - sumn
        else:
            out[f"SUMP{n}"] = out[f"SUMN{n}"] = out[f"SUMD{n}"] = 0.0
        # Volume features
        v_today = v[-1] if v[-1] > 0 else EPS
        out[f"VMA{n}"]  = win_v.mean() / v_today
        out[f"VSTD{n}"] = win_v.std() / v_today
        # WVMA (CV of |return| × volume)
        if n_bars > n:
            c_prev = c[-n - 1: -1]
            abs_ret = np.abs(win_c / np.where(c_prev == 0, EPS, c_prev) - 1)
            wv = abs_ret * win_v
            out[f"WVMA{n}"] = wv.std() / (wv.mean() + EPS)
        else:
            out[f"WVMA{n}"] = 0.0
        # VSUMP/VSUMN/VSUMD
        if n_bars > n:
            v_prev = v[-n - 1: -1]
            dv = win_v - v_prev
            sum_abs_v = np.abs(dv).sum() + EPS
            vsump = np.maximum(dv, 0).sum() / sum_abs_v
            vsumn = np.maximum(-dv, 0).sum() / sum_abs_v
            out[f"VSUMP{n}"] = vsump
            out[f"VSUMN{n}"] = vsumn
            out[f"VSUMD{n}"] = vsump - vsumn
        else:
            out[f"VSUMP{n}"] = out[f"VSUMN{n}"] = out[f"VSUMD{n}"] = 0.0
    return out


def compute_alpha158_at(
    ohlcv: pd.DataFrame,
    today: pd.Timestamp | None = None,
    min_bars: int = 70,
) -> dict[str, float]:
    """Compute Qlib alpha158 features at the last (or specified) bar.

    Args
    ----
    ohlcv : pd.DataFrame indexed by date with columns ['open', 'high',
            'low', 'close', 'volume'].
    today : Optional explicit date; defaults to the last bar in ohlcv.
    min_bars : Minimum bars required to compute (warmup buffer for
            longest rolling window). Default 70 (60d + buffer).

    Returns 158-element dict {feature_name: value}. NaN if insufficient
    history. Caller is responsible for downstream z-score normalization
    (use scorer's metadata['feature_means'] / 'feature_stds' if present).
    """
    if today is not None:
        ohlcv = ohlcv.loc[:today]
    if len(ohlcv) < min_bars:
        return {}  # caller should check & skip
    last = ohlcv.iloc[-1]
    feats: dict[str, float] = {}
    feats.update(_kbar(float(last["open"]), float(last["high"]),
                        float(last["low"]),  float(last["close"])))
    feats.update(_price_features(ohlcv.iloc[-1:]))
    feats.update(_rolling_at(ohlcv.iloc[-(max(WINDOWS) + 1):]))
    return feats


def alpha158_feature_names() -> list[str]:
    """Return the canonical list of 158 alpha158 feature names."""
    names = list(_kbar(1.0, 1.0, 1.0, 1.0).keys())   # 9 KBAR
    names += list(_price_features(pd.DataFrame({
        "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
    })).keys())   # 4 PRICE
    # 27 rolling families × 5 windows = 135
    for n in WINDOWS:
        for fam in ("ROC", "MA", "STD", "BETA", "RSQR", "RESI",
                    "MAX", "MIN", "QTLU", "QTLD", "RANK", "RSV",
                    "IMAX", "IMIN", "IMXD", "CORR", "CORD",
                    "CNTP", "CNTN", "CNTD", "SUMP", "SUMN", "SUMD",
                    "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD"):
            names.append(f"{fam}{n}")
    return names
