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
# pandas rolling.std() defaults to ddof=1.  The training builder uses that
# Qlib-compatible sample standard deviation, so inference must do the same.
STD_DDOF = 1


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
        out[f"STD{n}"]  = win_c.std(ddof=STD_DDOF) / c_today
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
        # Per §5.3: same denominator-floor invariant as `scripts/build_alpha158_qlib.py`.
        # If today's volume is zero / NaN (halt / delisting), fall back to a
        # rolling-mean of the prior 20 bars; if that is also zero, fall back to
        # 1.0. This avoids the 1e16 explosion that the EPS=1e-12 fallback
        # produced when v_today=0 (root cause of feature_stds blowup; §5.13.11).
        v_last = v[-1]
        if np.isfinite(v_last) and v_last > 0:
            v_today = float(v_last)
        else:
            # Rolling mean of up to 20 prior bars (matches build script's window=20)
            prior = v[-min(20, n_bars):]
            prior = prior[np.isfinite(prior) & (prior > 0)]
            v_today = float(prior.mean()) if prior.size > 0 else 1.0
        out[f"VMA{n}"]  = win_v.mean() / v_today
        out[f"VSTD{n}"] = win_v.std(ddof=STD_DDOF) / v_today
        # WVMA (CV of |return| × volume)
        if n_bars > n:
            c_prev = c[-n - 1: -1]
            abs_ret = np.abs(win_c / np.where(c_prev == 0, EPS, c_prev) - 1)
            wv = abs_ret * win_v
            out[f"WVMA{n}"] = wv.std(ddof=STD_DDOF) / (wv.mean() + EPS)
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


def _rolling_apply(s: pd.Series, window: int, fn) -> pd.Series:
    return s.rolling(window, min_periods=window).apply(fn, raw=True)


def compute_alpha158_frame(
    ohlcv: pd.DataFrame,
    min_bars: int = 70,
) -> pd.DataFrame:
    """Compute the full causal alpha158 panel for one ticker.

    This is the vectorized/cache companion to :func:`compute_alpha158_at`.
    For any date ``t`` with enough warmup:

    ``compute_alpha158_frame(df).loc[:t].iloc[-1] == compute_alpha158_at(df.loc[:t])``

    The invariant is pinned by ``tests/test_feature_cache.py`` because sim
    performance must not buy speed with a live/sim feature drift.
    """
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame(columns=alpha158_feature_names())
    df = ohlcv.sort_index().copy()
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    o = df["open"].astype(float)
    v = df["volume"].astype(float)

    cols: dict[str, pd.Series] = {}
    span = (h - l) + EPS
    g_oc = pd.Series(np.maximum(o.to_numpy(), c.to_numpy()), index=df.index)
    l_oc = pd.Series(np.minimum(o.to_numpy(), c.to_numpy()), index=df.index)
    open_safe = o.where(o != 0)

    cols["KMID"] = ((c - o) / open_safe).fillna(0.0)
    cols["KLEN"] = ((h - l) / open_safe).fillna(0.0)
    cols["KMID2"] = (c - o) / span
    cols["KUP"] = ((h - g_oc) / open_safe).fillna(0.0)
    cols["KUP2"] = (h - g_oc) / span
    cols["KLOW"] = ((l_oc - l) / open_safe).fillna(0.0)
    cols["KLOW2"] = (l_oc - l) / span
    cols["KSFT"] = ((2 * c - h - l) / open_safe).fillna(0.0)
    cols["KSFT2"] = (2 * c - h - l) / span

    close_safe = c.where(c != 0)
    vwap = (o + h + l + c) / 4.0
    cols["OPEN0"] = (o / close_safe).fillna(0.0)
    cols["HIGH0"] = (h / close_safe).fillna(0.0)
    cols["LOW0"] = (l / close_safe).fillna(0.0)
    cols["VWAP0"] = (vwap / close_safe).fillna(0.0)

    ret = c / c.shift(1) - 1.0
    delta = c - c.shift(1)
    vol_delta = v - v.shift(1)
    vol_pos = v.where(np.isfinite(v) & (v > 0))
    vol_fallback = vol_pos.rolling(20, min_periods=1).mean().fillna(1.0)
    v_today = v.where(np.isfinite(v) & (v > 0), vol_fallback).fillna(1.0)
    log_v = np.log(v.clip(lower=0.0) + 1.0)
    v_ret = v / v.shift(1)
    log_v_ret = np.log(v_ret + 1.0).replace([np.inf, -np.inf], np.nan)

    for n in WINDOWS:
        roll_c = c.rolling(n, min_periods=n)
        roll_h = h.rolling(n, min_periods=n)
        roll_l = l.rolling(n, min_periods=n)
        roll_v = v.rolling(n, min_periods=n)

        cols[f"ROC{n}"] = c.shift(n) / close_safe
        cols[f"MA{n}"] = roll_c.mean() / close_safe
        cols[f"STD{n}"] = roll_c.std(ddof=STD_DDOF) / close_safe
        cols[f"BETA{n}"] = _rolling_apply(c, n, _slope_at) / close_safe
        cols[f"RSQR{n}"] = _rolling_apply(c, n, _rsquare_at)
        cols[f"RESI{n}"] = _rolling_apply(c, n, _resi_at) / close_safe
        cols[f"MAX{n}"] = roll_h.max() / close_safe
        cols[f"MIN{n}"] = roll_l.min() / close_safe
        cols[f"QTLU{n}"] = roll_c.quantile(0.8) / close_safe
        cols[f"QTLD{n}"] = roll_c.quantile(0.2) / close_safe
        cols[f"RANK{n}"] = _rolling_apply(
            c, n, lambda arr: float((arr <= arr[-1]).sum()) / len(arr)
        )
        cols[f"RSV{n}"] = (c - roll_l.min()) / ((roll_h.max() - roll_l.min()) + EPS)
        imax = _rolling_apply(h, n, lambda arr: float(np.argmax(arr)))
        imin = _rolling_apply(l, n, lambda arr: float(np.argmin(arr)))
        cols[f"IMAX{n}"] = imax / n
        cols[f"IMIN{n}"] = imin / n
        cols[f"IMXD{n}"] = (imax - imin) / n
        cols[f"CORR{n}"] = (
            c.rolling(n, min_periods=n).corr(log_v)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        cols[f"CORD{n}"] = (
            ret.rolling(n, min_periods=n).corr(log_v_ret)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        up = (c > c.shift(1)).astype(float)
        dn = (c < c.shift(1)).astype(float)
        cntp = up.rolling(n, min_periods=n).sum() / n
        cntn = dn.rolling(n, min_periods=n).sum() / n
        cols[f"CNTP{n}"] = cntp
        cols[f"CNTN{n}"] = cntn
        cols[f"CNTD{n}"] = cntp - cntn

        pos_delta = delta.clip(lower=0.0)
        neg_delta = (-delta).clip(lower=0.0)
        abs_delta = delta.abs()
        sum_abs = abs_delta.rolling(n, min_periods=n).sum() + EPS
        sump = pos_delta.rolling(n, min_periods=n).sum() / sum_abs
        sumn = neg_delta.rolling(n, min_periods=n).sum() / sum_abs
        cols[f"SUMP{n}"] = sump
        cols[f"SUMN{n}"] = sumn
        cols[f"SUMD{n}"] = sump - sumn

        cols[f"VMA{n}"] = roll_v.mean() / v_today
        cols[f"VSTD{n}"] = roll_v.std(ddof=STD_DDOF) / v_today
        wv = ret.abs() * v
        cols[f"WVMA{n}"] = (
            wv.rolling(n, min_periods=n).std(ddof=STD_DDOF)
            / (wv.rolling(n, min_periods=n).mean() + EPS)
        )
        pos_vd = vol_delta.clip(lower=0.0)
        neg_vd = (-vol_delta).clip(lower=0.0)
        abs_vd = vol_delta.abs()
        sum_abs_v = abs_vd.rolling(n, min_periods=n).sum() + EPS
        vsump = pos_vd.rolling(n, min_periods=n).sum() / sum_abs_v
        vsumn = neg_vd.rolling(n, min_periods=n).sum() / sum_abs_v
        cols[f"VSUMP{n}"] = vsump
        cols[f"VSUMN{n}"] = vsumn
        cols[f"VSUMD{n}"] = vsump - vsumn

    out = pd.DataFrame(cols, index=df.index).reindex(columns=alpha158_feature_names())
    if len(out) < min_bars:
        return out.iloc[0:0]
    return out.iloc[min_bars - 1:]


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
