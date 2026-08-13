"""Training-time feature frame construction for renquant_103.

Builds per-ticker labelled DataFrames with relative features vs SPY,
rolling regime-context columns, and forward-return labels.

Used by the tournament (Cell 7) to produce training data.
LEAN / live inference uses kernel.indicators.build_feature_frame() instead
(broadcast scalar context, no labels).
"""
from __future__ import annotations

import hashlib
import threading

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from kernel.indicators import compute_all as _compute_all

_RATIO_FEATURES = {"rsi", "adx"}
_DIFF_FEATURES  = {"macd_hist", "cci", "bbp", "williams_r", "obv_slope"}
_REQUIRED_OHLCV = ("open", "high", "low", "close", "volume")


def _spy_rets_fingerprint(spy_rets: pd.Series) -> bytes:
    """Content fingerprint of the exact ``spy_rets`` input to ``rolling_hurst``.

    Keyed on the index labels + float values so that two callers with an
    identical ``spy_rets`` (same dates, same returns) resolve to the same
    cache slot. Deterministic and collision-safe for this use (SHA-1 over the
    raw bytes of both arrays). NaNs carry numpy's canonical bit pattern, so
    equal inputs always hash equal.
    """
    h = hashlib.sha1()
    idx = np.ascontiguousarray(spy_rets.index.to_numpy())
    h.update(np.asarray(idx.shape, dtype=np.int64).tobytes())
    h.update(idx.tobytes())
    vals = np.ascontiguousarray(spy_rets.to_numpy(dtype="float64"))
    h.update(vals.tobytes())
    return h.digest()


class SpyHurstMemo:
    """Run-scoped memoizer for ``rolling_hurst`` on the SPY return series.

    ``build_training_features`` reindexes SPY returns to each ticker's
    ``common_idx`` and then runs a 63-day rolling Hurst — the dominant
    per-ticker cost. Within one ``prepare_inference_panel_frames`` run SPY
    OHLCV is a single shared frame, so every ticker that shares a
    ``common_idx`` feeds an *identical* ``spy_rets`` into ``rolling_hurst``.

    This caches the exact computation keyed by the content of that input.
    A cache hit returns the identical Series that ``rolling_hurst`` would
    have produced (the subsequent ``.reindex(common_idx)`` still runs per
    ticker), so the memoized path is output-invariant by construction — the
    only difference is that N identical computations collapse to one per
    distinct input.

    Deliberately instance-scoped (created per run, passed down) rather than a
    module global, so nothing leaks across runs/dates. Thread-safe: the
    per-ticker chain runs under a ThreadPoolExecutor, and double-checked
    locking guarantees each distinct input is computed exactly once.
    """

    __slots__ = ("_cache", "_lock", "compute_calls")

    def __init__(self) -> None:
        self._cache: dict[bytes, pd.Series] = {}
        self._lock = threading.Lock()
        self.compute_calls = 0  # count of real rolling_hurst evaluations

    def get_or_compute(self, spy_rets: pd.Series, compute) -> pd.Series:
        key = _spy_rets_fingerprint(spy_rets)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        with self._lock:
            hit = self._cache.get(key)
            if hit is None:
                hit = compute()
                self.compute_calls += 1
                self._cache[key] = hit
            return hit


def _canonical_ohlcv_frame(rows: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return sorted, unique-date OHLCV input safe for pandas reindex.

    Vendor/cache appends can leave duplicate trading dates. Pandas refuses to
    ``reindex`` a duplicate axis, and keeping duplicate labels would also make
    forward-return labels ambiguous, so training canonicalizes once at the
    boundary and keeps the latest row for each date.
    """
    if rows is None or rows.empty:
        return None
    missing = [c for c in _REQUIRED_OHLCV if c not in rows.columns]
    if missing:
        return None

    df = rows.loc[:, list(_REQUIRED_OHLCV)].copy()
    idx = pd.to_datetime(df.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    df.index = idx
    df = df[~df.index.isna()].sort_index()
    if df.empty:
        return None
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df if not df.empty else None


def build_training_features(
    ticker: str,
    ohlcv: dict[str, pd.DataFrame],
    indicator_spec: dict,
    lookahead: int,
    threshold: float,
    hurst_cache: "SpyHurstMemo | None" = None,
) -> pd.DataFrame | None:
    """Build a labelled feature frame for one ticker.

    close = stock/SPY*100 — relative price prevents bull-market always-buy bias.
    Regime context (spy_realized_vol, spy_adx, spy_trend, hurst_proxy) are
    rolling per-bar values — accurate for training; inference broadcasts scalars.

    Returns DataFrame with features + fwd_return + label, or None on failure.
    """
    if ticker not in ohlcv or "SPY" not in ohlcv:
        return None

    stock_raw = _canonical_ohlcv_frame(ohlcv[ticker])
    spy_raw = _canonical_ohlcv_frame(ohlcv["SPY"])
    if stock_raw is None or spy_raw is None:
        return None

    stock_ind = _compute_all(stock_raw, indicator_spec)
    spy_ind   = _compute_all(spy_raw,  indicator_spec)
    if stock_ind is None or spy_ind is None:
        return None

    common_idx = stock_ind.index.intersection(spy_ind.index)
    s = stock_ind.loc[common_idx].copy()
    p = spy_ind.loc[common_idx].copy()

    result = pd.DataFrame(index=common_idx)

    # Relative close: stock / SPY * 100
    spy_close_aligned = spy_raw["close"].reindex(common_idx)
    result["close"] = s["close"] / spy_close_aligned.replace(0, np.nan) * 100

    for col in _RATIO_FEATURES:
        if col in s.columns and col in p.columns:
            result[col] = s[col] / p[col].replace(0, np.nan)
    for col in _DIFF_FEATURES:
        if col in s.columns and col in p.columns:
            result[col] = s[col] - p[col]

    close  = s["close"]
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    result["trend"]      = close / ema50
    result["trend_long"] = close / ema200
    rel_price = close / p["close"].replace(0, np.nan)
    result["rel_mom_20d"] = rel_price.pct_change(20)
    result["rel_mom_60d"] = rel_price.pct_change(60)

    # Rolling per-bar regime context
    spy_rets = spy_raw["close"].pct_change().reindex(common_idx)
    spy_adx  = p["adx"] if "adx" in p.columns else pd.Series(25.0, index=common_idx)
    spy_ema50 = spy_raw["close"].ewm(span=50, adjust=False).mean().reindex(common_idx)
    spy_close_full = spy_raw["close"].reindex(common_idx)
    result["spy_realized_vol"] = spy_rets.rolling(20).std() * np.sqrt(252)
    result["spy_adx"]          = spy_adx
    result["spy_trend"]        = spy_close_full / spy_ema50.replace(0, np.nan)
    # Audit fix TF-3 (2026-04-25): pre-fix `hurst_proxy` was just lag-1
    # autocorrelation under a misleading name. Now compute the real
    # Hurst exponent (rescaled-range / R/S) over a 63-day window via
    # the existing kernel.regime helper. Same column name preserved
    # for backwards-compat with downstream model artifacts'
    # `feature_columns`. Cleaner regime context = better model.
    from renquant_common.hurst import rolling_hurst as _rolling_hurst  # noqa: PLC0415
    # Perf (G-J): the 63-day rolling Hurst on SPY returns is the dominant
    # per-ticker cost and — for a given `spy_rets` — is identical across
    # tickers. When a run-scoped memoizer is supplied it collapses those
    # repeated computations to one per DISTINCT `spy_rets`. Output-invariant:
    # the cached Series is exactly what `_rolling_hurst(spy_rets, window=63)`
    # returns, and `.reindex(common_idx)` still runs unchanged per ticker.
    if hurst_cache is not None:
        spy_hurst = hurst_cache.get_or_compute(
            spy_rets, lambda: _rolling_hurst(spy_rets, window=63)
        )
    else:
        spy_hurst = _rolling_hurst(spy_rets, window=63)
    result["hurst_proxy"] = spy_hurst.reindex(common_idx)

    # Supervised labels: stock outperformance vs SPY over lookahead days
    stock_fwd = stock_raw["close"].pct_change(lookahead).shift(-lookahead)
    spy_fwd   = spy_raw["close"].pct_change(lookahead).shift(-lookahead)
    rel_fwd   = stock_fwd - spy_fwd.reindex(stock_fwd.index)
    result["fwd_return"] = rel_fwd.reindex(common_idx)
    result["label"] = np.where(result["fwd_return"] >  threshold,  1,
                      np.where(result["fwd_return"] < -threshold, -1, 0))

    # Audit TF-6 reconsidered (2026-04-25): the per-ticker tournament
    # consumers (Classification / Q-learning / XGBoost) require every
    # row to have non-NaN features — they have no native imputation
    # path. The panel pipeline handles its own warm-up via
    # build_panel_frame's `iloc[min_history_days:]` slice BEFORE
    # concatenating, so the panel path doesn't pay the dropna cost
    # twice. Keeping `dropna()` is correct for the tournament path
    # and harmless for the panel path. Documented here so future
    # readers don't try to "fix" this again.
    result = result.dropna()
    return result if not result.empty else None


def build_all_training_features(
    watchlist: list[str],
    ohlcv: dict[str, pd.DataFrame],
    indicator_spec: dict,
    lookahead: int,
    threshold: float,
) -> dict[str, pd.DataFrame]:
    """Build labelled feature frames for all watchlist tickers in parallel."""

    canonical_ohlcv = {
        symbol: frame
        for symbol, frame in (
            (symbol, _canonical_ohlcv_frame(rows))
            for symbol, rows in ohlcv.items()
        )
        if frame is not None
    }

    per_ticker_ohlcv: dict[str, dict[str, pd.DataFrame]] = {}
    for ticker in watchlist:
        local: dict[str, pd.DataFrame] = {}
        if "SPY" in canonical_ohlcv:
            local["SPY"] = canonical_ohlcv["SPY"].copy(deep=True)
        if ticker in canonical_ohlcv and ticker != "SPY":
            local[ticker] = canonical_ohlcv[ticker].copy(deep=True)
        elif ticker == "SPY" and "SPY" in canonical_ohlcv:
            local[ticker] = canonical_ohlcv["SPY"].copy(deep=True)
        per_ticker_ohlcv[ticker] = local

    def _build(ticker: str):
        df = build_training_features(
            ticker, per_ticker_ohlcv.get(ticker, {}),
            indicator_spec, lookahead, threshold,
        )
        return ticker, df

    feature_frames: dict[str, pd.DataFrame] = {}
    # ThreadPoolExecutor: numpy/pandas release the GIL during computation,
    # so multiple tickers genuinely run in parallel.
    with ThreadPoolExecutor() as pool:
        futures = {pool.submit(_build, t): t for t in watchlist}
        # Collect results in original watchlist order for clean output
        results = {}
        for future in as_completed(futures):
            ticker, df = future.result()
            results[ticker] = df

    for ticker in watchlist:
        df = results.get(ticker)
        if df is None or df.empty:
            print(f"  {ticker}: no feature frame")
            continue
        lv = df["label"].value_counts().to_dict()
        print(f"  {ticker}: {len(df)} rows  "
              f"buy={lv.get(1,0)}  sell={lv.get(-1,0)}  hold={lv.get(0,0)}")
        feature_frames[ticker] = df

    print(f"\nBuilt frames for {len(feature_frames)} / {len(watchlist)} symbols.")
    return feature_frames
