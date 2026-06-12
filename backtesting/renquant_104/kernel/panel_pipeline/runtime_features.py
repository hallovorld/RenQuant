"""Runtime feature-assembly helpers (caches + fund/PEAD/SUE/sentiment).

EXTRACTED 2026-06-12 from job_panel_scoring.py (eng plan S2 item 5,
decomposition slice 3; behavior-identical move, DRPH-gated with
pre-change baselines). Symbols re-exported from job_panel_scoring.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kernel.panel_pipeline.ngboost_tasks import _sentiment_cfg  # no cycle: ngboost_tasks imports calibration only

log = logging.getLogger("panel_pipeline.runtime_features")


def _runtime_cache(ctx: Any) -> dict | None:
    cache = getattr(ctx, "_panel_runtime_cache", None)
    return cache if isinstance(cache, dict) else None


def _cached_parquet(ctx: Any, key: tuple, path: Path) -> pd.DataFrame | None:
    cache = _runtime_cache(ctx)
    if cache is None:
        return pd.read_parquet(path)
    if key not in cache:
        cache[key] = pd.read_parquet(path)
    return cache[key]


def _cached_earnings_surprise(ctx: Any, path: Path) -> pd.DataFrame | None:
    cache_key = ("earnings_surprise", str(path))
    cache = _runtime_cache(ctx)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    earn = pd.read_parquet(path).reset_index()
    earn = earn.rename(columns={earn.columns[0]: "earnings_date"})
    earn["earnings_date"] = pd.to_datetime(earn["earnings_date"])
    earn = earn.sort_values("earnings_date").reset_index(drop=True)
    if cache is not None:
        cache[cache_key] = earn
    return earn


def _cached_sentiment(ctx: Any, path: Path) -> pd.DataFrame | None:
    cache_key = ("news_sentiment", str(path))
    cache = _runtime_cache(ctx)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    sdf = pd.read_parquet(path)
    sdf["date"] = pd.to_datetime(sdf["date"])
    if cache is not None:
        cache[cache_key] = sdf
    return sdf


def _alpha158_cached_rows(
    ctx: Any,
    tickers: list[str],
    today: Any,
) -> dict[str, dict[str, float]]:
    cache = getattr(ctx, "_alpha158_feature_cache", None)
    if not isinstance(cache, dict) or not cache:
        return {}
    today_ts = pd.Timestamp(today)
    rows: dict[str, dict[str, float]] = {}
    for ticker in tickers:
        frame = cache.get(ticker)
        if frame is None or frame.empty:
            continue
        sub = frame.loc[:today_ts]
        if sub.empty:
            continue
        rows[ticker] = sub.iloc[-1].to_dict()
    return rows


def _stable_feature_context_tickers(
    ctx: Any,
    target_tickers: list[str],
    scorer: Any | None = None,
) -> list[str]:
    """Return the stable cross-section used for extra-feature fill/rank.

    Training fills fundamentals/sentiment and ranks PEAD over the full date
    cross-section. Runtime must therefore not compute medians/ranks over the
    post-filter candidate subset; that makes a ticker's feature value depend on
    which other tickers survived gates on the same bar.
    """
    out: list[str] = []
    seen: set[str] = set()

    def add_many(values: Any) -> None:
        if isinstance(values, dict):
            values = values.keys()
        if not isinstance(values, (list, tuple, set)):
            return
        for value in values:
            if value is None:
                continue
            ticker = str(value)
            if ticker and ticker not in seen:
                seen.add(ticker)
                out.append(ticker)

    panel_cfg = (getattr(ctx, "config", {}) or {}).get("ranking", {}) \
        .get("panel_scoring", {})
    for key in (
        "feature_context_tickers",
        "training_universe",
        "train_tickers",
        "tickers",
    ):
        add_many(panel_cfg.get(key))
    metadata = getattr(scorer, "metadata", {}) or {}
    for key in (
        "feature_context_tickers",
        "training_universe",
        "train_tickers",
        "tickers",
        "watchlist",
    ):
        add_many(metadata.get(key))
    add_many((getattr(ctx, "config", {}) or {}).get("watchlist", []))
    add_many(getattr(ctx, "models", {}) or {})
    add_many(getattr(ctx, "holdings", {}) or {})
    add_many(target_tickers)
    return out


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _median_fill_rows(
    raw_by_ticker: dict[str, dict[str, float | None]],
    target_tickers: list[str],
    context_tickers: list[str],
    cols: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    medians: dict[str, float] = {}
    filled: dict[str, dict[str, float]] = {}
    for col in cols:
        vals = [
            float(raw_by_ticker.get(t, {}).get(col))
            for t in context_tickers
            if _finite_or_none(raw_by_ticker.get(t, {}).get(col)) is not None
        ]
        medians[col] = float(np.median(vals)) if vals else 0.0
    for ticker in target_tickers:
        row: dict[str, float] = {}
        raw = raw_by_ticker.get(ticker, {})
        for col in cols:
            value = _finite_or_none(raw.get(col))
            row[col] = value if value is not None else medians[col]
        filled[ticker] = row
    return filled, medians


def _apply_fund_features(
    rows: dict[str, dict[str, float]],
    fund_panel: pd.DataFrame,
    today: Any,
    context_tickers: list[str],
    fund_cols: list[str],
) -> tuple[int, int, dict[str, float]]:
    today_ts = pd.Timestamp(today)
    panel = fund_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    snap = panel[panel["date"] <= today_ts] \
        .sort_values("date").groupby("ticker").tail(1)
    by_ticker = {
        str(t): g.iloc[-1]
        for t, g in snap.groupby("ticker", sort=False)
    }
    raw: dict[str, dict[str, float | None]] = {}
    for ticker in context_tickers:
        src = by_ticker.get(str(ticker))
        raw[ticker] = {
            col: (_finite_or_none(src[col])
                  if src is not None and col in src.index else None)
            for col in fund_cols
        }
    target_tickers = list(rows.keys())
    filled, medians = _median_fill_rows(raw, target_tickers, context_tickers, fund_cols)
    n_real = 0
    n_imputed = 0
    for ticker in target_tickers:
        for col in fund_cols:
            if _finite_or_none(raw.get(ticker, {}).get(col)) is None:
                n_imputed += 1
            else:
                n_real += 1
            rows[ticker][col] = filled[ticker][col]
    return n_real, n_imputed, medians


def _earnings_raw_row(
    ctx: Any,
    earn_dir: Path,
    ticker: str,
    today_ts: pd.Timestamp,
) -> tuple[dict[str, float | None], bool, bool, bool]:
    ep = earn_dir / f"{ticker}.parquet"
    if not ep.exists():
        return {}, True, False, False
    earn = _cached_earnings_surprise(ctx, ep)
    if earn is None:
        return {}, True, False, False
    prior = earn[earn["earnings_date"] <= today_ts]
    if len(prior) == 0:
        return {}, False, True, False
    last = prior.iloc[-1]
    days_since = int((today_ts - last["earnings_date"]).days)
    if days_since > 60 or days_since < 0:
        return {}, False, False, True
    decay = max(0.0, 1.0 - days_since / 60)
    surprise = _finite_or_none(last.get("surprise_pct")) or 0.0
    return {
        "days_since_earnings": float(days_since),
        "pead_signal": surprise * decay,
        "pead_surprise": surprise,
    }, False, False, False


def _apply_pead_features(
    ctx: Any,
    rows: dict[str, dict[str, float]],
    earn_dir: Path,
    today_ts: pd.Timestamp,
    context_tickers: list[str],
    pead_cols: list[str],
) -> tuple[int, int, int, int]:
    raw: dict[str, dict[str, float | None]] = {}
    n_no_data = n_no_prior = n_out_of_window = 0
    for ticker in context_tickers:
        row, no_data, no_prior, oow = _earnings_raw_row(ctx, earn_dir, ticker, today_ts)
        n_no_data += int(no_data)
        n_no_prior += int(no_prior)
        n_out_of_window += int(oow)
        raw[ticker] = {
            "days_since_earnings": row.get("days_since_earnings"),
            "pead_signal": row.get("pead_signal"),
            "pead_quintile_rank": None,
        }
        if row.get("pead_surprise") is not None:
            raw[ticker]["pead_surprise"] = row["pead_surprise"]
    surprises = {
        ticker: raw[ticker]["pead_surprise"]
        for ticker in context_tickers
        if _finite_or_none(raw.get(ticker, {}).get("pead_surprise")) is not None
    }
    if surprises:
        ranks = pd.Series(surprises, dtype=float).rank(pct=True)
        for ticker, rank in ranks.items():
            raw[ticker]["pead_quintile_rank"] = float(rank)
    filled, _medians = _median_fill_rows(
        raw, list(rows.keys()), context_tickers, pead_cols,
    )
    for ticker, vals in filled.items():
        for col in pead_cols:
            rows[ticker][col] = vals[col]
    return len(surprises), n_no_data, n_no_prior, n_out_of_window


def _apply_sue_features(
    ctx: Any,
    rows: dict[str, dict[str, float]],
    earn_dir: Path,
    today_ts: pd.Timestamp,
    context_tickers: list[str],
    sue_cols: list[str],
) -> tuple[int, int, int]:
    raw: dict[str, dict[str, float | None]] = {}
    n_active = n_no_data = n_oow = 0
    for ticker in context_tickers:
        ep = earn_dir / f"{ticker}.parquet"
        if not ep.exists():
            n_no_data += 1
            raw[ticker] = {col: None for col in sue_cols}
            continue
        earn = _cached_earnings_surprise(ctx, ep)
        if earn is None:
            n_no_data += 1
            raw[ticker] = {col: None for col in sue_cols}
            continue
        prior = earn[earn["earnings_date"] <= today_ts]
        if len(prior) == 0:
            raw[ticker] = {col: 0.0 for col in sue_cols}
            continue
        last = prior.iloc[-1]
        days_since = int((today_ts - last["earnings_date"]).days)
        if days_since > 60 or days_since < 0:
            n_oow += 1
            raw[ticker] = {col: 0.0 for col in sue_cols}
            continue
        decay = max(0.0, 1.0 - days_since / 60)
        s = prior["surprise_pct"].astype(float)
        if len(s) >= 2:
            denom_window = s.iloc[max(0, len(s) - 1 - 4):len(s) - 1]
            denom = float(denom_window.std()) if len(denom_window) >= 2 else 0.0
            sue = float(s.iloc[-1]) / max(denom, 1e-6)
            sue = max(min(sue, 5.0), -5.0)
        else:
            sue = 0.0
        mom = float(s.iloc[-1] - s.iloc[-2]) if len(s) >= 2 else 0.0
        streak = 0
        cur_sign = 0
        for v in s:
            sign = 1 if v > 0 else (-1 if v < 0 else 0)
            if sign == 0 or sign != cur_sign:
                streak = sign
                cur_sign = sign
            else:
                streak += sign
        raw[ticker] = {
            "sue_signal": sue * decay,
            "surprise_momentum": mom * decay,
            "surprise_streak": float(streak) * decay,
        }
        n_active += 1
    filled, _medians = _median_fill_rows(raw, list(rows.keys()), context_tickers, sue_cols)
    for ticker, vals in filled.items():
        for col in sue_cols:
            rows[ticker][col] = vals[col]
    return n_active, n_no_data, n_oow


def _sentiment_runtime_gate_declared(scorer: Any) -> bool:
    metadata = getattr(scorer, "metadata", {}) or {}
    contract = (
        metadata.get("sentiment_runtime_gate_contract")
        or metadata.get("sentiment_gate_contract")
    )
    return contract in {"trained_zeroing", "runtime_zeroing"} or bool(
        metadata.get("sentiment_runtime_gate_trained", False)
    )


def _apply_sentiment_features(
    ctx: Any,
    scorer: Any,
    rows: dict[str, dict[str, float]],
    sent_dir: Path,
    today_ts: pd.Timestamp,
    context_tickers: list[str],
    sent_cols: list[str],
) -> tuple[int, int, bool]:
    raw: dict[str, dict[str, float | None]] = {}
    n_hit = n_miss = 0
    for ticker in context_tickers:
        sp = sent_dir / f"{ticker}.parquet"
        raw[ticker] = {col: None for col in sent_cols}
        if not sp.exists():
            n_miss += 1
            continue
        try:
            sdf = _cached_sentiment(ctx, sp)
        except Exception:
            n_miss += 1
            continue
        exact = sdf[pd.to_datetime(sdf["date"]) == today_ts]
        if len(exact) == 0:
            n_miss += 1
            continue
        last = exact.iloc[-1]
        if "sentiment_pos_share" in sent_cols:
            raw[ticker]["sentiment_pos_share"] = _finite_or_none(
                last.get("sentiment_pos_share")
            )
        if "mean_sentiment" in sent_cols:
            raw[ticker]["mean_sentiment"] = _finite_or_none(last.get("mean_sentiment"))
        if "n_articles_log" in sent_cols:
            if "n_articles_log" in last.index:
                raw[ticker]["n_articles_log"] = _finite_or_none(last.get("n_articles_log"))
            else:
                n_articles = _finite_or_none(last.get("n_articles")) or 0.0
                raw[ticker]["n_articles_log"] = float(np.log1p(n_articles))
        n_hit += 1
    filled, _medians = _median_fill_rows(
        raw, list(rows.keys()), context_tickers, sent_cols,
    )
    for ticker, vals in filled.items():
        for col in sent_cols:
            rows[ticker][col] = vals[col]
    sent_enabled = bool(_sentiment_cfg(ctx).get("enabled", True))
    gate_applied = False
    if not sent_enabled:
        if _sentiment_runtime_gate_declared(scorer):
            for ticker in rows:
                for col in sent_cols:
                    rows[ticker][col] = 0.0
            gate_applied = True
        else:
            log.warning(
                "ApplyScoresTask[panel_ltr_xgboost]: sentiment gate OFF for "
                "regime=%s, but artifact lacks trained runtime-zeroing "
                "contract; leaving exact-date sentiment features unchanged.",
                getattr(ctx, "regime", "?"),
            )
    return n_hit, n_miss, gate_applied


