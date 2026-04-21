"""Per-symbol model tournament: trains 4 approaches, picks best by OOS Sharpe.

Fixed train/OOS split at 2024-01-01. Tries Classification, QLearning, Manual,
and XGBoost; selects the winner by annualised Sharpe on the OOS period.

Exports:
  oos_sharpe(prices, signals) -> float
  run_tournament(ticker, df, prices, spy_prices, model_params, sharpe_floor, tax_config) -> dict
  run_tournament_all(watchlist, feature_frames, ohlcv, config, max_workers=None) -> dict[str, dict]
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .models import create_model, XGBoostModel
from .scoring import (
    fit_expected_return_calibration,
    fit_probability_calibration,
    raw_score_kind_for_model,
)

_TRAIN_CUTOFF = pd.Timestamp("2024-01-01")

# Strategy root dir — used to set PYTHONPATH in worker processes so they can
# import training.* even with spawn (which doesn't inherit sys.path).
_STRATEGY_DIR = str(Path(__file__).parent.parent)


def oos_sharpe(prices: pd.Series, signals: pd.Series) -> float:
    """Annualised Sharpe for a long-only OOS signal series."""
    try:
        prices = prices.dropna()
        if len(prices) < 20:
            return 0.0
        sigs      = signals.reindex(prices.index).ffill().fillna(0).clip(0, 1)
        daily_ret = prices.pct_change().fillna(0)
        strat_ret = daily_ret * sigs.shift(1).fillna(0)
        std = strat_ret.std()
        return 0.0 if std == 0 else float(strat_ret.mean() / std * np.sqrt(252))
    except Exception as e:
        return 0.0


def run_tournament(
    ticker: str,
    df: pd.DataFrame,
    prices: pd.Series,
    spy_prices: pd.Series,
    model_params: dict,
    sharpe_floor: float,
    tax_config: dict,
    nthread: int | None = None,
) -> dict:
    """Train all 4 approaches on pre-2024 data; evaluate OOS on 2024+; return best.

    df must contain feature columns + 'label'.
    prices / spy_prices should cover the full df period; OOS slice is derived internally.
    nthread: passed to XGBoostModel to limit CPU usage when running in parallel workers.
    """
    _log: list[str] = []

    feature_cols = model_params["feature_columns"]
    lookahead    = model_params["lookahead"]
    threshold    = model_params["threshold"]
    bags         = model_params["bags"]
    leaf_size    = model_params["leaf_size"]
    buy_thr      = model_params["buy_threshold"]
    sell_thr     = model_params["sell_threshold"]

    train_df = df[df.index < _TRAIN_CUTOFF]
    oos_df   = df[df.index >= _TRAIN_CUTOFF]

    _empty = {
        "sharpe": -99.0, "passes_floor": False, "best_approach": None, "model": None,
        "oos_signals": None, "oos_raw_scores": None, "score_calibration": None,
        "oos_prices": prices.reindex(oos_df.index) if not oos_df.empty else prices,
        "train_rows": len(train_df), "oos_rows": len(oos_df),
        "_log": _log,
    }
    if len(train_df) < 60 or len(oos_df) < 30:
        _log.append(f"{ticker}: insufficient data (train={len(train_df)}, oos={len(oos_df)}), skipping")
        return _empty

    prices_oos     = prices.reindex(oos_df.index)
    spy_prices_oos = spy_prices.reindex(oos_df.index).replace(0, np.nan)

    _seed       = abs(hash(ticker)) % (2 ** 32)
    best_sharpe = -99.0
    best_model  = best_name = best_sigs = best_scores = None

    # ── Approach 1: Classification ──────────────────────────────────────────────
    try:
        np.random.seed(_seed)
        clf = create_model("classification", feature_columns=feature_cols,
                           lookahead=lookahead, threshold=threshold,
                           leaf_size=leaf_size, bags=bags,
                           buy_threshold=buy_thr, sell_threshold=sell_thr)
        clf.train(train_df)
        sigs = clf.predict_bulk(oos_df).map({"buy": 1, "hold": 0, "sell": -1}).reindex(oos_df.index)
        sh   = oos_sharpe(prices_oos, sigs)
        _log.append(f"  {ticker} Classification OOS Sharpe: {sh:.3f}")
        if sh > best_sharpe:
            best_sharpe, best_model, best_name, best_sigs = sh, clf, "Classification", sigs
            best_scores = clf.predict_score_bulk(oos_df)
    except Exception as e:
        _log.append(f"  {ticker} Classification error: {e}")

    # ── Approach 2: Q-Learning ──────────────────────────────────────────────────
    try:
        import random as _random
        _random.seed(_seed); np.random.seed(_seed)
        ql = create_model("qlearning", feature_columns=feature_cols[:5])
        ql.train(train_df)
        sigs = ql.predict_bulk(oos_df).map({"buy": 1, "hold": 0, "sell": -1}).reindex(oos_df.index)
        sh   = oos_sharpe(prices_oos, sigs)
        _log.append(f"  {ticker} QLearning    OOS Sharpe: {sh:.3f}")
        if sh > best_sharpe:
            best_sharpe, best_model, best_name, best_sigs = sh, ql, "QLearning", sigs
            best_scores = ql.predict_score_bulk(oos_df)
    except Exception as e:
        _log.append(f"  {ticker} QLearning error: {e}")

    # ── Approach 3: Manual ──────────────────────────────────────────────────────
    try:
        manual = create_model("manual", buy_threshold=2, sell_threshold=-2)
        manual.train(train_df)
        sigs = manual.predict_bulk(oos_df).map({"buy": 1, "hold": 0, "sell": -1}).reindex(oos_df.index)
        sh   = oos_sharpe(prices_oos, sigs)
        _log.append(f"  {ticker} Manual       OOS Sharpe: {sh:.3f}")
        if sh > best_sharpe:
            best_sharpe, best_model, best_name, best_sigs = sh, manual, "Manual", sigs
            best_scores = manual.predict_score_bulk(oos_df)
    except Exception as e:
        _log.append(f"  {ticker} Manual error: {e}")

    # ── Approach 4: XGBoost ─────────────────────────────────────────────────────
    try:
        np.random.seed(_seed)
        xgb = XGBoostModel(feature_columns=feature_cols,
                           lookahead=lookahead, threshold=threshold,
                           buy_threshold=0.1, sell_threshold=0.1,
                           n_estimators=200, max_depth=4,
                           learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, min_child_weight=10,
                           nthread=nthread)
        xgb.train(train_df)
        sigs = xgb.predict_bulk(oos_df).map({"buy": 1, "hold": 0, "sell": -1}).reindex(oos_df.index)
        sh   = oos_sharpe(prices_oos, sigs)
        _log.append(f"  {ticker} XGBoost      OOS Sharpe: {sh:.3f}")
        if sh > best_sharpe:
            best_sharpe, best_model, best_name, best_sigs = sh, xgb, "XGBoost", sigs
            best_scores = xgb.predict_score(oos_df)
    except Exception as e:
        _log.append(f"  {ticker} XGBoost error: {e}")

    if best_scores is not None:
        best_scores = pd.Series(best_scores, index=oos_df.index, dtype=float)

    best_calibration = None
    if best_model is not None and best_scores is not None:
        rel_prices = (prices_oos / spy_prices_oos).replace([np.inf, -np.inf], np.nan)
        future_rel = rel_prices.shift(-lookahead) / rel_prices - 1.0
        best_calibration = fit_probability_calibration(
            best_scores, future_rel,
            lookahead=lookahead, threshold=threshold,
            score_kind=raw_score_kind_for_model(best_model),
        )
        er_fields = fit_expected_return_calibration(
            best_scores, future_rel, lookahead=lookahead,
        )
        for k, v in er_fields.items():
            setattr(best_calibration, k, v)

    passes = best_sharpe >= sharpe_floor
    _log.append(f"  → WINNER: {best_name}  Sharpe={best_sharpe:.3f}  "
                f"{'✓ PASS' if passes else '✗ FAIL (no model exported)'}\n")

    return {
        "sharpe":            best_sharpe,
        "best_approach":     best_name,
        "model":             best_model,
        "oos_signals":       best_sigs,
        "oos_raw_scores":    best_scores,
        "score_calibration": best_calibration,
        "oos_prices":        prices_oos,
        "train_rows":        len(train_df),
        "oos_rows":          len(oos_df),
        "passes_floor":      passes,
        "_log":              _log,
    }


def run_tournament_all(
    watchlist: list[str],
    feature_frames: dict[str, pd.DataFrame],
    ohlcv: dict[str, pd.DataFrame],
    config: dict,
    max_workers: int | None = None,
) -> dict[str, dict]:
    """Run tournament for every ticker in watchlist in parallel; return results dict.

    max_workers: number of parallel processes (default: cpu_count, capped at watchlist size).
    Each worker uses nthread=1 for XGBoost to avoid CPU oversubscription.
    """
    model_params = config["model_params"]
    sharpe_floor = float(config.get("sharpe_floor", 0.8))
    tax_config   = config["tax"]

    tickers = [t for t in watchlist if t in feature_frames]
    if not tickers:
        return {}

    n_workers = min(len(tickers), max_workers or os.cpu_count() or 4)
    # Each worker gets 1 XGBoost thread; remaining cores fill in from the OS scheduler
    xgb_nthread = max(1, (os.cpu_count() or 4) // n_workers)

    print(f"Tournament: {len(tickers)} tickers, {n_workers} workers, "
          f"XGBoost nthread={xgb_nthread} per worker")

    # Ensure worker processes (spawn) can import training.*
    existing = os.environ.get("PYTHONPATH", "")
    if _STRATEGY_DIR not in existing:
        os.environ["PYTHONPATH"] = _STRATEGY_DIR + (":" + existing if existing else "")

    spy_close = ohlcv["SPY"]["close"]
    results: dict[str, dict] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                run_tournament,
                ticker,
                feature_frames[ticker],
                ohlcv[ticker]["close"],
                spy_close,
                model_params,
                sharpe_floor,
                tax_config,
                xgb_nthread,
            ): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
                for line in result.pop("_log", []):
                    print(line)
                results[ticker] = result
            except Exception as e:
                print(f"{ticker}: tournament failed — {e}")
                results[ticker] = {
                    "sharpe": -99.0, "passes_floor": False, "best_approach": None,
                    "model": None, "oos_signals": None, "oos_raw_scores": None,
                    "score_calibration": None, "train_rows": 0, "oos_rows": 0,
                }

    return results
