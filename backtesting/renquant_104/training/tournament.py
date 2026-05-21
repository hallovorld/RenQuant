"""Per-symbol model tournament: trains 4 approaches, picks best by OOS Sharpe.

Rolling train/OOS split — default: today - 2 years. Tries Classification,
QLearning, Manual, and XGBoost; selects the winner by annualised Sharpe on
the OOS period. Override with config["oos_cutoff"] (ISO date) or
config["oos_years"] (int).

Exports:
  oos_sharpe(prices, signals) -> float
  resolve_oos_cutoff(config) -> pd.Timestamp
  run_tournament(ticker, df, prices, spy_prices, model_params, sharpe_floor, tax_config, *, oos_cutoff=None) -> dict
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

_DEFAULT_OOS_YEARS = 2     # legacy default — kept only for backward compat
_DEFAULT_OOS_DAYS  = 90    # 2026-05-04 user mandate: ~63 trading days = 3-month
                            # rolling Sharpe window. Recent performance dominates.


def resolve_oos_cutoff(config: dict) -> pd.Timestamp:
    """OOS cutoff: explicit `oos_cutoff` wins; else `today - oos_days` (default 90).

    2026-05-04 user mandate: per-ticker tournament Sharpe must reflect
    RECENT performance, not a 2-year aggregate. Default OOS window
    tightened from 2 years → ~3 trading-month (90 calendar days). Old
    `oos_years` config key is honored when set explicitly so legacy
    pipelines don't break, but the new default is days-based.

    Anchor:
      * If `config["sample_end"]` is set (B2 hold-out path): cutoff =
        min(sample_end, today) - oos_days. Train data ends at cutoff; OOS
        slice is cutoff → available data. Future-dated configs must not
        shift the OOS window forward before those bars exist.
      * Else (live / generic train): cutoff = today - oos_days.
    """
    raw = config.get("oos_cutoff") if config else None
    if raw:
        return pd.Timestamp(raw)

    sample_end = config.get("sample_end") if config else None
    today = pd.Timestamp.today().normalize()
    anchor = pd.Timestamp(sample_end).normalize() if sample_end else today
    if anchor > today:
        anchor = today

    # Prefer days-based config; fall back to legacy years key only when
    # operator explicitly opted in via `oos_years` (no implicit promotion).
    oos_days  = config.get("oos_days") if config else None
    oos_years = config.get("oos_years") if config else None
    if oos_days is not None:
        return anchor - pd.Timedelta(days=int(oos_days))
    if oos_years is not None:
        return anchor - pd.DateOffset(years=int(oos_years))
    return anchor - pd.Timedelta(days=_DEFAULT_OOS_DAYS)

# Strategy root dir — used to set PYTHONPATH in worker processes so they can
# import training.* even with spawn (which doesn't inherit sys.path).
_STRATEGY_DIR = str(Path(__file__).parent.parent)


def oos_sharpe(prices: pd.Series, signals: pd.Series) -> float:
    """Annualised Sharpe for a long-only OOS signal series.

    Audit fix TOURN-1 (Round 2 deep audit, 2026-04-25): pre-fix, NaN
    std (empty strat_ret after constant returns + NaN propagation, or
    a degenerate single-row case) hit `std == 0` which is False on
    NaN → fell through to division → returned NaN Sharpe. The
    tournament then ranked NaN as "neither best nor worst" depending
    on Python's sort behaviour with NaN, occasionally elevating an
    all-NaN model. Now: explicit isfinite guard on std AND mean —
    return 0.0 on any non-finite intermediate (signal "no edge").
    """
    try:
        prices = prices.dropna()
        if len(prices) < 20:
            return 0.0
        sigs      = signals.reindex(prices.index).ffill().fillna(0).clip(0, 1)
        daily_ret = prices.pct_change().fillna(0)
        strat_ret = daily_ret * sigs.shift(1).fillna(0)
        std  = strat_ret.std()
        mean = strat_ret.mean()
        if std == 0 or not np.isfinite(std) or not np.isfinite(mean):
            return 0.0
        return float(mean / std * np.sqrt(252))
    except Exception as e:
        return 0.0


def oos_single_ticker_ic(
    raw_scores: pd.Series,
    prices: pd.Series,
    spy_prices: pd.Series,
    lookahead: int = 5,
) -> float:
    """Spearman correlation between raw_score and future relative-to-SPY return.

    Single-ticker variant of cross-sectional IC: for each bar, compare
    today's raw_score with the next `lookahead` days' return net of SPY.
    Higher = the model's score orders same-ticker bars well.

    Used as an alternative tournament winner-selection metric (`winner_metric=ic`).
    """
    try:
        from scipy.stats import spearmanr
        if raw_scores is None or raw_scores.empty or prices is None or prices.empty:
            return 0.0
        idx = raw_scores.index.intersection(prices.index).intersection(spy_prices.index)
        if len(idx) < 20:
            return 0.0
        p    = prices.reindex(idx).astype(float).replace(0, np.nan)
        spy  = spy_prices.reindex(idx).astype(float).replace(0, np.nan)
        rel  = (p / spy).replace([np.inf, -np.inf], np.nan)
        fwd  = (rel.shift(-lookahead) / rel - 1.0).dropna()
        raw  = raw_scores.reindex(fwd.index).astype(float).dropna()
        if len(raw) < 20:
            return 0.0
        common = raw.index.intersection(fwd.index)
        if len(common) < 20:
            return 0.0
        rho, _ = spearmanr(raw.loc[common].values, fwd.loc[common].values)
        return 0.0 if (rho is None or rho != rho) else float(rho)
    except Exception:
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
    oos_cutoff: "pd.Timestamp | str | None" = None,
    exclude_models: "set[str] | None" = None,
    winner_metric: str = "sharpe",
) -> dict:
    """Train on data before oos_cutoff; evaluate OOS on data on/after; return best.

    df must contain feature columns + 'label'.
    prices / spy_prices should cover the full df period; OOS slice is derived internally.
    nthread: passed to XGBoostModel to limit CPU usage when running in parallel workers.
    oos_cutoff: explicit cutoff; defaults to today - 2 years.
    exclude_models: approaches to skip. Accepted values (case-insensitive):
        "classification", "qlearning", "manual", "xgboost". Defaults to none.
    winner_metric: "sharpe" (default, legacy) | "ic" (per-ticker Spearman of
        raw_score vs future relative-to-SPY return, lookahead=5). `passes_floor`
        always uses Sharpe so existing floors in strategy_config.json still
        apply regardless of the selection metric.
    """
    exclude = {str(m).strip().lower() for m in (exclude_models or set())}
    winner_metric = (winner_metric or "sharpe").strip().lower()
    _log: list[str] = []

    feature_cols = model_params["feature_columns"]
    lookahead    = model_params["lookahead"]
    threshold    = model_params["threshold"]
    bags         = model_params["bags"]
    leaf_size    = model_params["leaf_size"]
    buy_thr      = model_params["buy_threshold"]
    sell_thr     = model_params["sell_threshold"]

    if oos_cutoff is None:
        oos_cutoff = resolve_oos_cutoff({})
    else:
        oos_cutoff = pd.Timestamp(oos_cutoff)

    # Audit fix TOURN-OOS-LEAK (Round 2 deep audit, 2026-04-25): pre-fix,
    # `train_df = df[df.index < oos_cutoff]` included training rows at
    # `oos_cutoff - 1` whose forward-return labels span [oos_cutoff-1,
    # oos_cutoff-1+L]. With L=5, those labels read OOS prices for
    # 4 of the 5 days — direct lookahead leak. Same logic as the CV-1
    # purge fix, just applied at the train/OOS boundary instead of
    # the inter-fold boundary. Now: training rows must satisfy
    # `t + L < oos_cutoff` ⇔ `t < oos_cutoff - L`.
    train_cutoff = oos_cutoff - pd.Timedelta(days=int(lookahead))
    train_df = df[df.index < train_cutoff]
    oos_df   = df[df.index >= oos_cutoff]

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

    # Round-3 audit (#R3-23): Python's built-in hash() is salted by
    # PYTHONHASHSEED (random by default), so `hash("AAPL")` differs
    # between processes. The tournament was supposed to be reproducible
    # but the per-ticker seed kept flipping. Use a stable hash instead.
    import hashlib as _hashlib  # noqa: PLC0415
    _seed        = int(_hashlib.md5(ticker.encode()).hexdigest()[:8], 16)
    best_score   = -99.0   # comparison key — either sharpe or IC based on winner_metric
    best_sharpe  = -99.0   # always tracked for passes_floor check
    best_model   = best_name = best_sigs = best_scores = None

    def _evaluate(model, name: str, sigs, raw_scores) -> float:
        """Compute the metric used for winner selection. Always logs both."""
        sh = oos_sharpe(prices_oos, sigs)
        ic = 0.0
        if winner_metric == "ic" and raw_scores is not None:
            ic = oos_single_ticker_ic(
                pd.Series(raw_scores, index=oos_df.index, dtype=float).dropna(),
                prices_oos, spy_prices_oos, lookahead=lookahead,
            )
            _log.append(f"  {ticker} {name:<14s} OOS Sharpe: {sh:+.3f}  IC: {ic:+.4f}")
        else:
            _log.append(f"  {ticker} {name:<14s} OOS Sharpe: {sh:+.3f}")
        return ic if winner_metric == "ic" else sh, sh

    # ── Approach 1: Classification ──────────────────────────────────────────────
    if "classification" not in exclude:
        try:
            np.random.seed(_seed)
            clf = create_model("classification", feature_columns=feature_cols,
                               lookahead=lookahead, threshold=threshold,
                               leaf_size=leaf_size, bags=bags,
                               buy_threshold=buy_thr, sell_threshold=sell_thr)
            clf.train(train_df)
            sigs        = clf.predict_bulk(oos_df).map({"buy": 1, "hold": 0, "sell": -1}).reindex(oos_df.index)
            raw_scores  = clf.predict_score_bulk(oos_df)
            metric, sh  = _evaluate(clf, "Classification", sigs, raw_scores)
            if metric > best_score:
                best_score, best_sharpe = metric, sh
                best_model, best_name, best_sigs, best_scores = clf, "Classification", sigs, raw_scores
        except Exception as e:
            _log.append(f"  {ticker} Classification error: {e}")

    # ── Approach 2: Q-Learning ──────────────────────────────────────────────────
    if "qlearning" not in exclude:
        try:
            import random as _random
            _random.seed(_seed); np.random.seed(_seed)
            ql = create_model("qlearning", feature_columns=feature_cols[:5])
            ql.train(train_df)
            sigs        = ql.predict_bulk(oos_df).map({"buy": 1, "hold": 0, "sell": -1}).reindex(oos_df.index)
            raw_scores  = ql.predict_score_bulk(oos_df)
            metric, sh  = _evaluate(ql, "QLearning", sigs, raw_scores)
            if metric > best_score:
                best_score, best_sharpe = metric, sh
                best_model, best_name, best_sigs, best_scores = ql, "QLearning", sigs, raw_scores
        except Exception as e:
            _log.append(f"  {ticker} QLearning error: {e}")

    # ── Approach 3: Manual ──────────────────────────────────────────────────────
    if "manual" not in exclude:
        try:
            manual = create_model("manual", buy_threshold=2, sell_threshold=-2)
            manual.train(train_df)
            sigs        = manual.predict_bulk(oos_df).map({"buy": 1, "hold": 0, "sell": -1}).reindex(oos_df.index)
            raw_scores  = manual.predict_score_bulk(oos_df)
            metric, sh  = _evaluate(manual, "Manual", sigs, raw_scores)
            if metric > best_score:
                best_score, best_sharpe = metric, sh
                best_model, best_name, best_sigs, best_scores = manual, "Manual", sigs, raw_scores
        except Exception as e:
            _log.append(f"  {ticker} Manual error: {e}")

    # ── Approach 4: XGBoost ─────────────────────────────────────────────────────
    if "xgboost" not in exclude:
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
            sigs        = xgb.predict_bulk(oos_df).map({"buy": 1, "hold": 0, "sell": -1}).reindex(oos_df.index)
            raw_scores  = xgb.predict_score_bulk(oos_df)
            metric, sh  = _evaluate(xgb, "XGBoost", sigs, raw_scores)
            if metric > best_score:
                best_score, best_sharpe = metric, sh
                best_model, best_name, best_sigs, best_scores = xgb, "XGBoost", sigs, raw_scores
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
    best_metric_s = (f"Sharpe={best_sharpe:.3f}" if winner_metric == "sharpe"
                     else f"IC={best_score:+.4f} Sharpe={best_sharpe:+.3f}")
    _log.append(f"  → WINNER: {best_name}  {best_metric_s}  "
                f"{'✓ PASS' if passes else '✗ FAIL (no model exported)'}\n")

    return {
        "sharpe":            best_sharpe,
        "selection_metric":  winner_metric,
        "selection_score":   best_score,       # the value used for winner choice (ic or sharpe)
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
    oos_cutoff   = resolve_oos_cutoff(config)

    # Tournament can skip approaches via config — default behaviour unchanged
    # (empty exclude set). Values are case-insensitive.
    tournament_cfg = config.get("ranking", {}).get("tournament", {})
    exclude_models = set(tournament_cfg.get("exclude_models", []))
    winner_metric  = tournament_cfg.get("winner_metric", "sharpe")

    tickers = [t for t in watchlist if t in feature_frames]
    if not tickers:
        return {}

    n_workers = min(len(tickers), max_workers or os.cpu_count() or 4)
    # Each worker gets 1 XGBoost thread; remaining cores fill in from the OS scheduler
    xgb_nthread = max(1, (os.cpu_count() or 4) // n_workers)

    print(f"Tournament: {len(tickers)} tickers, {n_workers} workers, "
          f"XGBoost nthread={xgb_nthread} per worker")

    # Ensure worker processes (spawn) can import training.*
    # Round-3 audit (#R3-28): previously the env mutation persisted for the
    # rest of the process. Snapshot + restore on exit so subsequent code in
    # the same process doesn't see the polluted env.
    _orig_pythonpath = os.environ.get("PYTHONPATH", "")
    if _STRATEGY_DIR not in _orig_pythonpath:
        os.environ["PYTHONPATH"] = _STRATEGY_DIR + (
            ":" + _orig_pythonpath if _orig_pythonpath else ""
        )

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
                oos_cutoff,
                exclude_models,
                winner_metric,
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

    # Restore PYTHONPATH (R3-#28).
    if _orig_pythonpath:
        os.environ["PYTHONPATH"] = _orig_pythonpath
    elif "PYTHONPATH" in os.environ:
        del os.environ["PYTHONPATH"]

    return results
