"""Model artifact export and live model refresh for renquant_103.

Exports tournament winners to models/{ticker}/ and optionally retrains
on an expanding window (last 4 years) for live trading.

Exports:
  export_models(results, strategy_dir, today, lookahead, strategy_name)
      -> (exported: list[str], skipped: list[str])
  retrain_live_models(results, feature_frames, exported, strategy_dir, model_params, config, today, ohlcv=None)

Admission decisions (sharpe / ic / ... floors) live in
kernel.pipeline.job_universe.LoadUniverseJob and are applied at load
time by LeanAdapter / RunnerAdapter / SimAdapter.  Export emits every
ticker with a trained model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .models import create_model, XGBoostModel
from .tournament import oos_sharpe

_LIVE_HOLDOUT_DAYS = 126  # ~6 months of trading days
_LIVE_HOLDOUT_MIN_TRAIN = 60  # skip holdout if train portion would be shorter than this


def export_models(
    results: dict[str, dict],
    strategy_dir: Path,
    today: str,
    lookahead: int,
    strategy_name: str,
    models_root: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Save trained models to models/{ticker}/; patch metadata; return (exported, skipped).

    Export emits every ticker with a non-None model. Universe admission
    happens at load time via kernel.pipeline.job_universe.LoadUniverseJob,
    which reads ranking.universe_floor.{type, threshold} and applies the
    configured floor (none / sharpe / ic / ...). This separation keeps
    all admission decisions in one place (one Job, one config key).

    models_root (campaign A2, F-17): write-target override used by the
    tournament-acceptance staging flow. None (default) → the production
    ``strategy_dir / "models"`` path, byte-for-byte the pre-A2 behavior.
    """
    base_models_dir = models_root if models_root is not None else strategy_dir / "models"
    exported: list[str] = []
    skipped:  list[str] = []

    for ticker, r in results.items():
        if r.get("model") is None:
            skipped.append(ticker)
            continue
        sym_dir = base_models_dir / ticker
        sym_dir.mkdir(parents=True, exist_ok=True)
        r["model"].save(sym_dir, model_name=ticker)

        meta_path = sym_dir / f"{ticker}-policy-metadata.json"
        if meta_path.exists():
            with meta_path.open() as f:
                meta = json.load(f)
            meta["trained_date"]  = today
            meta["best_approach"] = r["best_approach"]
            meta["sharpe"]        = round(r["sharpe"], 4)
            meta["lookahead"]     = lookahead
            meta["strategy"]      = strategy_name
            if r.get("score_calibration") is not None:
                meta["score_calibration"] = r["score_calibration"].to_dict()
            with meta_path.open("w") as f:
                json.dump(meta, f, indent=2)
        exported.append(ticker)

    print(f"Exported : {sorted(exported)}")
    print(f"Skipped (no model): {sorted(skipped)}")
    return exported, skipped


def _build_unfitted_live_model(approach: str, feature_cols: list[str], mp: dict, seed: int):
    """Construct (not train) the live model for a chosen approach."""
    lookahead = mp["lookahead"]
    threshold = mp["threshold"]
    bags      = mp["bags"]
    leaf_size = mp["leaf_size"]
    buy_thr   = mp["buy_threshold"]
    sell_thr  = mp["sell_threshold"]

    np.random.seed(seed)
    if approach == "Classification":
        return create_model("classification", feature_columns=feature_cols,
                            lookahead=lookahead, threshold=threshold,
                            leaf_size=leaf_size, bags=bags,
                            buy_threshold=buy_thr, sell_threshold=sell_thr)
    if approach == "QLearning":
        import random as _random
        _random.seed(seed)
        return create_model("qlearning", feature_columns=feature_cols[:5])
    if approach == "XGBoost":
        return XGBoostModel(feature_columns=feature_cols,
                            lookahead=lookahead, threshold=threshold,
                            buy_threshold=0.1, sell_threshold=0.1,
                            n_estimators=200, max_depth=4,
                            learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=10)
    if approach == "Manual":
        return create_model("manual", buy_threshold=2, sell_threshold=-2)
    return None


def _compute_live_holdout_sharpe(
    approach: str,
    df_full: pd.DataFrame,
    prices: pd.Series,
    feature_cols: list[str],
    mp: dict,
    seed: int,
) -> float | None:
    """Train a fresh model on df_full[:-holdout] and evaluate OOS Sharpe on the tail.

    Returns None when df_full is too short to carve out a meaningful holdout.
    """
    if len(df_full) < _LIVE_HOLDOUT_DAYS + _LIVE_HOLDOUT_MIN_TRAIN:
        return None
    df_train   = df_full.iloc[:-_LIVE_HOLDOUT_DAYS]
    df_holdout = df_full.iloc[-_LIVE_HOLDOUT_DAYS:]
    holdout_model = _build_unfitted_live_model(approach, feature_cols, mp, seed)
    if holdout_model is None:
        return None
    holdout_model.train(df_train)
    sigs = (holdout_model.predict_bulk(df_holdout)
            .map({"buy": 1, "hold": 0, "sell": -1})
            .reindex(df_holdout.index))
    holdout_prices = prices.reindex(df_holdout.index)
    return oos_sharpe(holdout_prices, sigs)


def retrain_live_models(
    results: dict[str, dict],
    feature_frames: dict[str, pd.DataFrame],
    exported: list[str],
    strategy_dir: Path,
    model_params: dict,
    config: dict,
    today: str,
    live_train_years: int = 4,
    ohlcv: dict[str, pd.DataFrame] | None = None,
    models_root: Path | None = None,
) -> None:
    """Retrain each exported model on the last N years of data; overwrite artifacts.

    Also computes a walk-forward holdout Sharpe (last ~6 months of the 4yr window)
    and writes it to metadata as `live_holdout_sharpe`, so LEAN/live can filter
    on a figure that reflects the shipped weights — not the tournament weights.

    ohlcv: optional. When provided, holdout Sharpe uses absolute close prices from
    ohlcv[ticker]["close"]; without it, holdout is skipped.

    models_root (campaign A2, F-17): write-target override for the
    tournament-acceptance staging flow. None (default) → production path.
    """
    base_models_dir = models_root if models_root is not None else strategy_dir / "models"
    feature_cols = model_params["feature_columns"]
    lookahead    = model_params["lookahead"]
    strategy     = config.get("strategy", "renquant_103")

    print(f"\n=== Live model refresh: expanding window up to {today} ===")
    for ticker in exported:
        if ticker not in feature_frames:
            continue
        cutoff_4yr = pd.Timestamp(today) - pd.DateOffset(years=live_train_years)
        df_full = feature_frames[ticker][feature_frames[ticker].index >= cutoff_4yr]
        if len(df_full) < 60:
            df_full = feature_frames[ticker]
        best_approach = results[ticker]["best_approach"]
        sym_dir = base_models_dir / ticker
        _seed   = abs(hash(ticker)) % (2 ** 32)

        try:
            # ── walk-forward holdout (skipped when ohlcv unavailable) ──
            holdout_sharpe: float | None = None
            if ohlcv is not None and ticker in ohlcv:
                try:
                    holdout_sharpe = _compute_live_holdout_sharpe(
                        best_approach, df_full,
                        ohlcv[ticker]["close"],
                        feature_cols, model_params, _seed,
                    )
                except Exception as e:
                    print(f"  {ticker}: holdout Sharpe FAILED — {e}")

            # ── train shipped model on full window ──
            live_model = _build_unfitted_live_model(best_approach, feature_cols, model_params, _seed)
            if live_model is None:
                print(f"  {ticker}: unknown approach {best_approach!r}, skipping")
                continue

            live_model.train(df_full)
            live_model.save(sym_dir, model_name=ticker)

            meta_path = sym_dir / f"{ticker}-policy-metadata.json"
            if meta_path.exists():
                with meta_path.open() as f:
                    meta = json.load(f)
                meta["trained_date"]    = today
                meta["best_approach"]   = best_approach
                meta["sharpe"]          = round(results[ticker]["sharpe"], 4)
                meta["lookahead"]       = lookahead
                meta["strategy"]        = strategy
                meta["live_train_rows"] = len(df_full)
                meta["live_train_end"]  = str(df_full.index[-1].date())
                if holdout_sharpe is not None:
                    meta["live_holdout_sharpe"] = round(holdout_sharpe, 4)
                    meta["live_holdout_days"]   = _LIVE_HOLDOUT_DAYS
                if results[ticker].get("score_calibration") is not None:
                    meta["score_calibration"] = results[ticker]["score_calibration"].to_dict()
                with meta_path.open("w") as f:
                    json.dump(meta, f, indent=2)

            hs = f" holdout_sharpe={holdout_sharpe:.3f}" if holdout_sharpe is not None else ""
            print(f"  {ticker}: {best_approach}, {len(df_full)} rows, "
                  f"train_end={df_full.index[-1].date()}{hs}")
        except Exception as e:
            print(f"  {ticker}: refresh FAILED — {e}")

    print("Live model refresh complete.")


# ── Per-ticker helpers (used by TickerExportJob in parallel pipeline) ──────────

def export_one_model(
    ticker: str,
    result: dict,
    strategy_dir: Path,
    today: str,
    lookahead: int,
    strategy_name: str,
    models_root: Path | None = None,
) -> bool:
    """Export one ticker's model artifact; return True if exported.

    Admission decisions live in LoadUniverseJob (kernel.pipeline.job_universe).
    Export only skips when there is no trained model to save.

    models_root (campaign A2, F-17): write-target override for the
    tournament-acceptance staging flow. None (default) → production path.
    """
    if result.get("model") is None:
        return False
    base_models_dir = models_root if models_root is not None else strategy_dir / "models"
    sym_dir = base_models_dir / ticker
    sym_dir.mkdir(parents=True, exist_ok=True)
    result["model"].save(sym_dir, model_name=ticker)

    meta_path = sym_dir / f"{ticker}-policy-metadata.json"
    if meta_path.exists():
        import json as _json
        with meta_path.open() as f:
            meta = _json.load(f)
        meta["trained_date"]  = today
        meta["best_approach"] = result["best_approach"]
        meta["sharpe"]        = round(result["sharpe"], 4)
        meta["lookahead"]     = lookahead
        meta["strategy"]      = strategy_name
        if result.get("score_calibration") is not None:
            meta["score_calibration"] = result["score_calibration"].to_dict()
        with meta_path.open("w") as f:
            _json.dump(meta, f, indent=2)
    return True


def retrain_one_live_model(
    ticker: str,
    result: dict,
    feature_frame: pd.DataFrame,
    strategy_dir: Path,
    model_params: dict,
    config: dict,
    today: str,
    live_train_years: int = 4,
    ohlcv: dict[str, pd.DataFrame] | None = None,
    models_root: Path | None = None,
) -> None:
    """Retrain one ticker's live model on the last N years; overwrite artifact."""
    retrain_live_models(
        {ticker: result},
        {ticker: feature_frame},
        [ticker],
        strategy_dir,
        model_params,
        config,
        today,
        live_train_years=live_train_years,
        ohlcv=ohlcv,
        models_root=models_root,
    )
