"""Model artifact export and live model refresh for renquant_103.

Exports tournament winners to models/{ticker}/ and optionally retrains
on an expanding window (last 4 years) for live trading.

Exports:
  export_models(results, strategy_dir, today, sharpe_floor, lookahead, strategy_name)
      -> (exported: list[str], skipped: list[str])
  retrain_live_models(results, feature_frames, exported, strategy_dir, model_params, config, today)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .models import create_model, XGBoostModel


def export_models(
    results: dict[str, dict],
    strategy_dir: Path,
    today: str,
    sharpe_floor: float,
    lookahead: int,
    strategy_name: str,
) -> tuple[list[str], list[str]]:
    """Save winning models to models/{ticker}/; patch metadata; return (exported, skipped)."""
    exported: list[str] = []
    skipped:  list[str] = []

    for ticker, r in results.items():
        if not r.get("passes_floor"):
            skipped.append(ticker)
            continue
        sym_dir = strategy_dir / "models" / ticker
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
    print(f"Skipped (below Sharpe floor {sharpe_floor}): {sorted(skipped)}")
    return exported, skipped


def retrain_live_models(
    results: dict[str, dict],
    feature_frames: dict[str, pd.DataFrame],
    exported: list[str],
    strategy_dir: Path,
    model_params: dict,
    config: dict,
    today: str,
    live_train_years: int = 4,
) -> None:
    """Retrain each exported model on the last N years of data; overwrite artifacts.

    The tournament used a fixed 2024-01-01 cutoff for reproducible OOS evaluation.
    This second pass trains on all recent data so the live model has seen current patterns.
    """
    feature_cols = model_params["feature_columns"]
    lookahead    = model_params["lookahead"]
    threshold    = model_params["threshold"]
    bags         = model_params["bags"]
    leaf_size    = model_params["leaf_size"]
    buy_thr      = model_params["buy_threshold"]
    sell_thr     = model_params["sell_threshold"]
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
        sym_dir = strategy_dir / "models" / ticker
        _seed   = abs(hash(ticker)) % (2 ** 32)

        try:
            np.random.seed(_seed)
            if best_approach == "Classification":
                live_model = create_model("classification", feature_columns=feature_cols,
                                          lookahead=lookahead, threshold=threshold,
                                          leaf_size=leaf_size, bags=bags,
                                          buy_threshold=buy_thr, sell_threshold=sell_thr)
            elif best_approach == "QLearning":
                import random as _random
                _random.seed(_seed)
                live_model = create_model("qlearning", feature_columns=feature_cols[:5])
            elif best_approach == "XGBoost":
                live_model = XGBoostModel(feature_columns=feature_cols,
                                          lookahead=lookahead, threshold=threshold,
                                          buy_threshold=0.1, sell_threshold=0.1,
                                          n_estimators=200, max_depth=4,
                                          learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8, min_child_weight=10)
            elif best_approach == "Manual":
                live_model = create_model("manual", buy_threshold=2, sell_threshold=-2)
            else:
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
                if results[ticker].get("score_calibration") is not None:
                    meta["score_calibration"] = results[ticker]["score_calibration"].to_dict()
                with meta_path.open("w") as f:
                    json.dump(meta, f, indent=2)

            print(f"  {ticker}: {best_approach}, {len(df_full)} rows, "
                  f"train_end={df_full.index[-1].date()}")
        except Exception as e:
            print(f"  {ticker}: refresh FAILED — {e}")

    print("Live model refresh complete.")
