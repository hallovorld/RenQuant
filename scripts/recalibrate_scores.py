#!/usr/bin/env python
"""Recalibrate per-symbol model score calibrations after daily retraining.

Run immediately after the notebook retraining step in daily_103.sh so that
the isotonic/Platt curves always reflect the *current* model's score
distribution, not the distribution from when the notebook was last run
manually.

Also computes data-driven ranking blend weights (rank_score vs RS momentum)
from OOS history and writes them to strategy_config.json under
``ranking.blend_weights``.

Usage:
    python scripts/recalibrate_scores.py --strategy renquant_103
    python scripts/recalibrate_scores.py --strategy renquant_103 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.data import fetch_ohlcv
from common.models import create_model
from common.models.scoring import (
    ScoreCalibration,
    extract_raw_scores_bulk,
    fit_probability_calibration,
    raw_score_kind_for_model,
)
from live.runner import _build_relative_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("recalibrate")


def _load_model(models_dir: Path, symbol: str) -> tuple[object, dict] | tuple[None, None]:
    meta_path = models_dir / symbol / f"{symbol}-policy-metadata.json"
    if not meta_path.exists():
        return None, None
    metadata = json.loads(meta_path.read_text())
    policy_type = metadata["policy_type"]
    model = create_model(policy_type)
    model.load(models_dir / symbol, symbol)
    model._policy_metadata = metadata
    return model, metadata


def _compute_blend_weights(symbol_data: list[dict]) -> tuple[float, float]:
    """Estimate blend weights via Pearson correlation averaged across symbols.

    For each symbol we compute:
      - corr(normalised rank_score, binary_outcome)
      - corr(normalised rs_proxy,   binary_outcome)

    Weights are proportional to average positive correlation.
    Falls back to 0.5 / 0.5 when there is not enough data.
    """
    rank_corrs: list[float] = []
    rs_corrs: list[float] = []

    for d in symbol_data:
        rank_arr = np.asarray(d["rank_scores"], dtype=float)
        rs_arr   = np.asarray(d["rs_scores"],   dtype=float)
        out_arr  = np.asarray(d["outcomes"],     dtype=float)

        mask = np.isfinite(rank_arr) & np.isfinite(rs_arr) & np.isfinite(out_arr)
        if mask.sum() < 30:
            continue

        rank_arr, rs_arr, out_arr = rank_arr[mask], rs_arr[mask], out_arr[mask]

        def _norm(a: np.ndarray) -> np.ndarray:
            lo, hi = a.min(), a.max()
            return (a - lo) / (hi - lo) if hi > lo else np.full_like(a, 0.5)

        c_rank = float(np.corrcoef(_norm(rank_arr), out_arr)[0, 1])
        c_rs   = float(np.corrcoef(_norm(rs_arr),   out_arr)[0, 1])
        rank_corrs.append(max(0.0, c_rank))
        rs_corrs.append(max(0.0, c_rs))

    if not rank_corrs:
        log.warning("Not enough data for blend weight estimation — keeping 0.5 / 0.5")
        return 0.5, 0.5

    avg_rank = float(np.mean(rank_corrs))
    avg_rs   = float(np.mean(rs_corrs))
    total    = avg_rank + avg_rs

    if total < 1e-6:
        return 0.5, 0.5

    w_rank = round(avg_rank / total, 4)
    w_rs   = round(avg_rs   / total, 4)
    log.info(
        "Blend weights: rank=%.3f  rs=%.3f  "
        "(from %d symbols, avg corr rank=%.3f rs=%.3f)",
        w_rank, w_rs, len(rank_corrs), avg_rank, avg_rs,
    )
    return w_rank, w_rs


def recalibrate(strategy: str, dry_run: bool = False) -> None:
    strategy_dir  = REPO_ROOT / "backtesting" / strategy
    config_path   = strategy_dir / "strategy_config.json"
    models_dir    = strategy_dir / "models"

    if not config_path.exists():
        log.error("Strategy config not found: %s", config_path)
        sys.exit(1)

    config         = json.loads(config_path.read_text())
    watchlist      = config["watchlist"]
    indicator_spec = config.get("indicator_spec", {})
    feature_cols   = config["model_params"]["feature_columns"]
    lookahead      = int(config["model_params"].get("lookahead", 5))
    threshold      = float(config["model_params"].get("threshold", 0.03))
    benchmark      = config.get("benchmark", "SPY")

    log.info("Fetching benchmark (%s) data …", benchmark)
    df_spy = fetch_ohlcv(benchmark, provider=config.get("data_src", "yfinance"))
    if df_spy.empty:
        log.error("No data for benchmark %s", benchmark)
        sys.exit(1)

    symbol_data: list[dict] = []
    ok, skipped, failed = 0, 0, 0

    for symbol in watchlist:
        model, metadata = _load_model(models_dir, symbol)
        if model is None:
            log.warning("  %-6s  no model — skipping", symbol)
            skipped += 1
            continue

        try:
            df_stock = fetch_ohlcv(symbol, provider=config.get("data_src", "yfinance"))
            if df_stock.empty:
                log.warning("  %-6s  no OHLCV data — skipping", symbol)
                skipped += 1
                continue

            model_feat_cols = getattr(model, "feature_columns", None) or feature_cols
            rel = _build_relative_features(df_stock, df_spy, model_feat_cols, indicator_spec)
            if rel is None or rel.empty:
                log.warning("  %-6s  feature build failed — skipping", symbol)
                skipped += 1
                continue

            features = rel.copy()
            features["position_flag"] = 0
            raw_scores = extract_raw_scores_bulk(model, features)

            # Forward relative return vs SPY
            stock_close = df_stock.loc[rel.index, "close"].astype(float)
            spy_close   = df_spy.loc[rel.index, "close"].astype(float).replace(0, np.nan)
            rel_price   = stock_close / spy_close
            future_rel_returns = rel_price.shift(-lookahead) / rel_price - 1.0

            calibration = fit_probability_calibration(
                raw_scores,
                future_rel_returns,
                lookahead=lookahead,
                threshold=threshold,
                score_kind=raw_score_kind_for_model(model),
            )

            log.info(
                "  %-6s  method=%-20s  n=%-4d  base_rate=%.3f",
                symbol, calibration.method, calibration.sample_size, calibration.base_rate,
            )

            # Collect data for blend weight estimation
            # RS proxy: 20-day stock/SPY relative return (momentum vs market)
            rs_proxy = rel_price.pct_change(20)
            rank_scores_series = raw_scores.apply(calibration.calibrate)
            outcomes = (future_rel_returns > threshold).astype(float)

            common_idx = rank_scores_series.index \
                .intersection(rs_proxy.index) \
                .intersection(outcomes.index)
            symbol_data.append({
                "symbol":      symbol,
                "rank_scores": rank_scores_series.reindex(common_idx).to_numpy(),
                "rs_scores":   rs_proxy.reindex(common_idx).to_numpy(),
                "outcomes":    outcomes.reindex(common_idx).to_numpy(),
            })

            if not dry_run:
                meta_path = models_dir / symbol / f"{symbol}-policy-metadata.json"
                metadata["score_calibration"] = calibration.to_dict()
                metadata["score_calibration_date"] = str(date.today())
                meta_path.write_text(json.dumps(metadata, indent=2))

            ok += 1

        except Exception as exc:
            log.error("  %-6s  ERROR: %s", symbol, exc)
            failed += 1

    log.info("Calibration complete: %d ok  %d skipped  %d failed", ok, skipped, failed)

    # Compute and persist blend weights
    w_rank, w_rs = _compute_blend_weights(symbol_data)
    if not dry_run:
        config.setdefault("ranking", {})
        config["ranking"]["blend_weights"]   = [w_rank, w_rs]
        config["ranking"]["blend_updated"]   = str(date.today())
        config["ranking"]["blend_n_symbols"] = len(symbol_data)
        config_path.write_text(json.dumps(config, indent=2))
        log.info("Updated strategy_config.json: ranking.blend_weights=[%.4f, %.4f]", w_rank, w_rs)
    else:
        log.info("[dry-run] Would write blend_weights=[%.4f, %.4f]", w_rank, w_rs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalibrate model score calibrations")
    parser.add_argument("--strategy", required=True, help="Strategy name, e.g. renquant_103")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute calibrations but do not write any files")
    args = parser.parse_args()
    recalibrate(args.strategy, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
