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

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler as _BlendScaler
except ImportError:  # pragma: no cover
    LogisticRegression = None
    _BlendScaler = None

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

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
    from training.models import create_model  # lazy — training/ added to sys.path in recalibrate()
    model = create_model(policy_type)
    model.load(models_dir / symbol, symbol)
    model._policy_metadata = metadata
    return model, metadata


def _compute_blend_weights(symbol_data: list[dict]) -> tuple[float, float]:
    """Estimate blend weights from a logistic model on rank_score and RS.

    We fit a pooled logistic regression on per-symbol normalised features:
      P(outperform) = sigmoid(b0 + b1 * norm(rank_score) + b2 * norm(rs_score))

    The positive coefficients are then normalised into live blend weights.
    This is smoother and more defensible than the earlier Pearson-correlation
    heuristic, while preserving the runner's simple weighted-sum ranking.
    """

    def _norm(a: np.ndarray) -> np.ndarray:
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo) if hi > lo else np.full_like(a, 0.5)

    rows: list[np.ndarray] = []
    outcomes: list[np.ndarray] = []
    n_symbols = 0

    for d in symbol_data:
        rank_arr = np.asarray(d["rank_scores"], dtype=float)
        rs_arr = np.asarray(d["rs_scores"], dtype=float)
        out_arr = np.asarray(d["outcomes"], dtype=float)

        mask = np.isfinite(rank_arr) & np.isfinite(rs_arr) & np.isfinite(out_arr)
        if mask.sum() < 30:
            continue

        rank_arr = rank_arr[mask]
        rs_arr = rs_arr[mask]
        out_arr = out_arr[mask]

        rows.append(np.column_stack([_norm(rank_arr), _norm(rs_arr)]))
        outcomes.append(out_arr.astype(int))
        n_symbols += 1

    if not rows:
        log.warning("Not enough data for blend weight estimation — keeping 0.5 / 0.5")
        return 0.5, 0.5

    if LogisticRegression is None:
        log.warning("scikit-learn unavailable for blend logistic regression — keeping 0.5 / 0.5")
        return 0.5, 0.5

    X = np.vstack(rows)
    y = np.concatenate(outcomes)
    if len(X) < 120 or len(np.unique(y)) < 2:
        log.warning("Not enough labelled rows for blend logistic regression — keeping 0.5 / 0.5")
        return 0.5, 0.5

    # StandardScaler centers and scales each feature column so lbfgs line search
    # stays in a numerically safe range and avoids overflow warnings in matmul.
    if _BlendScaler is not None:
        blend_scaler = _BlendScaler()
        X = blend_scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X, y)
    coef_rank = max(0.0, float(clf.coef_[0][0]))
    coef_rs = max(0.0, float(clf.coef_[0][1]))
    total = coef_rank + coef_rs

    if total < 1e-6:
        log.warning(
            "Blend logistic regression produced non-positive coefficients "
            "(rank=%.4f rs=%.4f) — keeping 0.5 / 0.5",
            float(clf.coef_[0][0]), float(clf.coef_[0][1]),
        )
        return 0.5, 0.5

    w_rank = round(coef_rank / total, 4)
    w_rs = round(coef_rs / total, 4)
    log.info(
        "Blend weights: rank=%.3f  rs=%.3f  "
        "(logit coef rank=%.4f rs=%.4f intercept=%.4f, symbols=%d, rows=%d)",
        w_rank, w_rs,
        float(clf.coef_[0][0]), float(clf.coef_[0][1]), float(clf.intercept_[0]),
        n_symbols, len(X),
    )
    return w_rank, w_rs


def recalibrate(strategy: str, dry_run: bool = False) -> None:
    strategy_dir  = REPO_ROOT / "backtesting" / strategy
    config_path   = strategy_dir / "strategy_config.json"
    models_dir    = strategy_dir / "models"

    if not config_path.exists():
        log.error("Strategy config not found: %s", config_path)
        sys.exit(1)

    # Add strategy dir to sys.path so kernel.* and training.* are importable
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    from kernel.data import fetch_ohlcv
    from kernel.scoring import ScoreCalibration, extract_raw_scores_bulk
    from training.scoring import fit_probability_calibration, raw_score_kind_for_model

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
