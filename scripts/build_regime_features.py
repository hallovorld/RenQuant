#!/usr/bin/env python
"""Compute SPY regime probabilities and add as features to alpha158 panel.

Uses the existing GMM artifact (backtesting/renquant_104/artifacts/spy-gmm-regime.json)
which has 3 clusters: BULL_VOLATILE, BEAR, BULL_CALM, fitted on:
  - 10d_return
  - 20d_realized_vol
  - spy_adx
  - return_autocorr

For each trading day, computes the 4 SPY-derived features and the GMM
posterior probability of each regime. The 3 regime probs are added as
features (regime_p_bull_volatile, regime_p_bear, regime_p_bull_calm).

Output: data/alpha158_291_fund_regime_dataset.parquet

Usage:
    python scripts/build_regime_features.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("build-regime")

REPO = Path(__file__).resolve().parent.parent


def compute_spy_features(spy_close: pd.Series) -> pd.DataFrame:
    """Compute the 4 SPY features the GMM was trained on."""
    feats = pd.DataFrame(index=spy_close.index)

    # 10d_return
    feats["10d_return"] = spy_close.pct_change(periods=10)

    # 20d_realized_vol (annualized stdev of 1-day returns × √252)
    daily_ret = spy_close.pct_change()
    feats["20d_realized_vol"] = daily_ret.rolling(20).std() * np.sqrt(252)

    # spy_adx (simplified: 14-day directional movement strength)
    # Using diff-based momentum sign-strength as proxy
    diff_close = spy_close.diff()
    pos_dm = diff_close.where(diff_close > 0, 0)
    neg_dm = (-diff_close).where(diff_close < 0, 0)
    pos_di = pos_dm.rolling(14).mean()
    neg_di = neg_dm.rolling(14).mean()
    dx = (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-9) * 100
    feats["spy_adx"] = dx.rolling(14).mean()

    # return_autocorr (1-day lag autocorrelation of returns over 20d window)
    feats["return_autocorr"] = daily_ret.rolling(20).apply(
        lambda x: x.autocorr(lag=1) if len(x.dropna()) >= 5 else np.nan, raw=False
    )

    return feats


def gmm_posterior(X: np.ndarray, means: np.ndarray, covariances: np.ndarray,
                  weights: np.ndarray) -> np.ndarray:
    """Compute posterior probability for each component (full-cov GMM)."""
    n_samples, _ = X.shape
    n_components = len(weights)
    log_probs = np.zeros((n_samples, n_components))
    for k in range(n_components):
        try:
            rv = multivariate_normal(mean=means[k], cov=covariances[k],
                                      allow_singular=True)
            log_probs[:, k] = rv.logpdf(X) + np.log(weights[k] + 1e-12)
        except Exception:
            log_probs[:, k] = -1e10
    # Normalize via log-sum-exp
    max_lp = log_probs.max(axis=1, keepdims=True)
    probs = np.exp(log_probs - max_lp)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def main():
    log.info("Loading GMM artifact...")
    gmm = json.loads((REPO / "backtesting" / "renquant_104" / "artifacts"
                      / "spy-gmm-regime.json").read_text())
    cluster_labels = gmm["cluster_labels"]
    means          = np.array(gmm["means"])
    covariances    = np.array(gmm["covariances"])
    weights        = np.array(gmm["weights"])
    feature_order  = gmm["feature_order"]
    scaler_mean    = np.array(gmm["scaler_mean"])
    scaler_scale   = np.array(gmm["scaler_scale"])
    log.info("GMM: %d clusters %s, %d features %s",
             len(cluster_labels), cluster_labels, len(feature_order), feature_order)

    log.info("Loading SPY OHLCV...")
    spy = pd.read_parquet(REPO / "data" / "ohlcv" / "SPY" / "1d.parquet")
    spy.index = pd.to_datetime(spy.index)
    spy = spy.sort_index()
    log.info("SPY: %d bars, %s → %s", len(spy), spy.index.min().date(), spy.index.max().date())

    log.info("Computing SPY regime features...")
    feats = compute_spy_features(spy["close"])
    feats = feats[feature_order]   # ensure column order matches GMM
    feats = feats.dropna()
    log.info("SPY features: %d valid days", len(feats))

    # Scale + GMM posterior
    X = (feats.values - scaler_mean) / scaler_scale
    probs = gmm_posterior(X, means, covariances, weights)
    regime_df = pd.DataFrame(probs, index=feats.index,
                              columns=[f"regime_p_{c.lower()}" for c in cluster_labels])
    regime_df.index.name = "date"
    log.info("Posterior probs: shape=%s, mean per cluster: %s",
             probs.shape, dict(zip(cluster_labels, probs.mean(axis=0).round(3))))

    # Merge with alpha158+fund panel
    log.info("Merging into alpha158_291_fundamental_dataset.parquet...")
    panel = pd.read_parquet(REPO / "data" / "alpha158_291_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    regime_df = regime_df.reset_index()
    merged = panel.merge(regime_df, on="date", how="left")

    # Forward-fill missing regime values (for trading days with no SPY signal)
    regime_cols = list(regime_df.columns)
    regime_cols.remove("date")
    merged[regime_cols] = merged.groupby("ticker")[regime_cols].ffill()
    merged[regime_cols] = merged[regime_cols].fillna(1.0 / len(cluster_labels))  # uniform if still NaN

    out = REPO / "data" / "alpha158_291_fund_regime_dataset.parquet"
    merged.to_parquet(out, index=False)
    log.info("Written %s: %d rows × %d cols, +3 regime features",
             out, len(merged), len(merged.columns))


if __name__ == "__main__":
    main()
