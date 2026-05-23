#!/usr/bin/env python
"""Build a leakage-free SPY GMM regime artifact for sim.

Fits a 3-component Gaussian mixture on SPY 4-feature regime descriptors
(10d return, 20d realized vol, 14d ADX, return autocorr) using ONLY
SPY history ≤ ``--as-of``. The resulting artifact follows the format
``kernel.regime.load_gmm_artifact`` expects (means + covariances +
weights + scaler + cluster_labels).

Per CLAUDE.md §5.13.5: production GMM lives in artifacts/prod/ and
is trained weekly off the latest SPY. This script is sim-only —
locking --as-of guarantees the resulting GMM has no forward-looking
information beyond the cutoff.

Usage::

    python scripts/build_sim_gmm.py --as-of 2023-12-31

Output: backtesting/renquant_104/artifacts/sim/spy-gmm-regime.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("build-sim-gmm")


def _build_features(spy: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Reproduce the 4 regime features used at inference time.

    Order MUST match the production GMM's ``feature_order`` array so
    ``ctx.regime`` dispatch works identically. Mirrors
    ``kernel.regime.compute_spy_adx`` + helpers.
    """
    from kernel.indicators import compute_atr  # noqa: PLC0415

    close = spy["close"].astype(float)
    high = spy["high"].astype(float)
    low = spy["low"].astype(float)
    rets = close.pct_change()

    feat = pd.DataFrame(index=close.index)
    feat["10d_return"] = close.pct_change(periods=10)
    feat["20d_realized_vol"] = rets.rolling(20).std(ddof=1) * np.sqrt(252)

    # ADX(14) reusing the same TR pattern as kernel.regime.compute_spy_adx.
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = compute_atr(high, low, close, period=period)
    plus_di = (100 * pd.Series(plus_dm, index=close.index)
               .ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
               / atr.replace(0, np.nan))
    minus_di = (100 * pd.Series(minus_dm, index=close.index)
                .ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
                / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    feat["spy_adx"] = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    feat["return_autocorr"] = rets.rolling(20).apply(
        lambda v: v.autocorr() if len(v.dropna()) > 5 else np.nan, raw=False
    )
    return feat.dropna()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of", required=True,
                   help="ISO date (YYYY-MM-DD). Last day included.")
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--strategy-config-name",
                   default="strategy_config.sim_baseline.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    sys.path.insert(0, str(strategy_dir))
    as_of = pd.Timestamp(args.as_of)

    from kernel.data import fetch_ohlcv  # noqa: PLC0415
    from sklearn.mixture import GaussianMixture  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    log.info("Loading SPY history ≤ %s", as_of.date())
    spy = fetch_ohlcv("SPY")
    if spy is None or spy.empty:
        log.error("SPY OHLCV unavailable")
        sys.exit(2)
    spy = spy[spy.index <= as_of]
    if len(spy) < 252:
        log.error("Only %d SPY bars ≤ %s; need ≥252 for GMM training",
                  len(spy), as_of.date())
        sys.exit(2)

    feat = _build_features(spy)
    if feat.empty:
        log.error("No finite SPY regime feature rows available ≤ %s", as_of.date())
        sys.exit(2)
    values = feat.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad_cols = [
            col for i, col in enumerate(feat.columns)
            if not np.isfinite(values[:, i]).all()
        ]
        log.error("Non-finite SPY regime features in columns: %s", bad_cols)
        sys.exit(2)
    log.info("Built %d feature rows (cols=%s)", len(feat), list(feat.columns))

    scaler = StandardScaler()
    X = scaler.fit_transform(feat.values)
    gmm = GaussianMixture(
        n_components=3, covariance_type="full",
        random_state=args.seed, n_init=10, max_iter=500,
    )
    gmm.fit(X)
    log.info("GMM fit: means_norms=%s weights=%s",
             np.linalg.norm(gmm.means_, axis=1).round(3).tolist(),
             gmm.weights_.round(3).tolist())

    # 2026-05-11 G3 fix: use the SAME label heuristic as production
    # (training/regime.RegimeGMM._auto_label): sort clusters by
    # 20d_realized_vol ASC → [lowest-vol=BULL_CALM, mid=BULL_VOLATILE,
    # highest=BEAR]. Audit Round 2 caught this drift — sim was using a
    # return-first then vol heuristic, producing different cluster_labels
    # than prod and changing the semantic regime each bar falls into.
    feature_cols = list(feat.columns)
    centers = scaler.inverse_transform(gmm.means_)
    inv_df = pd.DataFrame(centers, columns=feature_cols)
    vol_order = inv_df["20d_realized_vol"].argsort().values
    cluster_labels = [""] * 3
    cluster_labels[vol_order[0]] = "BULL_CALM"
    cluster_labels[vol_order[1]] = "BULL_VOLATILE"
    cluster_labels[vol_order[2]] = "BEAR"

    out_path = (
        Path(args.out) if args.out else
        strategy_dir / "artifacts" / "sim" / "spy-gmm-regime.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "trained_date": as_of.strftime("%Y-%m-%d"),
        "n_train_rows": int(len(feat)),
        "feature_order": list(feat.columns),
        "means": gmm.means_.tolist(),
        "covariances": gmm.covariances_.tolist(),
        "weights": gmm.weights_.tolist(),
        "cluster_labels": cluster_labels,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    out_path.write_text(json.dumps(artifact, indent=2))
    log.info("Saved → %s (cluster_labels=%s)", out_path, cluster_labels)


if __name__ == "__main__":
    main()
