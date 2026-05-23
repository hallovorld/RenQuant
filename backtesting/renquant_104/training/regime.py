"""Training-time regime utilities: build_gmm_features, RegimeGMM.

Requires sklearn (GaussianMixture, StandardScaler).  Used by the notebook
for training and exporting the SPY GMM artifact.

Inference (live runner, LEAN) uses kernel.regime.gmm_predict which reads
the saved JSON artifact without sklearn.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Re-export pure-Python regime functions from kernel for convenience
from kernel.regime import (
    compute_hurst,
    compute_cusum,
    rolling_hurst,
    rolling_cusum,
)


def build_gmm_features(
    spy_ohlcv: pd.DataFrame,
    vol_window: int = 20,
    hurst_window: int = 63,
) -> pd.DataFrame:
    """Build the 4-feature frame used to train the SPY GMM.

    Features: 10d_return, 20d_realized_vol (annualised), spy_adx_14, return_autocorr
    """
    close = spy_ohlcv["close"]
    high  = spy_ohlcv["high"]
    low   = spy_ohlcv["low"]
    rets  = close.pct_change()

    feat = pd.DataFrame(index=spy_ohlcv.index)
    feat["10d_return"] = close.pct_change(10)
    feat["20d_realized_vol"] = rets.rolling(vol_window).std() * np.sqrt(252)

    up_move  = high.diff()
    down_move = -low.diff()
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    period   = 14
    atr      = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm, index=spy_ohlcv.index).ewm(
        alpha=1/period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=spy_ohlcv.index).ewm(
        alpha=1/period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    feat["spy_adx"] = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    feat["return_autocorr"] = rets.rolling(20).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0.0,
        raw=True,
    )

    return feat.dropna()


class RegimeGMM:
    """GMM-based regime classifier with JSON serialisation."""

    def __init__(self, n_components: int = 3, random_state: int = 42, n_init: int = 10):
        self.n_components  = n_components
        self._gmm          = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state,
            n_init=n_init,
        )
        self._scaler       = StandardScaler()
        self.cluster_labels: list[str] = []
        self._feature_cols: list[str]  = []
        self._fitted       = False

    def fit(self, features: pd.DataFrame, cluster_labels: list[str] | None = None) -> "RegimeGMM":
        if features.empty:
            raise ValueError("RegimeGMM.fit received an empty feature frame")
        arr = features.values
        if not np.isfinite(arr).all():
            bad_cols = [
                col for i, col in enumerate(features.columns)
                if not np.isfinite(arr[:, i]).all()
            ]
            raise ValueError(
                "RegimeGMM.fit received non-finite feature values "
                f"in columns {bad_cols}"
            )
        self._feature_cols = list(features.columns)
        X = self._scaler.fit_transform(features.values)
        self._gmm.fit(X)
        self._fitted = True
        if cluster_labels is not None:
            assert len(cluster_labels) == self.n_components
            self.cluster_labels = cluster_labels
        else:
            self.cluster_labels = self._auto_label(features)
        return self

    def _auto_label(self, features: pd.DataFrame) -> list[str]:
        centers   = self._scaler.inverse_transform(self._gmm.means_)
        center_df = pd.DataFrame(centers, columns=self._feature_cols)
        n         = self.n_components
        order     = center_df["20d_realized_vol"].argsort().values
        labels    = [""] * n
        if n == 3:
            labels[order[0]] = "BULL_CALM"
            labels[order[2]] = "BEAR"
            labels[order[1]] = "BULL_VOLATILE"
        else:
            regime_names = ["BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"]
            for rank, idx in enumerate(order):
                labels[idx] = regime_names[min(rank, len(regime_names) - 1)]
        return labels

    def predict(self, features: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        X         = self._scaler.transform(features[self._feature_cols].values)
        proba     = self._gmm.predict_proba(X)
        label_idx = proba.argmax(axis=1)
        label_names = [self.cluster_labels[i] for i in label_idx]
        proba_df  = pd.DataFrame(proba, index=features.index, columns=self.cluster_labels)
        return pd.Series(label_names, index=features.index), proba_df

    def save(
        self,
        path: str | Path,
        *,
        as_of_date: str | None = None,
        data_window_start: str | None = None,
        data_window_end: str | None = None,
        n_train_rows: int | None = None,
    ) -> None:
        artifact = {
            "means":          self._gmm.means_.tolist(),
            "covariances":    self._gmm.covariances_.tolist(),
            "weights":        self._gmm.weights_.tolist(),
            "cluster_labels": self.cluster_labels,
            "feature_order":  self._feature_cols,
            "scaler_mean":    self._scaler.mean_.tolist(),
            "scaler_scale":   self._scaler.scale_.tolist(),
        }
        if as_of_date is not None:
            artifact["as_of_date"] = as_of_date
            artifact["trained_date"] = as_of_date
        if data_window_start is not None:
            artifact["data_window_start"] = data_window_start
        if data_window_end is not None:
            artifact["data_window_end"] = data_window_end
        if n_train_rows is not None:
            artifact["n_train_rows"] = int(n_train_rows)
        Path(path).write_text(json.dumps(artifact, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RegimeGMM":
        artifact = json.loads(Path(path).read_text())
        obj = cls(n_components=len(artifact["means"]))
        obj._feature_cols  = artifact["feature_order"]
        obj.cluster_labels = artifact["cluster_labels"]
        obj._scaler.mean_  = np.array(artifact["scaler_mean"])
        obj._scaler.scale_ = np.array(artifact["scaler_scale"])
        gmm = GaussianMixture(n_components=len(artifact["means"]))
        gmm.means_       = np.array(artifact["means"])
        gmm.covariances_ = np.array(artifact["covariances"])
        gmm.weights_     = np.array(artifact["weights"])
        gmm.precisions_chol_ = np.array([
            np.linalg.cholesky(np.linalg.inv(c)) for c in gmm.covariances_
        ])
        gmm.converged_   = True
        gmm.n_iter_      = 0
        obj._gmm         = gmm
        obj._fitted      = True
        return obj


__all__ = [
    "build_gmm_features", "RegimeGMM",
    "compute_hurst", "compute_cusum", "rolling_hurst", "rolling_cusum",
]
