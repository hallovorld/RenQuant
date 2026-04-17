"""Regime detection indicators: Hurst exponent, CUSUM changepoint, GMM classification.

Usage::

    from common.indicators.regime import compute_hurst, compute_cusum, RegimeGMM

    hurst = compute_hurst(spy_returns, window=63)
    transition = compute_cusum(spy_returns, threshold=3.0, drift=0.5)

    gmm = RegimeGMM(n_components=3)
    gmm.fit(feature_df)
    labels, probs = gmm.predict(feature_df)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ── Layer 1: Hurst Exponent ───────────────────────────────────────────────────

def compute_hurst(returns: pd.Series | np.ndarray, max_lag: int = 40) -> float:
    """Rescaled Range (R/S) Hurst exponent.

    Parameters
    ----------
    returns : daily return series
    max_lag : upper bound on lag range (capped at len//2)

    Returns
    -------
    H in [0, 1].  0.5 = random walk, >0.55 = trending, <0.45 = mean-reverting.
    """
    arr = np.asarray(returns, dtype=float)
    n   = len(arr)
    if n < 10:
        return 0.5

    lags    = range(2, min(n // 2, max_lag))
    rs_vals = []
    for lag in lags:
        chunks   = [arr[i:i + lag] for i in range(0, n - lag, lag)]
        rs_chunk = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean = chunk.mean()
            devs = np.cumsum(chunk - mean)
            R    = devs.max() - devs.min()
            S    = chunk.std(ddof=1)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_vals.append(np.mean(rs_chunk))

    if len(rs_vals) < 2:
        return 0.5
    try:
        lags_used = list(range(2, 2 + len(rs_vals)))
        poly      = np.polyfit(np.log(lags_used), np.log(rs_vals), 1)
        return float(np.clip(poly[0], 0.0, 1.0))
    except Exception:
        return 0.5


def rolling_hurst(returns: pd.Series, window: int = 63, max_lag: int = 40) -> pd.Series:
    """Rolling Hurst exponent on a return series."""
    result = pd.Series(index=returns.index, dtype=float)
    for i in range(window, len(returns) + 1):
        chunk         = returns.iloc[i - window:i]
        result.iloc[i - 1] = compute_hurst(chunk, max_lag=max_lag)
    return result


# ── Layer 2: CUSUM Changepoint ────────────────────────────────────────────────

def compute_cusum(returns: pd.Series | np.ndarray,
                  threshold: float = 3.0,
                  drift: float     = 0.5,
                  reference_returns: pd.Series | np.ndarray | None = None) -> bool:
    """CUSUM changepoint test.

    Returns True if a structural break is detected in ``returns``.
    When ``reference_returns`` is provided, that series defines the in-control
    baseline used to estimate mean and standard deviation.
    """
    arr = np.asarray(returns, dtype=float)
    ref = np.asarray(reference_returns, dtype=float) if reference_returns is not None else arr
    if len(arr) < 5 or len(ref) < 5:
        return False
    mu    = ref.mean()
    sigma = ref.std(ddof=1)
    if sigma <= 0:
        return False
    s_pos = s_neg = 0.0
    for r in arr:
        z     = (r - mu) / sigma
        s_pos = max(0.0, s_pos + z - drift)
        s_neg = max(0.0, s_neg - z - drift)
        if s_pos > threshold or s_neg > threshold:
            return True
    return False


def rolling_cusum(returns: pd.Series,
                  window:    int   = 20,
                  threshold: float = 3.0,
                  drift:     float = 0.5) -> pd.Series:
    """Rolling CUSUM using the prior window as the in-control baseline."""
    result = pd.Series(False, index=returns.index)
    for i in range(window * 2, len(returns) + 1):
        reference = returns.iloc[i - (window * 2):i - window].values
        chunk = returns.iloc[i - window:i].values
        result.iloc[i - 1] = compute_cusum(
            chunk,
            threshold=threshold,
            drift=drift,
            reference_returns=reference,
        )
    return result


# ── Layer 3: GMM Regime Classifier ───────────────────────────────────────────

_REGIME_LABELS = ["BULL_CALM", "BULL_VOLATILE", "BEAR"]


def build_gmm_features(spy_ohlcv: pd.DataFrame,
                       vol_window:  int = 20,
                       hurst_window: int = 63) -> pd.DataFrame:
    """Build the 4-feature frame used to train the SPY GMM.

    Features: 10d_return, 20d_realized_vol (annualised), spy_adx_14, return_autocorr
    """
    close = spy_ohlcv["close"]
    high  = spy_ohlcv["high"]
    low   = spy_ohlcv["low"]
    rets  = close.pct_change()

    feat = pd.DataFrame(index=spy_ohlcv.index)

    # 10-day log return
    feat["10d_return"] = close.pct_change(10)

    # 20-day realized vol (annualised)
    feat["20d_realized_vol"] = rets.rolling(vol_window).std() * np.sqrt(252)

    # ADX (14)
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

    # Return autocorrelation lag=1 over rolling 20-day window
    feat["return_autocorr"] = rets.rolling(20).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0.0,
        raw=True,
    )

    return feat.dropna()


class RegimeGMM:
    """GMM-based regime classifier. Wraps sklearn GaussianMixture with:
      - standardisation
      - post-hoc cluster label assignment
      - JSON serialisation for LEAN compatibility
    """

    def __init__(self, n_components: int = 3, random_state: int = 42, n_init: int = 10):
        self.n_components  = n_components
        self._gmm          = GaussianMixture(
            n_components   = n_components,
            covariance_type = "full",
            random_state   = random_state,
            n_init         = n_init,
        )
        self._scaler       = StandardScaler()
        self.cluster_labels: list[str] = []
        self._feature_cols: list[str]  = []
        self._fitted       = False

    def fit(self, features: pd.DataFrame,
            cluster_labels: list[str] | None = None) -> "RegimeGMM":
        """Fit GMM on feature DataFrame. Assigns labels by post-hoc cluster inspection."""
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
        """Auto-assign labels based on cluster center statistics.

        Logic:
          - Lowest realized vol + positive 10d return → BULL_CALM
          - Highest realized vol + mixed/negative return → BEAR
          - Middle → BULL_VOLATILE
        """
        centers    = self._scaler.inverse_transform(self._gmm.means_)
        center_df  = pd.DataFrame(centers, columns=self._feature_cols)
        n          = self.n_components

        # Sort by realized vol ascending
        vol_col = "20d_realized_vol"
        ret_col = "10d_return"
        order   = center_df[vol_col].argsort().values  # low → high vol

        labels  = [""] * n
        if n == 3:
            labels[order[0]] = "BULL_CALM"      # lowest vol
            labels[order[2]] = "BEAR"            # highest vol
            labels[order[1]] = "BULL_VOLATILE"  # middle
        else:
            # Generic: rank by vol, assign sequentially
            regime_names = ["BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"]
            for rank, idx in enumerate(order):
                labels[idx] = regime_names[min(rank, len(regime_names) - 1)]
        return labels

    def predict(self, features: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        """Return (label_series, proba_DataFrame) for each row."""
        X           = self._scaler.transform(features[self._feature_cols].values)
        proba       = self._gmm.predict_proba(X)
        label_idx   = proba.argmax(axis=1)
        label_names = [self.cluster_labels[i] for i in label_idx]
        proba_df    = pd.DataFrame(proba, index=features.index,
                                   columns=self.cluster_labels)
        return pd.Series(label_names, index=features.index), proba_df

    def save(self, path: str | Path) -> None:
        """Serialise to JSON for LEAN consumption."""
        artifact = {
            "means":          self._gmm.means_.tolist(),
            "covariances":    self._gmm.covariances_.tolist(),
            "weights":        self._gmm.weights_.tolist(),
            "cluster_labels": self.cluster_labels,
            "feature_order":  self._feature_cols,
            "scaler_mean":    self._scaler.mean_.tolist(),
            "scaler_scale":   self._scaler.scale_.tolist(),
        }
        Path(path).write_text(json.dumps(artifact, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RegimeGMM":
        artifact = json.loads(Path(path).read_text())
        obj = cls(n_components=len(artifact["means"]))
        obj._feature_cols  = artifact["feature_order"]
        obj.cluster_labels = artifact["cluster_labels"]
        obj._scaler.mean_  = np.array(artifact["scaler_mean"])
        obj._scaler.scale_ = np.array(artifact["scaler_scale"])

        # Reconstruct sklearn GMM internals
        gmm = GaussianMixture(n_components=len(artifact["means"]))
        gmm.means_          = np.array(artifact["means"])
        gmm.covariances_    = np.array(artifact["covariances"])
        gmm.weights_        = np.array(artifact["weights"])
        gmm.precisions_chol_ = np.array([
            np.linalg.cholesky(np.linalg.inv(c)) for c in gmm.covariances_
        ])
        gmm.converged_      = True
        gmm.n_iter_         = 0
        obj._gmm            = gmm
        obj._fitted         = True
        return obj
