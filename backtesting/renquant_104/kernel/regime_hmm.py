"""HMM-based regime detector (replaces stateless GMM).

Implements Hamilton 1989 Markov-switching with hmmlearn-fit artifact.
Per-bar prediction uses FORWARD FILTERING (causal: only conditions on
past observations) — matches sim/live execution semantics exactly.

Artifact format: see scripts/train_spy_hmm.py. Includes:
  - means [n_states, n_features]
  - covariances [n_states, n_features, n_features]
  - transition_matrix [n_states, n_states]
  - start_prob [n_states]
  - scaler_mean / scaler_scale [n_features]
  - cluster_labels [n_states]
  - feature_order ['r10d', 'ann_vol20', 'adx_proxy', 'autocorr12']

API matches kernel.regime.gmm_predict: takes spy_returns + spy_df,
returns {regime_label: posterior_prob}. Drop-in replacement.

Reference: Hamilton 1989 Econometrica 57:357; Rabiner 1989 IEEE 77:257
(forward-algorithm canonical statement).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("kernel.regime_hmm")


def is_hmm_artifact(artifact: dict | None) -> bool:
    """True iff this artifact is an HMM (with transition_matrix), not legacy GMM."""
    if not isinstance(artifact, dict):
        return False
    return artifact.get("model_type") == "GaussianHMM" \
           and "transition_matrix" in artifact


def load_hmm_artifact(path: Path | str) -> dict | None:
    """Load artifact JSON. Returns None if file missing (caller falls back)."""
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _build_feature_vector(
    spy_returns: np.ndarray,
    spy_df: pd.DataFrame | None,
    vol_window: int = 20,
) -> Optional[np.ndarray]:
    """Compute the 4-feature vector for the LATEST bar (matches gmm_predict input)."""
    if len(spy_returns) < vol_window + 10:
        return None
    recent = spy_returns[-max(vol_window, 11):]
    if not np.all(np.isfinite(recent)):
        return None
    r10d = float(np.sum(recent[-10:]))
    vol20 = float(np.std(recent[-vol_window:], ddof=1) * math.sqrt(252))
    # ADX proxy: 14-day price range / current close (matches trainer)
    if spy_df is not None and "high" in spy_df.columns and "low" in spy_df.columns:
        sub = spy_df.tail(14)
        high = float(sub["high"].max())
        low = float(sub["low"].min())
        c = float(spy_df["close"].iloc[-1])
        adx = (high - low) / c * 100 if c > 0 else 25.0
    else:
        adx = 25.0
    arr = recent[-12:] if len(recent) >= 12 else recent
    ac = float(np.corrcoef(arr[:-1], arr[1:])[0, 1]) if len(arr) > 2 else 0.0
    if not math.isfinite(ac):
        ac = 0.0
    return np.array([r10d, vol20, adx, ac])


def _gaussian_log_pdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """Log-density of multivariate Gaussian at x."""
    d = x.shape[0]
    diff = x - mu
    try:
        _sign, logdet = np.linalg.slogdet(cov)
        inv = np.linalg.inv(cov)
        mahal = float(diff @ inv @ diff)
        return -0.5 * (d * math.log(2 * math.pi) + logdet + mahal)
    except Exception:
        return -1e10


def hmm_predict(
    artifact: dict,
    spy_returns: np.ndarray,
    spy_df: pd.DataFrame | None,
    vol_window: int = 20,
    *,
    history_n_bars: int = 252,
) -> dict[str, float]:
    """Forward-filtered HMM posterior P(regime_t | obs_1..t).

    For Markov-switching to add value over stateless GMM, predictions
    must depend on the SEQUENCE of past observations, not just the
    latest one. We run the forward algorithm on the last
    ``history_n_bars`` of features to estimate P(regime_today | history).

    Causal: only uses spy_returns / spy_df data ≤ today. No lookahead.

    Returns: dict {regime_label: posterior_prob} with same keys the
    legacy GMM produced (BEAR / BULL_CALM / BULL_VOLATILE if cluster
    labels include those).
    """
    if not is_hmm_artifact(artifact):
        log.warning("hmm_predict called with non-HMM artifact; returning uniform")
        labels = artifact.get("cluster_labels", []) if isinstance(artifact, dict) else []
        if not labels:
            return {}
        u = 1.0 / len(labels)
        return {l: u for l in labels}

    if len(spy_returns) < vol_window + 10:
        labels = artifact["cluster_labels"]
        u = 1.0 / len(labels)
        return {l: u for l in labels}

    means = np.array(artifact["means"])
    covs = np.array(artifact["covariances"])
    transmat = np.array(artifact["transition_matrix"])
    start_prob = np.array(artifact["start_prob"])
    scaler_mean = np.array(artifact["scaler_mean"])
    scaler_scale = np.array(artifact["scaler_scale"])
    scaler_scale = np.where(scaler_scale > 0, scaler_scale, 1.0)
    labels = artifact["cluster_labels"]
    n_states = len(labels)

    # Build feature history (last history_n_bars bars)
    # We rebuild each bar's features from spy_returns + spy_df slice
    n_bars_avail = len(spy_returns) - vol_window
    n_use = min(history_n_bars, n_bars_avail)
    if n_use < 10:
        u = 1.0 / n_states
        return {l: u for l in labels}

    feats: list[np.ndarray] = []
    for k in range(n_use):
        # Position from-end: latest = -1; earliest = -n_use
        end_idx = len(spy_returns) - (n_use - 1 - k)
        rets = spy_returns[:end_idx]
        df_slice = spy_df.iloc[:end_idx] if spy_df is not None else None
        fv = _build_feature_vector(rets, df_slice, vol_window=vol_window)
        if fv is None:
            continue
        feats.append((fv - scaler_mean) / scaler_scale)

    if len(feats) == 0:
        u = 1.0 / n_states
        return {l: u for l in labels}

    # Forward algorithm with log-probabilities for stability
    log_alpha = np.full((len(feats), n_states), -np.inf)
    # Initial step
    for s in range(n_states):
        emit = _gaussian_log_pdf(feats[0], means[s], covs[s])
        log_alpha[0, s] = math.log(max(start_prob[s], 1e-12)) + emit
    # Normalize initial step to prevent underflow drift
    log_alpha[0] -= log_alpha[0].max()

    log_trans = np.log(np.maximum(transmat, 1e-12))
    for t in range(1, len(feats)):
        for s in range(n_states):
            # log_alpha[t, s] = logsumexp_{s'} (log_alpha[t-1, s'] + log_trans[s', s]) + emit
            ll = log_alpha[t - 1] + log_trans[:, s]
            m = ll.max()
            log_sum = m + math.log(np.exp(ll - m).sum())
            emit = _gaussian_log_pdf(feats[t], means[s], covs[s])
            log_alpha[t, s] = log_sum + emit
        # Renormalize each step
        log_alpha[t] -= log_alpha[t].max()

    # Filtered posterior at final step
    probs = np.exp(log_alpha[-1])
    probs = probs / probs.sum()
    return {label: float(p) for label, p in zip(labels, probs)}
