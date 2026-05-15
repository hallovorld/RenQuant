"""Train a 3-state Gaussian HMM regime detector on SPY (Hamilton 1989).

Replaces the legacy stateless GMM with temporal-coherent HMM:
  - 3 hidden states: BEAR / BULL_CALM / BULL_STRONG (mapped post-fit
    by sorting cluster centers on the r10d feature)
  - 4 features: r10d, ann_vol20, ADX-proxy, autocorr12 (matches GMM)
  - Train on 5-year SPY history ending at --train-end; OOS for sims
  - Initialization: KMeans means + identity-like transition matrix
    with P(stay)=0.95 prior (Ang-Bekaert 2002 floor)

Output artifact (JSON): means, covariances, transition_matrix,
start_prob, scaler_mean/scale, cluster_labels, feature_order,
training_window, trained_date.

Reference: Hamilton 1989 Econometrica 57(2):357-384;
hmmlearn 0.3.x GaussianHMM (https://github.com/hmmlearn/hmmlearn).
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_features(df: pd.DataFrame) -> np.ndarray:
    """4-feature matrix per bar, matching kernel.regime.gmm_predict input.

    Columns: [r10d, ann_vol20, adx_proxy, autocorr12].
    Rows with NaN (first 20 bars from rolling) are returned but caller masks.
    """
    rets = df["ret"].values
    n = len(rets)
    out = np.full((n, 4), np.nan)
    has_hl = "high" in df.columns and "low" in df.columns
    for i in range(20, n):
        recent = rets[max(0, i - 20) : i + 1]
        r10d = float(np.sum(recent[-10:]))
        vol20 = float(np.std(recent[-20:], ddof=1) * np.sqrt(252))
        if has_hl:
            high = df["high"].iloc[max(0, i - 13) : i + 1].max()
            low = df["low"].iloc[max(0, i - 13) : i + 1].min()
        else:
            high = df["close"].iloc[max(0, i - 13) : i + 1].max()
            low = df["close"].iloc[max(0, i - 13) : i + 1].min()
        c = df["close"].iloc[i]
        adx = (high - low) / c * 100 if c > 0 else 25.0
        arr = recent[-12:] if len(recent) >= 12 else recent
        ac = (
            float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
            if len(arr) > 2
            else 0.0
        )
        out[i] = [r10d, vol20, adx, ac if np.isfinite(ac) else 0.0]
    return out


def fit_hmm(X_train: np.ndarray, *, random_state: int = 42):
    """Fit HMM with KMeans-initialized means + persistent transition prior."""
    from hmmlearn.hmm import GaussianHMM
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=3, random_state=random_state, n_init=10).fit(X_train)
    hmm = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=200,
        random_state=random_state,
        init_params="",  # manual init
        params="stmc",   # learn all params
    )
    hmm.means_ = km.cluster_centers_
    hmm.covars_ = np.array(
        [np.cov(X_train.T) + np.eye(X_train.shape[1]) * 0.01 for _ in range(3)]
    )
    hmm.startprob_ = np.array([1 / 3, 1 / 3, 1 / 3])
    # Ang-Bekaert 2002: P(stay) ≥ 0.85 indicates real regimes;
    # we initialize at 0.95 to avoid converging to per-bar GMM behavior.
    hmm.transmat_ = np.array(
        [[0.95, 0.025, 0.025], [0.025, 0.95, 0.025], [0.025, 0.025, 0.95]]
    )
    hmm.fit(X_train)
    return hmm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spy-data",
                   default=str(REPO_ROOT / "data" / "ohlcv" / "SPY" / "1d.parquet"))
    p.add_argument("--train-end", default="2022-01-01",
                   help="Train on data BEFORE this date (avoid lookahead)")
    p.add_argument("--train-years", type=int, default=10,
                   help="Years of training history")
    p.add_argument("--output", default=str(REPO_ROOT / "backtesting" /
                                          "renquant_104" / "artifacts" /
                                          "sim" / "spy-hmm-regime.json"))
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    spy = pd.read_parquet(args.spy_data)
    spy.index = pd.to_datetime(spy.index)
    spy["ret"] = spy["close"].pct_change()
    spy = spy.dropna(subset=["ret"]).copy()
    print(f"SPY range: {spy.index.min().date()} → {spy.index.max().date()} "
          f"({len(spy)} bars)")

    end = pd.Timestamp(args.train_end)
    start = end - pd.DateOffset(years=args.train_years)
    train_mask = (spy.index >= start) & (spy.index < end)
    print(f"Training window: {start.date()} → {end.date()} "
          f"({train_mask.sum()} bars)")

    feats = make_features(spy)
    valid = ~np.isnan(feats).any(axis=1)
    X = feats[train_mask & valid]
    print(f"Training samples after NaN filter: {len(X)}")

    # Standardize (HMM is scale-sensitive)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    X_s = (X - mu) / sd

    hmm = fit_hmm(X_s, random_state=args.random_state)
    print(f"HMM converged: {hmm.monitor_.converged}, "
          f"iters: {hmm.monitor_.iter}")
    diag = np.diag(hmm.transmat_)
    print(f"Transition diagonal P(stay): {diag.round(3)}")
    assert diag.min() >= 0.80, (
        f"Min P(stay)={diag.min():.3f} < 0.80; regimes are not persistent. "
        f"Per Ang-Bekaert 2002, ≥0.85 is the floor; check feature quality."
    )

    # Map cluster idx → regime label using the EXISTING codebase taxonomy:
    # {BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR}. CHOPPY is the Hurst-layer
    # label (not HMM); HMM emits the other three.
    # Heuristic: sort by r10d (mean) AND vol20 (vol).
    #   lowest r10d  → BEAR
    #   highest r10d + low vol  → BULL_CALM (low-vol uptrend)
    #   highest r10d + high vol → BULL_VOLATILE (high-vol uptrend)
    # Tie-break by vol20 (feature index 1).
    r10d_means = hmm.means_[:, 0]
    vol_means = hmm.means_[:, 1]
    sorted_idx = np.argsort(r10d_means)
    cluster_labels = ["UNK"] * 3
    cluster_labels[sorted_idx[0]] = "BEAR"  # lowest r10d
    # The two upper clusters: lower-vol = BULL_CALM, higher-vol = BULL_VOLATILE
    up1, up2 = sorted_idx[1], sorted_idx[2]
    if vol_means[up1] <= vol_means[up2]:
        cluster_labels[up1] = "BULL_CALM"
        cluster_labels[up2] = "BULL_VOLATILE"
    else:
        cluster_labels[up1] = "BULL_VOLATILE"
        cluster_labels[up2] = "BULL_CALM"
    print(f"Cluster r10d means (standardized): {r10d_means.round(3)}")
    print(f"Cluster → label mapping: "
          f"{ {i: cluster_labels[i] for i in range(3)} }")

    artifact = {
        "model_type": "GaussianHMM",
        "n_components": 3,
        "feature_order": ["r10d", "ann_vol20", "adx_proxy", "autocorr12"],
        "scaler_mean": mu.tolist(),
        "scaler_scale": sd.tolist(),
        "means": hmm.means_.tolist(),
        "covariances": [c.tolist() for c in hmm.covars_],
        "transition_matrix": hmm.transmat_.tolist(),
        "start_prob": hmm.startprob_.tolist(),
        "cluster_labels": cluster_labels,
        "training_window": [str(start.date()), str(end.date())],
        "training_n_samples": int(len(X)),
        "trained_date": datetime.now(timezone.utc).isoformat(),
        "hmm_converged": bool(hmm.monitor_.converged),
        "hmm_iters": int(hmm.monitor_.iter),
        "p_stay_min": float(diag.min()),
        "random_state": args.random_state,
        # Documentation of the reference
        "_reference": (
            "Hamilton 1989 Econometrica 57(2):357 + hmmlearn 0.3.x "
            "GaussianHMM. Replaces legacy spy-gmm-regime.json (stateless) "
            "with HMM (Viterbi-smoothed via transition_matrix)."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))
    print(f"\nWrote artifact → {out}")
    print(f"Size: {out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
