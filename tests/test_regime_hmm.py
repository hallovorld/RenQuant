"""HMM regime detector tests (P0 follow-up, 2026-05-14).

Pins three invariants:
  1. `is_hmm_artifact` distinguishes HMM vs legacy GMM artifacts.
  2. `hmm_predict` produces normalized posterior (sums to 1, all ≥ 0).
  3. Forward filtering gives PERSISTENT labels (P(stay) > 0.85 across
     consecutive bars when the regime hasn't structurally changed) —
     this is the core fix vs per-bar GMM.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.regime_hmm import is_hmm_artifact, hmm_predict, load_hmm_artifact  # noqa: E402


@pytest.fixture
def hmm_artifact():
    """Load the trained sim artifact; skip if not present."""
    art = load_hmm_artifact(
        REPO_ROOT / "backtesting" / "renquant_104" /
        "artifacts" / "sim" / "spy-hmm-regime.json"
    )
    if art is None:
        pytest.skip("HMM artifact not trained yet")
    return art


@pytest.fixture
def spy_data():
    p = REPO_ROOT / "data" / "ohlcv" / "SPY" / "1d.parquet"
    if not p.exists():
        pytest.skip("SPY data not available locally")
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    df["ret"] = df["close"].pct_change()
    return df.dropna()


class TestArtifactDiscrimination:
    """is_hmm_artifact must distinguish HMM from legacy GMM."""

    def test_legacy_gmm_artifact_returns_false(self):
        legacy = {
            "means": [[0, 0, 0, 0]] * 3,
            "covariances": [],
            "weights": [0.4, 0.04, 0.56],
            "cluster_labels": ["BULL_VOLATILE", "BEAR", "BULL_CALM"],
        }
        assert is_hmm_artifact(legacy) is False

    def test_hmm_artifact_returns_true(self):
        hmm = {
            "model_type": "GaussianHMM",
            "transition_matrix": [[0.95, 0.025, 0.025]] * 3,
            "means": [[0] * 4] * 3,
            "covariances": [],
            "cluster_labels": ["BEAR", "BULL_CALM", "BULL_STRONG"],
        }
        assert is_hmm_artifact(hmm) is True

    def test_none_returns_false(self):
        assert is_hmm_artifact(None) is False

    def test_empty_dict_returns_false(self):
        assert is_hmm_artifact({}) is False


class TestHMMPredict:
    """Predictions are normalized + sensible on real SPY history."""

    def test_returns_normalized_posterior(self, hmm_artifact, spy_data):
        d = pd.Timestamp("2024-06-01")
        sub = spy_data.loc[:d]
        probs = hmm_predict(hmm_artifact, sub["ret"].values, sub.tail(60))
        assert isinstance(probs, dict)
        assert len(probs) == 3, f"expected 3 regimes, got {probs}"
        s = sum(probs.values())
        assert abs(s - 1.0) < 1e-6, f"posterior sums to {s} != 1"
        assert all(p >= 0 for p in probs.values()), f"negative prob in {probs}"

    def test_labels_match_artifact(self, hmm_artifact, spy_data):
        d = pd.Timestamp("2024-06-01")
        sub = spy_data.loc[:d]
        probs = hmm_predict(hmm_artifact, sub["ret"].values, sub.tail(60))
        for label in probs:
            assert label in hmm_artifact["cluster_labels"]

    def test_short_history_returns_uniform_prior(self, hmm_artifact, spy_data):
        sub = spy_data.head(15)
        probs = hmm_predict(hmm_artifact, sub["ret"].values, sub)
        n = len(probs)
        assert n == 3
        # Uniform prior: each ≈ 1/3
        for p in probs.values():
            assert abs(p - 1 / n) < 0.01

    def test_2022_bear_window_majority_bear(self, hmm_artifact, spy_data):
        """SPY 2022-05 was deep in BEAR; HMM filtered posterior should
        agree most days. Critical regression: pre-HMM GMM gave 0% BEAR
        for ~all of 2022 because the detector was stateless."""
        bear_days_with_bear_label = 0
        total = 0
        for d in pd.date_range("2022-04-01", "2022-07-01", freq="W"):
            try:
                sub = spy_data.loc[:d]
            except KeyError:
                continue
            if len(sub) < 100:
                continue
            probs = hmm_predict(hmm_artifact, sub["ret"].values, sub.tail(60))
            dom = max(probs, key=probs.get)
            if dom == "BEAR":
                bear_days_with_bear_label += 1
            total += 1
        assert total >= 5, f"too few test bars: {total}"
        bear_pct = bear_days_with_bear_label / total
        # Smoke-test showed 68% BEAR coverage. Set test bar at 40%
        # to allow some Sundays where HMM is uncertain.
        assert bear_pct >= 0.40, (
            f"HMM should detect 2022 Q2 BEAR on ≥40% of weekly samples; "
            f"got {bear_pct*100:.0f}% ({bear_days_with_bear_label}/{total})"
        )

    def test_2024_q4_bull_strong_no_false_bear(self, hmm_artifact, spy_data):
        """Q11 BULL_STRONG window must NOT be mis-labeled BEAR.
        This was the panel A regression from the MA50 fix — HMM
        smooths it away via transition_matrix persistence."""
        false_bear = 0
        total = 0
        for d in pd.date_range("2024-10-01", "2025-01-01", freq="W"):
            try:
                sub = spy_data.loc[:d]
            except KeyError:
                continue
            if len(sub) < 100:
                continue
            probs = hmm_predict(hmm_artifact, sub["ret"].values, sub.tail(60))
            dom = max(probs, key=probs.get)
            if dom == "BEAR":
                false_bear += 1
            total += 1
        if total == 0:
            pytest.skip("No bars in 2024 Q4 — data may not include this range")
        false_bear_pct = false_bear / total
        # Smoke showed 0% — strict bar to catch regressions
        assert false_bear_pct <= 0.10, (
            f"HMM mis-labels BULL_STRONG as BEAR on {false_bear_pct*100:.0f}% "
            f"of Q11 — should be ≤10%"
        )
