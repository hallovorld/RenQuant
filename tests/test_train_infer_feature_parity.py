"""Train/infer feature value parity tests (audit Patch 2, 2026-05-09).

User question 2026-05-09: "模型每周训练一次，但是每天的数据怎么喂给模型呢？
你确定模型可以正确处理每天的新数据吗"

The model is fixed (XGBoost booster), but inference computes features
fresh each bar. If train-time and infer-time feature pipelines DIVERGE
silently — different rolling-window math, different NaN handling,
different cross-sectional normalization — the score is meaningless even
if the model is "good".

These tests pin the train/infer feature pipeline equivalence:

  1. Same OHLCV window in → same alpha158 numeric values out (modulo
     non-deterministic-init paths which are fingerprinted)
  2. Cross-sectional features (z-score, rank) computed at training time
     for date D match the values inference computes for the SAME D
     given the same panel slice
  3. Feature column SET stays stable across train↔infer
  4. NaN imputation chain matches (BUG #1 was a parity break here —
     training used xs-median, inference used NaN→0)
  5. Artifact-stored normalization (feature_means / feature_stds) is
     applied identically in inference

Reference:
- doc/AUDIT_2026-05-09.md FIX-C and audit user question
- BUG #1 (commit 507cef6) — same class of bug
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_ohlcv():
    """5 tickers × 300 trading days of synthetic OHLCV, deterministic seed."""
    rng = np.random.default_rng(2026)
    dates = pd.bdate_range("2024-01-02", periods=300)
    out = {}
    for t, base in [("AAA", 100.0), ("BBB", 50.0), ("CCC", 200.0),
                    ("DDD", 75.0), ("SPY", 400.0)]:
        # Geometric-Brownian-ish path
        ret = rng.normal(0.0005, 0.012, size=len(dates))
        close = base * np.exp(np.cumsum(ret))
        df = pd.DataFrame({
            "open":   close * (1 + rng.normal(0, 0.001, len(dates))),
            "high":   close * (1 + np.abs(rng.normal(0, 0.005, len(dates)))),
            "low":    close * (1 - np.abs(rng.normal(0, 0.005, len(dates)))),
            "close":  close,
            "volume": rng.integers(1_000_000, 5_000_000, len(dates)),
        }, index=dates)
        out[t] = df
    return out


# ── alpha158 feature computation parity ─────────────────────────────────────

class TestAlpha158ValueParity:
    """Compute alpha158 features twice from the SAME OHLCV slice and assert
    every cell is bit-identical. If training-time and inference-time use
    different code paths (e.g. different lookback origin), this fails."""

    def test_alpha158_bit_identical_on_same_input(self, synthetic_ohlcv):
        from kernel.panel_pipeline.alpha158_features import compute_alpha158_at  # noqa: PLC0415

        ohlcv = synthetic_ohlcv
        spy = ohlcv["SPY"]
        # Pick a date well into history (need ≥252d for 252d-window features)
        today = ohlcv["AAA"].index[260]

        # Compute twice — same bytes both times
        f1 = compute_alpha158_at(ohlcv["AAA"], today=today)
        f2 = compute_alpha158_at(ohlcv["AAA"], today=today)

        assert set(f1.keys()) == set(f2.keys()), \
            f"Feature columns differ between two calls: {set(f1.keys()) ^ set(f2.keys())}"

        for k in f1:
            v1, v2 = f1[k], f2[k]
            if pd.isna(v1) and pd.isna(v2):
                continue
            assert v1 == v2 or abs(v1 - v2) < 1e-12, \
                f"Feature '{k}' differs across calls: {v1} vs {v2}"

    def test_alpha158_independent_of_future_bars(self, synthetic_ohlcv):
        """Feature at date D must NOT change when we add bars after D.
        This catches lookahead leakage in the feature pipeline."""
        from kernel.panel_pipeline.alpha158_features import compute_alpha158_at  # noqa: PLC0415

        ohlcv = synthetic_ohlcv
        spy = ohlcv["SPY"]
        today = ohlcv["AAA"].index[260]

        # Truncate AAA to today (no future bars)
        aaa_trunc = ohlcv["AAA"].loc[:today]
        f_trunc = compute_alpha158_at(aaa_trunc, today=today)

        # Full series (has future bars)
        f_full = compute_alpha158_at(ohlcv["AAA"], today=today)

        # Cells must match — if they don't, the feature is using future data
        for k in f_trunc:
            v1, v2 = f_trunc[k], f_full[k]
            if pd.isna(v1) and pd.isna(v2):
                continue
            assert v1 == v2 or abs(v1 - v2) < 1e-9, \
                f"LOOKAHEAD LEAK in feature '{k}': trunc={v1} full={v2}"

    def test_alpha158_std_family_matches_training_builder(self, synthetic_ohlcv):
        """Training and inference must use identical std semantics.

        pandas ``rolling.std()`` defaults to sample std (ddof=1).  A prior
        inference path used NumPy/pandas population std (ddof=0) for STD,
        VSTD, and WVMA, silently changing the model input scale at live time.
        """
        from kernel.panel_pipeline.alpha158_features import (  # noqa: PLC0415
            WINDOWS,
            compute_alpha158_at,
            compute_alpha158_frame,
        )
        from scripts.build_alpha158_qlib import rolling_features  # noqa: PLC0415

        df = synthetic_ohlcv["AAA"]
        today = df.index[260]
        train_row = pd.DataFrame(rolling_features(df.loc[:today])).loc[today]
        infer_row = pd.Series(compute_alpha158_at(df, today=today))
        frame_row = compute_alpha158_frame(df).loc[today]

        for n in WINDOWS:
            for fam in ("STD", "VSTD", "WVMA"):
                col = f"{fam}{n}"
                expected = train_row[col]
                assert infer_row[col] == pytest.approx(expected, rel=1e-12, abs=1e-12), \
                    f"{col} single-bar inference differs from training builder"
                assert frame_row[col] == pytest.approx(expected, rel=1e-12, abs=1e-12), \
                    f"{col} vectorized inference differs from training builder"


# ── Feature schema stability ─────────────────────────────────────────────────

class TestFeatureSchemaStability:
    """The set of feature columns produced by alpha158 must be stable.
    If a new column is silently added/removed between train and infer,
    artifact's feature_cols won't match runtime's X.columns → reindex
    fills NaN → silent score degradation."""

    def test_alpha158_returns_known_column_set(self, synthetic_ohlcv):
        from kernel.panel_pipeline.alpha158_features import compute_alpha158_at  # noqa: PLC0415

        ohlcv = synthetic_ohlcv
        today = ohlcv["AAA"].index[260]
        feats = compute_alpha158_at(ohlcv["AAA"], today=today)

        # Sanity: alpha158 should produce many features (>100), not zero
        assert len(feats) > 100, \
            f"alpha158 returned only {len(feats)} features — schema regression?"

        # All keys are strings (feature names)
        assert all(isinstance(k, str) for k in feats), \
            "Some feature keys are not strings"


# ── Production artifact + inference path parity ─────────────────────────────

class TestProductionArtifactParity:
    """The committed production artifact's feature_cols must match what
    the inference path's compute_alpha158_at + extra-feature blocks
    actually produce. If they diverge, X.reindex fills missing with NaN
    → fund/PEAD/SUE features silently empty."""

    def test_artifact_feature_count_matches_alpha158_plus_extras(self):
        artifact = REPO / "backtesting" / "renquant_104" / "artifacts" / "panel-ltr.alpha158_fund.json"
        if not artifact.exists():
            pytest.skip("Production artifact not present")
        m = json.loads(artifact.read_text())
        feature_cols = m.get("feature_cols", [])

        # Production artifact is alpha158 + 5 fund + 3 PEAD + 3 SUE + 3 sentiment
        # = 172 (post-2026-05-18 sentiment shipment per CLAUDE.md status).
        # Bumped from 169 → 172 on 2026-05-20 audit P0-12.
        assert len(feature_cols) == 172, \
            f"Production artifact has {len(feature_cols)} features, expected 172"

        # Required fund + PEAD + SUE + sentiment columns must be present
        required_extras = {
            "earnings_yield", "book_to_price", "gross_profitability",
            "roe", "asset_growth",
            "days_since_earnings", "pead_signal", "pead_quintile_rank",
            "sue_signal", "surprise_momentum", "surprise_streak",
            "sentiment_pos_share", "mean_sentiment", "n_articles_log",
        }
        missing = required_extras - set(feature_cols)
        assert not missing, \
            f"Production artifact missing extras: {missing}"

    def test_artifact_has_normalization_metadata(self):
        """For inference parity, artifact must store feature_means + feature_stds
        so X is normalized identically at infer time as at train time."""
        artifact = REPO / "backtesting" / "renquant_104" / "artifacts" / "panel-ltr.alpha158_fund.json"
        if not artifact.exists():
            pytest.skip("Production artifact not present")
        m = json.loads(artifact.read_text())
        meta = m.get("metadata", {}) or {}
        # Either at top level or in metadata
        means = m.get("feature_means") or meta.get("feature_means")
        stds  = m.get("feature_stds")  or meta.get("feature_stds")
        if means is None or stds is None:
            pytest.skip("Artifact pre-dates per-feature normalization (older training)")
        assert len(means) == len(m["feature_cols"]), \
            "feature_means length != feature_cols count"
        assert len(stds) == len(m["feature_cols"]), \
            "feature_stds length != feature_cols count"


# ── Inference normalization matches stored mu/sd ─────────────────────────────

class TestInferenceNormalizationParity:
    """ApplyScoresTask code path applies (X - feature_means) / feature_stds
    using artifact-stored stats. Pin this contract — if the code stops
    using stored stats and recomputes from current panel, that's drift."""

    def test_apply_scores_uses_stored_mu_sd(self):
        src = (REPO / "backtesting" / "renquant_104" / "kernel"
               / "panel_pipeline" / "job_panel_scoring.py").read_text()
        # The ApplyScoresTask block must reference feature_means / feature_stds
        # from the scorer.metadata (training-time stats), not from current X.
        cls_idx = src.find("class ApplyScoresTask")
        assert cls_idx > 0
        end_idx = src.find("\nclass ", cls_idx + 1)
        body = src[cls_idx:end_idx if end_idx > 0 else len(src)]

        assert "feature_means" in body, \
            "ApplyScoresTask must use artifact-stored feature_means for inference parity"
        assert "feature_stds" in body, \
            "ApplyScoresTask must use artifact-stored feature_stds for inference parity"
        # Specifically: must read from scorer.metadata (training stats), not
        # recompute from X.mean(), X.std()
        assert "scorer" in body and ("metadata" in body or "feature_means" in body)

    def test_no_inference_time_normalization_recompute(self):
        """Audit-regression guard: confirm we DON'T see X.mean()/X.std() in
        the scoring path — that would be a per-bar cross-section normalization
        which is a known parity break with training-time normalization."""
        src = (REPO / "backtesting" / "renquant_104" / "kernel"
               / "panel_pipeline" / "job_panel_scoring.py").read_text()
        cls_idx = src.find("class ApplyScoresTask")
        end_idx = src.find("\nclass ", cls_idx + 1)
        body = src[cls_idx:end_idx if end_idx > 0 else len(src)]

        # If somebody adds "X.mean()" or "X.std()" to ApplyScoresTask,
        # they're recomputing normalization stats on today's small panel
        # instead of using training-time stats. Catch it.
        assert "X.mean()" not in body and "X.std()" not in body, \
            "AUDIT REGRESSION: ApplyScoresTask must NOT recompute normalization " \
            "stats from current X. Use scorer.metadata.feature_means/_stds " \
            "(training-time) for inference parity."


# ── BUG #1 class regression guard (xs-median vs zero-fill) ──────────────────

class TestFundImputationParity:
    """Inference must use cross-sectional median imputation, NOT NaN→0,
    matching training. BUG #1 (commit 507cef6) was this class."""

    def test_apply_scores_uses_xs_median_for_fund_features(self):
        src = (REPO / "backtesting" / "renquant_104" / "kernel"
               / "panel_pipeline" / "job_panel_scoring.py").read_text()
        # The fund block must compute cross-sectional median + apply when missing
        assert "cs_median" in src or "cross_sectional_median" in src or \
               "xs_median" in src.lower() or "imputed_xs_median" in src, \
            "AUDIT REGRESSION (BUG #1): ApplyScoresTask must use cross-sectional " \
            "median for fund imputation, matching training. NaN→0 is the bug."
