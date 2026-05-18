"""Tests for FinBERT news sentiment scorer (TDD-first; 2026-05-18).

Roadmap C5 step 2: per-headline → per-ticker-per-date sentiment features.

Reference: HuggingFace ProsusAI/finbert — fine-tuned BERT on financial
news, 3-class output (positive / neutral / negative). Standard quant
finance NLP baseline (Ke-Kelly-Xiu 2019, Garcia 2013).

These tests pin the scorer's behavioral contracts WITHOUT loading the
real 440MB model (unit test speed). Integration tests that load the
model are gated by ``RUN_FINBERT_INTEGRATION=1`` env flag.

Output features (per ticker × date):
  • mean_sentiment       — average per-article signed score in [-1, +1]
  • sentiment_dispersion — std-dev (Garcia 2013 "disagreement" proxy)
  • n_articles           — count of articles that day
  • sentiment_pos_share  — fraction with score > +0.2
  • sentiment_neg_share  — fraction with score < -0.2

Sanity-gate (per CLAUDE.md §5.2): reject the scored output if
  • all scores identical (degenerate), OR
  • all scores at ±1 (saturation bug), OR
  • > 95% of scores at 0 (tokenizer mis-config).
"""
from __future__ import annotations
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def scorer_mod():
    spec = importlib.util.spec_from_file_location(
        "score_news_finbert", REPO / "scripts/score_news_finbert.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_news_finbert"] = module
    spec.loader.exec_module(module)
    return module


# ── Score normalization ──────────────────────────────────────────────────

class TestScoreToSigned:
    """FinBERT outputs 3-class probs (P_pos, P_neu, P_neg). Convert to
    signed scalar in [-1, +1] via (P_pos - P_neg). Pin this contract."""

    def test_pure_positive_returns_plus_one(self, scorer_mod):
        # 100% positive prob
        s = scorer_mod.probs_to_signed(p_pos=1.0, p_neu=0.0, p_neg=0.0)
        assert s == pytest.approx(1.0)

    def test_pure_negative_returns_minus_one(self, scorer_mod):
        s = scorer_mod.probs_to_signed(p_pos=0.0, p_neu=0.0, p_neg=1.0)
        assert s == pytest.approx(-1.0)

    def test_pure_neutral_returns_zero(self, scorer_mod):
        s = scorer_mod.probs_to_signed(p_pos=0.0, p_neu=1.0, p_neg=0.0)
        assert s == pytest.approx(0.0)

    def test_mixed_distribution(self, scorer_mod):
        # 60% pos / 30% neu / 10% neg → +0.5
        s = scorer_mod.probs_to_signed(p_pos=0.6, p_neu=0.3, p_neg=0.1)
        assert s == pytest.approx(0.5)

    def test_output_always_in_range(self, scorer_mod):
        # 100 random probability triplets — all signed scores must be in [-1, +1]
        import numpy as np
        rng = np.random.RandomState(42)
        for _ in range(100):
            raw = rng.rand(3)
            p = raw / raw.sum()
            s = scorer_mod.probs_to_signed(p_pos=p[0], p_neu=p[1], p_neg=p[2])
            assert -1.0 <= s <= 1.0


# ── Per-ticker/date aggregation ──────────────────────────────────────────

class TestAggregateDaily:
    def test_aggregate_basic_features(self, scorer_mod):
        import pandas as pd
        df = pd.DataFrame({
            "symbol": ["AAPL"] * 5,
            "date": ["2025-06-01"] * 3 + ["2025-06-02"] * 2,
            "sentiment": [0.8, 0.5, -0.3, 0.1, -0.7],
        })
        df["date"] = pd.to_datetime(df["date"])
        agg = scorer_mod.aggregate_daily(df)
        # Day 1: mean = (0.8+0.5-0.3)/3 = 0.333; n=3
        row1 = agg[(agg["symbol"] == "AAPL") & (agg["date"] == pd.Timestamp("2025-06-01"))].iloc[0]
        assert row1["mean_sentiment"] == pytest.approx(0.333, abs=0.01)
        assert row1["n_articles"] == 3
        # Dispersion = std (ddof=0 for population, or ddof=1 — pin via test)
        assert row1["sentiment_dispersion"] > 0  # 3 distinct values
        # Day 2: mean = (0.1 - 0.7)/2 = -0.3
        row2 = agg[(agg["symbol"] == "AAPL") & (agg["date"] == pd.Timestamp("2025-06-02"))].iloc[0]
        assert row2["mean_sentiment"] == pytest.approx(-0.3, abs=0.01)
        assert row2["n_articles"] == 2

    def test_pos_neg_share_thresholds(self, scorer_mod):
        import pandas as pd
        df = pd.DataFrame({
            "symbol": ["AAPL"] * 5,
            "date": ["2025-06-01"] * 5,
            "sentiment": [0.8, 0.5, 0.1, -0.5, -0.8],
        })
        df["date"] = pd.to_datetime(df["date"])
        agg = scorer_mod.aggregate_daily(df)
        row = agg.iloc[0]
        # pos = > +0.2: 2 of 5 = 0.4
        # neg = < -0.2: 2 of 5 = 0.4
        assert row["sentiment_pos_share"] == pytest.approx(0.4)
        assert row["sentiment_neg_share"] == pytest.approx(0.4)

    def test_single_article_has_zero_dispersion(self, scorer_mod):
        import pandas as pd
        df = pd.DataFrame({"symbol": ["AAPL"], "date": ["2025-06-01"],
                           "sentiment": [0.5]})
        df["date"] = pd.to_datetime(df["date"])
        agg = scorer_mod.aggregate_daily(df)
        row = agg.iloc[0]
        assert row["sentiment_dispersion"] == 0.0  # std of single sample
        assert row["n_articles"] == 1


# ── Sanity gate ─────────────────────────────────────────────────────────

class TestSanityGate:
    def test_rejects_all_zero_scores(self, scorer_mod):
        import pandas as pd
        df = pd.DataFrame({"sentiment": [0.0] * 100})
        # The gate is a validate() function returning (ok, reason)
        ok, reason = scorer_mod.validate_sanity(df["sentiment"])
        assert not ok
        assert "degenerate" in reason.lower() or "constant" in reason.lower() or "zero" in reason.lower()

    def test_rejects_all_saturated(self, scorer_mod):
        import pandas as pd
        df = pd.DataFrame({"sentiment": [1.0] * 100})
        ok, reason = scorer_mod.validate_sanity(df["sentiment"])
        assert not ok

    def test_accepts_diverse_distribution(self, scorer_mod):
        import pandas as pd, numpy as np
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"sentiment": rng.uniform(-1, 1, 1000)})
        ok, _ = scorer_mod.validate_sanity(df["sentiment"])
        assert ok

    def test_rejects_mostly_zero(self, scorer_mod):
        import pandas as pd, numpy as np
        # 99% zeros + 1% diverse — likely a tokenizer mis-config
        scores = [0.0] * 99 + [0.5]
        ok, reason = scorer_mod.validate_sanity(pd.Series(scores))
        assert not ok


# ── Empty/edge inputs ────────────────────────────────────────────────────

class TestEdgeInputs:
    def test_empty_headline_returns_neutral(self, scorer_mod):
        # Empty string / None should NOT crash; should return 0.0
        # (Tested without loading model — score_headline_text dispatches
        # to fallback for empty inputs.)
        assert scorer_mod.is_skippable_text("") is True
        assert scorer_mod.is_skippable_text(None) is True
        assert scorer_mod.is_skippable_text("   \n\t  ") is True

    def test_non_empty_text_is_scoreable(self, scorer_mod):
        assert scorer_mod.is_skippable_text("Apple beats Q3 earnings") is False


# ── Concurrency contract ─────────────────────────────────────────────────

class TestConcurrencyContract:
    """Per user mandate #3 (concurrency for efficiency).

    The scorer must accept a `batch_size` parameter and process in
    batches (GPU-friendly). Test that the helper exists and respects
    the batch_size param.
    """

    def test_chunk_for_batching(self, scorer_mod):
        items = list(range(100))
        chunks = list(scorer_mod._chunk(items, 32))
        assert len(chunks) == 4  # 32 + 32 + 32 + 4
        assert chunks[0] == list(range(32))
        assert len(chunks[-1]) == 4

    def test_default_batch_size_64(self, scorer_mod):
        # Default 64 = standard FinBERT GPU batch on M2 8GB VRAM
        assert scorer_mod.DEFAULT_BATCH_SIZE == 64


# ── Output schema pin ────────────────────────────────────────────────────

class TestOutputSchema:
    def test_per_ticker_per_date_columns(self, scorer_mod):
        import pandas as pd
        df = pd.DataFrame({
            "symbol": ["AAPL"] * 2,
            "date": pd.to_datetime(["2025-06-01", "2025-06-02"]),
            "sentiment": [0.5, -0.3],
        })
        agg = scorer_mod.aggregate_daily(df)
        expected_cols = {"symbol", "date", "mean_sentiment",
                         "sentiment_dispersion", "n_articles",
                         "sentiment_pos_share", "sentiment_neg_share"}
        assert set(agg.columns) >= expected_cols


# ── Integration (gated by env flag) ──────────────────────────────────────

@pytest.mark.skipif(os.environ.get("RUN_FINBERT_INTEGRATION") != "1",
                    reason="set RUN_FINBERT_INTEGRATION=1 to run "
                           "(loads 440MB model + ~30s)")
class TestFinBertIntegration:
    def test_positive_headline_scores_positive(self, scorer_mod):
        scorer = scorer_mod.FinBertScorer()
        s = scorer.score_text("Apple beats Q3 earnings, raises guidance")
        assert s > 0.2, f"expected positive, got {s:.3f}"

    def test_negative_headline_scores_negative(self, scorer_mod):
        scorer = scorer_mod.FinBertScorer()
        s = scorer.score_text("Apple misses earnings, slashes guidance, layoffs")
        assert s < -0.2, f"expected negative, got {s:.3f}"

    def test_batch_equals_single(self, scorer_mod):
        scorer = scorer_mod.FinBertScorer()
        texts = ["Beats earnings", "Misses earnings", "Stock price unchanged"]
        single = [scorer.score_text(t) for t in texts]
        batched = scorer.score_batch(texts)
        for s, b in zip(single, batched):
            assert abs(s - b) < 0.001
