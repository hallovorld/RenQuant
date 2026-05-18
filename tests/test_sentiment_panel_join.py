"""Regression tests for the 2026-05-18 sentiment-merge in
build_alpha158_fund_panel.py.

Pin two invariants:
  1. Pre-2020 sentiment rows are dropped (Alpaca News API edge case;
     ~13 rows / 91k = 0.014% in 6y backfill — created_at metadata
     occasionally earlier than the --since cutoff).
  2. SENT_COLS = ['sentiment_pos_share', 'mean_sentiment',
     'n_articles_log'] — the regime-stratified IC-survivor set, not
     the full FinBERT output.

Without this regression guard, a future regen could silently let bad
edge-case rows back in (e.g. a 2013 article with extreme score on
n_articles=1 inflating a date that should otherwise have median fill).
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_alpha158_fund_panel",
        REPO / "scripts/build_alpha158_fund_panel.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_alpha158_fund_panel"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestSentimentColumnSet:
    """SENT_COLS pinning — only the regime-stratified IC survivors land
    in the panel. Per 2026-05-18 verdict doc:
      KEEP: sentiment_pos_share, mean_sentiment, n_articles_log
      DROP: sentiment_dispersion (ts-30 placebo eats it)
      DROP: sentiment_neg_share (NULL)
    """

    def test_survivors_only(self):
        mod = _load_builder()
        assert mod.SENT_COLS == ["sentiment_pos_share", "mean_sentiment",
                                  "n_articles_log"]

    def test_n_articles_log_not_raw(self):
        # 2026-05-18 design choice: log-transform compresses heavy
        # right tail (raw max=42 on big-news days; log1p(42)≈3.76).
        mod = _load_builder()
        assert "n_articles_log" in mod.SENT_COLS
        assert "n_articles" not in mod.SENT_COLS  # raw not used


class TestPre2020FilterApplied:
    """Pre-2020 sentiment rows must be dropped during the merge.

    Pin the filter at the SOURCE-CODE level since unit-testing the
    merge with synthetic parquets would require building a mock
    panel + earnings_surprise dir.
    """

    def test_filter_substring_present(self):
        src = (REPO / "scripts/build_alpha158_fund_panel.py").read_text()
        # The filter condition must reject any row with date < 2020-01-01
        assert "pd.Timestamp(\"2020-01-01\")" in src or \
               "pd.Timestamp('2020-01-01')" in src
        # And drop, not just warn
        assert "df = df[df[\"date\"] >= pd.Timestamp(\"2020-01-01\")]" in src \
            or "df = df[df['date'] >= pd.Timestamp('2020-01-01')]" in src

    def test_2026_05_18_marker(self):
        src = (REPO / "scripts/build_alpha158_fund_panel.py").read_text()
        assert "SENTIMENT-PRE2020 noise filter" in src

    def test_drop_count_logged(self):
        # If filter fires, log a single info line so operators can
        # spot growing drift over time.
        src = (REPO / "scripts/build_alpha158_fund_panel.py").read_text()
        assert "dropped %d pre-2020 sentiment rows" in src
