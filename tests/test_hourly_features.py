"""Tests for training_panel/hourly_features.py (Plan G scaffolding).

Synthetic hourly bars are easy to reason about: 7 bars/session for the
standard US equity day, or fewer to test edge cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from training_panel.hourly_features import (  # noqa: E402
    HOURLY_FEATURE_COLS,
    compute_hourly_features,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_session(date: str, opens: list[float], highs: list[float] | None = None,
                   lows: list[float] | None = None, closes: list[float] | None = None,
                   volumes: list[float] | None = None) -> pd.DataFrame:
    """Build one session's hourly bars. Defaults fabricate plausible OHLC."""
    n = len(opens)
    highs    = highs    or [o * 1.01 for o in opens]
    lows     = lows     or [o * 0.99 for o in opens]
    closes   = closes   or [o * 1.002 for o in opens]
    volumes  = volumes  or [1_000_000] * n
    ts = pd.date_range(f"{date} 09:30", periods=n, freq="1h")
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    }, index=ts)


def _concat_sessions(*sessions: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(sessions).sort_index()


# ── Shape / contract tests ───────────────────────────────────────────────────

class TestContract:
    def test_empty_input_returns_empty_frame(self):
        out = compute_hourly_features(pd.DataFrame())
        assert out.empty
        assert list(out.columns) == HOURLY_FEATURE_COLS

    def test_missing_columns_raise(self):
        bad = pd.DataFrame({"open": [1.0], "close": [1.0]}, index=pd.to_datetime(["2024-01-02 10:00"]))
        with pytest.raises(KeyError, match="open|high|low|close|volume"):
            compute_hourly_features(bad)

    def test_single_session_produces_one_row(self):
        sess = _make_session("2024-01-02", opens=[100, 101, 102, 103, 104, 105, 106])
        out = compute_hourly_features(sess)
        assert len(out) == 1
        assert list(out.columns) == HOURLY_FEATURE_COLS

    def test_single_bar_session_dropped(self):
        """Session with 1 hourly bar → insufficient data for drift."""
        sess = _make_session("2024-01-02", opens=[100])
        out = compute_hourly_features(sess)
        # overnight_gap might still be computable, but intraday features
        # all NaN and overnight has no prior session either → dropped by how="all".
        assert out.empty

    def test_output_index_is_date(self):
        sess = _make_session("2024-01-02", opens=[100, 101, 102])
        out = compute_hourly_features(sess)
        assert out.index.name == "date"
        assert out.index.inferred_type in ("datetime64", "datetime64[ns]")


# ── Feature-correctness tests ────────────────────────────────────────────────

class TestMorningAfternoonDrift:
    def test_rising_morning_positive_afternoon_flat(self):
        # open=100, hr1_close=105 → +5% morning. close=105 → 0% afternoon.
        sess = _make_session(
            "2024-01-02",
            opens =[100] * 7,
            closes=[105, 105, 105, 105, 105, 105, 105],
        )
        out = compute_hourly_features(sess).iloc[0]
        assert out["morning_drift"] == pytest.approx(0.05)
        assert out["afternoon_drift"] == pytest.approx(0.0)

    def test_flat_morning_falling_afternoon(self):
        # open=100, hr1_close=100 → 0. Afternoon: 100→95 = -5%.
        sess = _make_session(
            "2024-01-02",
            opens =[100] * 7,
            closes=[100, 99, 98, 97, 96, 95, 95],
        )
        out = compute_hourly_features(sess).iloc[0]
        assert out["morning_drift"] == pytest.approx(0.0, abs=1e-9)
        assert out["afternoon_drift"] == pytest.approx(-0.05)


class TestVwapPremium:
    def test_close_at_vwap_means_zero_premium(self):
        # All bars identical → VWAP = price = close → premium = 0.
        sess = _make_session("2024-01-02", opens=[100] * 7,
                              closes=[100] * 7, highs=[100] * 7, lows=[100] * 7)
        out = compute_hourly_features(sess).iloc[0]
        assert out["vwap_premium"] == pytest.approx(0.0)

    def test_close_above_vwap_positive_premium(self):
        # First bars cheap, last bars expensive → close > VWAP.
        sess = _make_session(
            "2024-01-02",
            opens=[100, 100, 100, 100, 100, 100, 100],
            highs=[100, 100, 100, 100, 100, 100, 110],
            lows =[100, 100, 100, 100, 100, 100, 110],
            closes=[100, 100, 100, 100, 100, 100, 110],
        )
        out = compute_hourly_features(sess).iloc[0]
        assert out["vwap_premium"] > 0


class TestVolumeRatio:
    def test_last_hour_double_volume(self):
        sess = _make_session(
            "2024-01-02", opens=[100] * 7,
            volumes=[1_000, 800, 600, 500, 700, 1_500, 2_000],
        )
        out = compute_hourly_features(sess).iloc[0]
        assert out["vol_ratio"] == pytest.approx(2.0)  # 2000 / 1000

    def test_zero_first_hour_volume_is_nan(self):
        sess = _make_session(
            "2024-01-02", opens=[100] * 7,
            volumes=[0, 100, 100, 100, 100, 100, 100],
        )
        out = compute_hourly_features(sess).iloc[0]
        assert pd.isna(out["vol_ratio"])


class TestIntradayVol:
    def test_flat_session_zero_vol(self):
        sess = _make_session(
            "2024-01-02", opens=[100] * 7,
            closes=[100] * 7,
        )
        out = compute_hourly_features(sess).iloc[0]
        assert out["intraday_realized_vol"] == pytest.approx(0.0, abs=1e-9)

    def test_volatile_session_positive(self):
        sess = _make_session(
            "2024-01-02", opens=[100] * 7,
            closes=[100, 105, 95, 110, 90, 120, 100],
        )
        out = compute_hourly_features(sess).iloc[0]
        assert out["intraday_realized_vol"] > 0.05


class TestOvernightGap:
    def test_gap_up(self):
        s1 = _make_session("2024-01-02", opens=[100] * 7, closes=[100] * 7)
        s2 = _make_session("2024-01-03", opens=[102, 102, 102], closes=[102] * 3)
        out = compute_hourly_features(_concat_sessions(s1, s2))
        # First session has no prior day → NaN.
        assert pd.isna(out["overnight_gap"].iloc[0])
        # Second session: (102 - 100) / 100 = 0.02
        assert out["overnight_gap"].iloc[1] == pytest.approx(0.02)

    def test_gap_down(self):
        s1 = _make_session("2024-01-02", opens=[100] * 3, closes=[100] * 3)
        s2 = _make_session("2024-01-03", opens=[98, 98, 98], closes=[98] * 3)
        out = compute_hourly_features(_concat_sessions(s1, s2))
        assert out["overnight_gap"].iloc[1] == pytest.approx(-0.02)


# ── Multi-session integration ────────────────────────────────────────────────

class TestMultipleSessions:
    def test_three_sessions_produce_three_rows(self):
        s1 = _make_session("2024-01-02", opens=[100] * 7)
        s2 = _make_session("2024-01-03", opens=[101] * 7)
        s3 = _make_session("2024-01-04", opens=[102] * 7)
        out = compute_hourly_features(_concat_sessions(s1, s2, s3))
        assert len(out) == 3
        assert not out[["morning_drift", "afternoon_drift", "vwap_premium",
                        "vol_ratio", "intraday_realized_vol"]].isna().all().any()

    def test_no_nan_in_output_except_first_overnight(self):
        """With clean synthetic sessions, all features should be finite
        except overnight_gap on the first session.
        """
        sessions = [_make_session(f"2024-01-{d:02d}", opens=[100 + d] * 7)
                    for d in (2, 3, 4, 5, 8)]
        out = compute_hourly_features(_concat_sessions(*sessions))
        # overnight_gap should be NaN only on the first session
        assert pd.isna(out["overnight_gap"].iloc[0])
        assert not out["overnight_gap"].iloc[1:].isna().any()
        # Other features should be finite for every session
        for col in ("morning_drift", "afternoon_drift", "vwap_premium",
                    "vol_ratio", "intraday_realized_vol"):
            assert not out[col].isna().any(), f"{col} has NaN in clean synthetic data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
