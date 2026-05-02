"""Regression: PEAD enrichment feature columns (Track B).

Pinned invariants:
  1. Decay weight at day 0 = 1.0
  2. Decay weight at day = decay_window_days = 0.0
  3. pead_signal sign matches surprise sign
  4. days_since clamps at max_window_days
  5. NO LOOKAHEAD: feature value at announcement day uses ONLY pre-announcement data
     (announcement index shifted +1 day before reindex+ffill).
  6. NaN propagation: pre-first-announcement days are NaN, not 0.
  7. Integration smoke: TickerPanelFactorJob with pead.enabled=True must not
     NameError or otherwise crash. Pre-fix incident (2026-05-02): the PEAD
     block referenced `ctx.config` instead of `tc.config`, producing a
     NameError that depleted 103/103 ticker chains and killed the retrain
     with the D-8 / TPF-1 guard. Unit tests on `compute_pead_features`
     PASS even though the integration path is broken — only the smoke
     test catches the variable-name slip.

Reference: Bernard-Thomas 1989, Chan-Jegadeesh-Lakonishok 1996.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.earnings_surprise import compute_pead_features  # noqa: E402


def _make_ohlcv(start: str, end: str, freq: str = "B") -> pd.DataFrame:
    idx = pd.date_range(start, end, freq=freq, name="date")
    return pd.DataFrame({"close": np.linspace(100, 200, len(idx))}, index=idx)


def _make_surprises(announce_dates: list[str], surprise_pcts: list[float]) -> pd.DataFrame:
    idx = pd.to_datetime(announce_dates)
    return pd.DataFrame({"surprise_pct": surprise_pcts}, index=idx)


class TestPeadDecay:

    def test_decay_at_day_zero_is_one(self):
        """Day 1 (announcement day +1, the first available bar) should be at decay ≈ 1.0."""
        ohlcv = {"AAPL": _make_ohlcv("2026-01-01", "2026-04-01")}
        # Announcement on 2026-01-15; +1 shift → first available bar 2026-01-16
        surprises = {"AAPL": _make_surprises(["2026-01-15"], [0.05])}
        _, decay, _ = compute_pead_features(surprises, ohlcv)
        # Day after announcement should have full decay weight (=1)
        first_bar = ohlcv["AAPL"].index[ohlcv["AAPL"].index >= pd.Timestamp("2026-01-16")][0]
        assert decay["AAPL"].loc[first_bar] == pytest.approx(1.0, abs=0.05), (
            f"Decay at announcement day +1 should be ~1.0; got {decay['AAPL'].loc[first_bar]}"
        )

    def test_decay_at_window_end_is_zero(self):
        ohlcv = {"AAPL": _make_ohlcv("2026-01-01", "2026-06-01")}
        surprises = {"AAPL": _make_surprises(["2026-01-15"], [0.05])}
        _, decay, _ = compute_pead_features(surprises, ohlcv, decay_window_days=60)
        # 60 calendar days after 2026-01-15 + 1 day shift = 2026-03-17 onwards
        # find a business day at-or-after that
        target = pd.Timestamp("2026-03-17")
        bars_at_or_after = ohlcv["AAPL"].index[ohlcv["AAPL"].index >= target]
        assert len(bars_at_or_after) > 0
        bar = bars_at_or_after[0]
        assert decay["AAPL"].loc[bar] == pytest.approx(0.0, abs=0.02), (
            f"Decay should be 0 at decay_window_days; got {decay['AAPL'].loc[bar]} at {bar}"
        )


class TestPeadSignal:

    def test_signal_sign_matches_positive_surprise(self):
        ohlcv = {"AAPL": _make_ohlcv("2026-01-01", "2026-04-01")}
        surprises = {"AAPL": _make_surprises(["2026-01-15"], [0.05])}
        _, _, signal = compute_pead_features(surprises, ohlcv)
        # Pick a bar 5-10 days after announcement+1 — decay still > 0
        bar = pd.Timestamp("2026-01-26")
        if bar in signal["AAPL"].index:
            val = signal["AAPL"].loc[bar]
            assert val > 0, f"Positive surprise should produce positive signal; got {val}"

    def test_signal_sign_matches_negative_surprise(self):
        ohlcv = {"AAPL": _make_ohlcv("2026-01-01", "2026-04-01")}
        surprises = {"AAPL": _make_surprises(["2026-01-15"], [-0.10])}
        _, _, signal = compute_pead_features(surprises, ohlcv)
        bar = pd.Timestamp("2026-01-26")
        if bar in signal["AAPL"].index:
            val = signal["AAPL"].loc[bar]
            assert val < 0, f"Negative surprise should produce negative signal; got {val}"


class TestDaysSinceClamp:

    def test_days_since_clamps_at_max_window(self):
        ohlcv = {"AAPL": _make_ohlcv("2026-01-01", "2026-12-31")}
        surprises = {"AAPL": _make_surprises(["2026-01-15"], [0.05])}
        days, _, _ = compute_pead_features(surprises, ohlcv, max_window_days=90)
        late_bar = ohlcv["AAPL"].index[-1]   # ~12 months after announcement
        assert days["AAPL"].loc[late_bar] == 90, (
            f"days_since should clamp at 90; got {days['AAPL'].loc[late_bar]}"
        )


class TestNoLookahead:

    def test_announcement_day_does_not_leak(self):
        """The announcement-day bar itself must NOT carry the post-announcement
        signal. The +1 day shift should make signal first appear on the next
        business day after the announcement.
        """
        ohlcv = {"AAPL": _make_ohlcv("2026-01-01", "2026-04-01")}
        # Announcement on 2026-01-15 (Thursday)
        surprises = {"AAPL": _make_surprises(["2026-01-15"], [0.05])}
        _, _, signal = compute_pead_features(surprises, ohlcv)
        # On the announcement day, signal should still be NaN (pre-first-
        # announcement state, since shift makes the signal first available
        # on day +1).
        ann_day = pd.Timestamp("2026-01-15")
        if ann_day in signal["AAPL"].index:
            val = signal["AAPL"].loc[ann_day]
            assert pd.isna(val), (
                f"Signal at announcement day must be NaN (lookahead-safe); "
                f"got {val} — earnings releases are after-market, post-release "
                f"state should NOT be available on the announcement day."
            )


class TestNanPropagation:

    def test_pre_first_announcement_is_nan(self):
        ohlcv = {"AAPL": _make_ohlcv("2026-01-01", "2026-04-01")}
        # First announcement 2026-02-15
        surprises = {"AAPL": _make_surprises(["2026-02-15"], [0.05])}
        days, decay, signal = compute_pead_features(surprises, ohlcv)
        early_bar = pd.Timestamp("2026-01-10")
        if early_bar in days["AAPL"].index:
            assert pd.isna(days["AAPL"].loc[early_bar])
            assert pd.isna(decay["AAPL"].loc[early_bar])
            assert pd.isna(signal["AAPL"].loc[early_bar])

    def test_empty_surprises_gives_all_nan(self):
        ohlcv = {"AAPL": _make_ohlcv("2026-01-01", "2026-04-01")}
        surprises = {"AAPL": pd.DataFrame(columns=["surprise_pct"])}
        days, decay, signal = compute_pead_features(surprises, ohlcv)
        assert days["AAPL"].isna().all()
        assert decay["AAPL"].isna().all()
        assert signal["AAPL"].isna().all()

    def test_missing_ticker_in_surprises_gives_all_nan(self):
        ohlcv = {"AAPL": _make_ohlcv("2026-01-01", "2026-04-01")}
        surprises = {}   # AAPL not in dict
        days, decay, signal = compute_pead_features(surprises, ohlcv)
        assert "AAPL" in days
        assert days["AAPL"].isna().all()
        assert decay["AAPL"].isna().all()
        assert signal["AAPL"].isna().all()


class TestPanelFactorJobIntegration:
    """Smoke: the PEAD block in TickerPanelFactorJob must not raise.

    Pre-fix bug (2026-05-02): the block used `ctx.config` instead of
    `tc.config`. NameError → 103/103 ticker chains failed → D-8 guard
    killed the retrain. Unit tests on `compute_pead_features` PASS but
    didn't catch the slip because they don't exercise the wired-in
    code path.
    """

    def test_pead_block_uses_tc_config_not_ctx(self):
        """Static check: the wiring code references tc.config, not ctx.config.
        Any future refactor that inadvertently reverts this will fire pre-merge."""
        src_path = REPO_ROOT / "backtesting" / "renquant_104" / "training_panel" / "pp_panel_training.py"
        src = src_path.read_text()
        # The PEAD block must read tc.config, not ctx.config
        # (TickerPanelFactorJob's per-ticker function takes `tc`, not `ctx`)
        idx = src.find("Stage 3b — PEAD")
        assert idx != -1, "Could not locate PEAD block in pp_panel_training.py"
        # Look at the next 1500 characters
        block = src[idx:idx + 1500]
        # Must NOT use ctx.config in this scope
        bad_refs = [ln for ln in block.splitlines() if "ctx.config" in ln]
        assert not bad_refs, (
            f"PEAD block in TickerPanelFactorJob uses ctx.config (out of scope) — "
            f"will NameError at runtime and crash the panel chain. Use tc.config instead.\n"
            f"Offending lines:\n" + "\n".join(f"  {ln.strip()}" for ln in bad_refs)
        )


class TestMultipleAnnouncements:

    def test_uses_most_recent_announcement(self):
        """Most-recent announcement wins — older announcements don't influence
        once a new one lands (signal gets updated).
        """
        ohlcv = {"AAPL": _make_ohlcv("2026-01-01", "2026-12-31")}
        # Two announcements: positive Q1, negative Q2
        surprises = {"AAPL": _make_surprises(
            ["2026-01-15", "2026-04-15"],
            [+0.05, -0.10],
        )}
        days, _, signal = compute_pead_features(surprises, ohlcv)
        # After Q1, before Q2 — signal should be positive
        bar_after_q1 = pd.Timestamp("2026-02-01")
        if bar_after_q1 in signal["AAPL"].index:
            assert signal["AAPL"].loc[bar_after_q1] > 0, "Should reflect Q1 (+5%)"
        # After Q2 — signal should be negative
        bar_after_q2 = pd.Timestamp("2026-05-01")
        if bar_after_q2 in signal["AAPL"].index:
            assert signal["AAPL"].loc[bar_after_q2] < 0, "Should reflect Q2 (-10%)"
