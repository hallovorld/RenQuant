"""Tests for Plan G step 2 — hourly features wired into PanelDataJob /
TickerPanelFactorJob / FactorZScoreTask.

Covers:
  - HourlyBarStore round-trip (save/load parquet)
  - LoadHourlyBarsTask skips when flag off / populates ctx when on
  - TickerPanelFactorJob merges the 6 hourly columns into raw_factor_frame
  - FactorZScoreTask emits `{col}_z` for each hourly column
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

from kernel.intraday import HourlyBarStore  # noqa: E402
from training_panel.context import PanelTrainingContext, TickerPanelContext  # noqa: E402
from training_panel.hourly_features import HOURLY_FEATURE_COLS  # noqa: E402
from training_panel.pp_panel_training import (  # noqa: E402
    FactorZScoreTask,
    LoadHourlyBarsTask,
    TickerPanelFactorJob,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _daily_ohlcv(start: str, days: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=days, freq="B")  # business days
    rng = np.arange(days, dtype=float)
    return pd.DataFrame({
        "open":  100 + rng * 0.1,
        "high":  101 + rng * 0.1,
        "low":    99 + rng * 0.1,
        "close": 100 + rng * 0.1 + 0.5,
        "volume": 1_000_000 + rng,
    }, index=idx)


def _hourly_bars(start: str, days: int, bars_per_session: int = 7) -> pd.DataFrame:
    """Fabricate hourly OHLCV across N sessions."""
    frames: list[pd.DataFrame] = []
    base_open = 100.0
    for d, bdate in enumerate(pd.bdate_range(start, periods=days)):
        session_ts = pd.date_range(f"{bdate.date()} 09:30",
                                    periods=bars_per_session, freq="1h")
        opens  = [base_open + d + i * 0.5  for i in range(bars_per_session)]
        closes = [o + 0.25                  for o in opens]
        highs  = [o + 0.6                   for o in opens]
        lows   = [o - 0.4                   for o in opens]
        vols   = [1_000_000 + i * 50_000    for i in range(bars_per_session)]
        frames.append(pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": vols,
        }, index=session_ts))
    return pd.concat(frames)


# ── HourlyBarStore ──────────────────────────────────────────────────────────

class TestHourlyBarStore:
    def test_empty_cache_returns_none(self, tmp_path):
        store = HourlyBarStore(data_dir=tmp_path)
        assert store.load("AAPL") is None

    def test_save_then_load_round_trip(self, tmp_path):
        store = HourlyBarStore(data_dir=tmp_path)
        df = _hourly_bars("2024-01-02", days=3)
        p = store.save(df, "AAPL")
        assert p.exists()
        loaded = store.load("AAPL")
        assert loaded is not None
        assert len(loaded) == len(df)
        assert list(loaded.columns) == list(df.columns)

    def test_save_merges_and_dedups(self, tmp_path):
        store = HourlyBarStore(data_dir=tmp_path)
        a = _hourly_bars("2024-01-02", days=2)
        b = _hourly_bars("2024-01-03", days=2)  # overlaps with `a` on 2024-01-03
        store.save(a, "AAPL")
        store.save(b, "AAPL")
        loaded = store.load("AAPL")
        # Dedup keeps last; sorted timestamps strictly increase.
        assert loaded.index.is_monotonic_increasing
        assert not loaded.index.duplicated().any()


# ── LoadHourlyBarsTask ──────────────────────────────────────────────────────

class TestLoadHourlyBarsTask:
    def test_skip_when_flag_off(self, tmp_path):
        ctx = PanelTrainingContext(
            config={"panel_ltr": {"hourly": {"enabled": False,
                                              "cache_dir": str(tmp_path)}}},
            watchlist=["AAPL"],
        )
        LoadHourlyBarsTask().run(ctx)
        assert ctx.hourly_bars == {}

    def test_loads_from_cache_when_enabled(self, tmp_path):
        store = HourlyBarStore(data_dir=tmp_path)
        store.save(_hourly_bars("2024-01-02", days=3), "AAPL")
        ctx = PanelTrainingContext(
            config={"panel_ltr": {"hourly": {"enabled": True,
                                              "cache_dir": str(tmp_path)}}},
            watchlist=["AAPL", "MSFT"],
        )
        LoadHourlyBarsTask().run(ctx)
        assert set(ctx.hourly_bars.keys()) == {"AAPL"}  # MSFT not in cache
        assert not ctx.hourly_bars["AAPL"].empty


# ── TickerPanelFactorJob + FactorZScoreTask integration ─────────────────────

class TestFactorJobWithHourly:
    def _tc(self, hourly: dict[str, pd.DataFrame]) -> TickerPanelContext:
        aapl_daily = _daily_ohlcv("2024-01-02", 30)
        spy_daily  = _daily_ohlcv("2024-01-02", 30)
        return TickerPanelContext(
            ticker="AAPL",
            ohlcv={"AAPL": aapl_daily, "SPY": spy_daily},
            sector_momentum={},
            ticker_sectors={"AAPL": "Tech"},
            config={"panel_ltr": {"factor_mom_window": 10, "factor_skip": 2,
                                   "beta_window": 10}},
            hourly_bars=hourly,
        )

    def test_factor_job_merges_hourly_cols(self):
        hourly = {"AAPL": _hourly_bars("2024-01-02", days=20)}
        tc = self._tc(hourly)
        TickerPanelFactorJob().run(tc)
        assert tc.raw_factor_frame is not None
        for col in HOURLY_FEATURE_COLS:
            assert col in tc.raw_factor_frame.columns, f"missing {col}"
        # At least one non-NaN row per column.
        for col in HOURLY_FEATURE_COLS:
            non_nan = tc.raw_factor_frame[col].notna().sum()
            assert non_nan > 0, f"{col} is all NaN"

    def test_factor_job_no_hourly_when_unwired(self):
        tc = self._tc(hourly={})
        TickerPanelFactorJob().run(tc)
        assert tc.raw_factor_frame is not None
        for col in HOURLY_FEATURE_COLS:
            assert col not in tc.raw_factor_frame.columns

    def test_zscore_task_emits_z_cols_for_hourly(self):
        # Two tickers so cross-sectional z-score is meaningful.
        hourly_a = _hourly_bars("2024-01-02", days=20)
        hourly_b = _hourly_bars("2024-01-02", days=20)
        # Perturb ticker B so its hourly feats differ → non-zero z-scores.
        hourly_b = hourly_b * 1.05
        hourly_b.index = hourly_a.index  # scalar op preserved index anyway

        ctx = PanelTrainingContext(
            config={"panel_ltr": {"factor_mom_window": 10, "factor_skip": 2,
                                   "beta_window": 10}},
            watchlist=["AAPL", "MSFT"],
            ticker_sectors={"AAPL": "Tech", "MSFT": "Tech"},
        )
        daily_a = _daily_ohlcv("2024-01-02", 30)
        daily_b = _daily_ohlcv("2024-01-02", 30) * 1.05
        daily_b.index = daily_a.index
        ctx.ohlcv = {"AAPL": daily_a, "MSFT": daily_b,
                      "SPY": _daily_ohlcv("2024-01-02", 30)}
        ctx.hourly_bars = {"AAPL": hourly_a, "MSFT": hourly_b}

        # Build ticker-level raw_factor_frames via the factor job.
        raw_frames: dict[str, pd.DataFrame] = {}
        for t in ("AAPL", "MSFT"):
            tc = TickerPanelContext(
                ticker=t, ohlcv=ctx.ohlcv, sector_momentum={},
                ticker_sectors=ctx.ticker_sectors, config=ctx.config,
                hourly_bars=ctx.hourly_bars,
            )
            TickerPanelFactorJob().run(tc)
            if tc.raw_factor_frame is not None:
                raw_frames[t] = tc.raw_factor_frame
        ctx.raw_factor_frames = raw_frames

        FactorZScoreTask().run(ctx)
        assert ctx.factor_frames
        first = next(iter(ctx.factor_frames.values()))
        for col in HOURLY_FEATURE_COLS:
            assert f"{col}_z" in first.columns, f"{col}_z missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
