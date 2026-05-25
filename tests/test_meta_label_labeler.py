"""TDD — label_snapshots() applies triple-barrier across all snapshot rows.

Given:
  * snapshot_df: rows of (date, ticker, ...features..., fwd_*_ret) per P4.1 schema
  * close_paths: dict[ticker] → close Series indexed by date

Produces a copy of snapshot_df with:
  * fwd_5d_ret  — geometric 5-bar fwd return from snapshot date
  * fwd_20d_ret — geometric 20-bar fwd return from snapshot date
  * meta_label  — 1 if exit was correct (continued loss), 0 otherwise.
                  Only computed for rows with any_trigger==1.
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

from kernel.meta_label.labeler import label_snapshots  # noqa: E402
from kernel.meta_label.snapshot import FEATURE_COLUMNS  # noqa: E402


def _snapshot_row(*, date_iso: str, ticker: str, any_trigger: int,
                  trigger_stop_loss: int = 0, **extras) -> dict:
    """Build a minimal valid snapshot row (other features default to NaN/0)."""
    row = {col: float("nan") for col in FEATURE_COLUMNS}
    row["date"] = date_iso
    row["ticker"] = ticker
    row["any_trigger"] = any_trigger
    row["trigger_stop_loss"] = trigger_stop_loss
    row["trigger_trailing_stop"] = 0
    row["trigger_single_day_loss"] = 0
    row["trigger_max_hold"] = 0
    row["realized_vol_20d"] = 0.20    # 20% ann → ~1.26% daily
    for k, v in extras.items():
        row[k] = v
    return row


def _close_series(prices: list[float], start: str = "2025-01-01") -> pd.Series:
    idx = pd.bdate_range(start=start, periods=len(prices))
    return pd.Series(prices, index=idx, name="close")


class TestLabelSnapshotsBasics:

    def test_empty_input_returns_empty_output(self):
        df = pd.DataFrame({col: [] for col in FEATURE_COLUMNS})
        out = label_snapshots(df, close_paths={})
        assert len(out) == 0
        # Schema preserved + extra label columns
        for col in FEATURE_COLUMNS:
            assert col in out.columns
        assert "meta_label" in out.columns

    def test_no_triggers_yields_nan_meta_labels(self):
        # Row has any_trigger=0 → no meta_label assigned
        row = _snapshot_row(date_iso="2025-01-15", ticker="AAPL", any_trigger=0)
        df = pd.DataFrame([row])
        prices = [100.0] * 30
        out = label_snapshots(df, close_paths={"AAPL": _close_series(prices)})
        assert pd.isna(out.iloc[0]["meta_label"])

    def test_any_trigger_without_path_rule_yields_nan_meta_label(self):
        # Regression: old snapshots sometimes marked model/QP exits as
        # any_trigger=1. The runtime veto never sees those exit types, so the
        # labeler must not train on them.
        row = _snapshot_row(date_iso="2025-01-15", ticker="AAPL", any_trigger=1)
        df = pd.DataFrame([row])
        prices = [100.0 * (0.99 ** i) for i in range(30)]
        out = label_snapshots(
            df,
            close_paths={"AAPL": _close_series(prices, start="2025-01-01")},
        )
        assert pd.isna(out.iloc[0]["meta_label"])


class TestLabelSnapshotsTripleBarrierIntegration:

    def test_triggered_row_with_continued_fall_gets_label_1(self):
        # Trigger on 2025-01-01 at price 100. Next 20 bars drop -1%/day
        # → falls to 80% by ~bar 21 → lower barrier (-20% with σ=0.01
        # daily and sl_mult=10) → continued fall → meta_label = 1
        row = _snapshot_row(
            date_iso="2025-01-01", ticker="AAPL", any_trigger=1,
            trigger_stop_loss=1, realized_vol_20d=0.0,  # disables σ scaling
        )
        df = pd.DataFrame([row])
        # Use realized_vol_20d=0 so the test uses default sigma_daily
        # (the labeler should accept a default σ when realized_vol_20d
        # is 0 or NaN)
        prices = [100.0 * (0.99 ** i) for i in range(30)]
        out = label_snapshots(
            df, close_paths={"AAPL": _close_series(prices)},
            pt_mult=10.0, sl_mult=10.0, default_sigma_daily=0.01,
            fwd_window=20,
        )
        assert out.iloc[0]["meta_label"] == 1

    def test_triggered_row_with_recovery_gets_label_0(self):
        # Trigger on 2025-01-01 at price 100. Next bars rise +1%/day
        # → climbs to 110% → upper barrier hit → recovery → meta_label = 0
        row = _snapshot_row(
            date_iso="2025-01-01", ticker="AAPL", any_trigger=1,
            trigger_stop_loss=1, realized_vol_20d=0.0,
        )
        df = pd.DataFrame([row])
        prices = [100.0 * (1.01 ** i) for i in range(30)]
        out = label_snapshots(
            df, close_paths={"AAPL": _close_series(prices)},
            pt_mult=10.0, sl_mult=10.0, default_sigma_daily=0.01,
            fwd_window=20,
        )
        assert out.iloc[0]["meta_label"] == 0

    def test_fwd_returns_populated(self):
        # fwd_5d_ret and fwd_20d_ret should be computed regardless of
        # any_trigger flag
        row = _snapshot_row(date_iso="2025-01-01", ticker="AAPL", any_trigger=0)
        df = pd.DataFrame([row])
        prices = [100.0 * (1.01 ** i) for i in range(30)]
        out = label_snapshots(
            df, close_paths={"AAPL": _close_series(prices)},
            fwd_window=20,
        )
        # 5-day fwd: (1.01)^5 - 1 ≈ +5.10%
        assert out.iloc[0]["fwd_5d_ret"] == pytest.approx(0.01 ** 5 + 0.0510, rel=0.05) \
            or out.iloc[0]["fwd_5d_ret"] == pytest.approx(1.01**5 - 1, abs=1e-4)
        # 20-day: (1.01)^20 - 1 ≈ +22%
        assert out.iloc[0]["fwd_20d_ret"] == pytest.approx(1.01 ** 20 - 1, abs=1e-3)

    def test_missing_ticker_in_close_paths_skips_row(self):
        row = _snapshot_row(date_iso="2025-01-01", ticker="UNKNOWN_TICKER",
                             any_trigger=1, trigger_stop_loss=1)
        df = pd.DataFrame([row])
        out = label_snapshots(df, close_paths={})
        assert pd.isna(out.iloc[0]["meta_label"])

    def test_date_past_series_end_skips_row(self):
        # Snapshot date AFTER any data in close series → can't label
        row = _snapshot_row(date_iso="2099-01-01", ticker="AAPL",
                             any_trigger=1, trigger_stop_loss=1)
        df = pd.DataFrame([row])
        prices = [100.0] * 30
        out = label_snapshots(df, close_paths={"AAPL": _close_series(prices)})
        assert pd.isna(out.iloc[0]["meta_label"])


class TestLabelSnapshotsScale:
    """Realistic-size run: 30 tickers × 200 bars × ~5% triggers."""

    def test_many_rows_labelled_consistently(self):
        rows = []
        close_paths = {}
        idx = pd.bdate_range("2025-01-01", periods=200)
        np.random.seed(42)
        for tkr in [f"T{i:02d}" for i in range(30)]:
            # Random walk with slight upward drift
            ret = np.random.normal(0.0005, 0.015, size=200)
            prices = 100.0 * np.cumprod(1.0 + ret)
            close_paths[tkr] = pd.Series(prices, index=idx, name="close")
            # 10 random trigger snapshots per ticker
            trigger_dates = np.random.choice(idx[:150], size=10, replace=False)
            for d in trigger_dates:
                rows.append(_snapshot_row(
                    date_iso=pd.Timestamp(d).strftime("%Y-%m-%d"),
                    ticker=tkr, any_trigger=1, trigger_stop_loss=1,
                    realized_vol_20d=0.20,
                ))
        df = pd.DataFrame(rows)
        out = label_snapshots(
            df, close_paths=close_paths,
            pt_mult=10.0, sl_mult=10.0, default_sigma_daily=0.01,
            fwd_window=20,
        )
        # Every row should have a meta_label (1 or 0; nothing NaN)
        labels = out["meta_label"].dropna()
        assert len(labels) == len(df) * 1   # all triggered, all labelled
        assert set(labels.unique()).issubset({0, 1})
        # Random walk → roughly balanced classes
        balance = labels.mean()
        assert 0.10 < balance < 0.90, f"class balance off: {balance:.2%}"
