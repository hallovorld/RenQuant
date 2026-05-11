"""TDD step 1 — SnapshotLogger unit tests.

The SnapshotLogger is the persistent buffer for per-day per-position
feature snapshots used to train the meta-label classifier (López de
Prado AFML ch.20). Owns:
  * a list/dict buffer (records appended per bar across the sim)
  * a column schema (FEATURE_COLUMNS) — pinned so trained model matches
  * a dump_to_parquet() method called on adapter teardown

Tests pin its public contract BEFORE implementation (TDD red phase).
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.meta_label.snapshot import SnapshotLogger, FEATURE_COLUMNS  # noqa: E402


class TestSnapshotLoggerContract:
    """Pin the SnapshotLogger public API."""

    def test_feature_columns_is_nonempty_tuple(self):
        assert isinstance(FEATURE_COLUMNS, tuple)
        assert len(FEATURE_COLUMNS) >= 20  # baseline expectation: 30+ features
        # Must include the row-identity fields (date, ticker)
        assert "date" in FEATURE_COLUMNS
        assert "ticker" in FEATURE_COLUMNS

    def test_feature_columns_contains_position_state_keys(self):
        # Position-state group per the design doc
        for k in ("cum_pnl_pct", "peak_gain_pct", "drawdown_from_peak_pct",
                 "days_held", "consec_underwater_days"):
            assert k in FEATURE_COLUMNS, f"missing position-state feature: {k}"

    def test_feature_columns_contains_market_state_keys(self):
        for k in ("spy_5d_ret", "spy_20d_ret", "spy_realized_vol_20d",
                 "regime_code"):
            assert k in FEATURE_COLUMNS, f"missing market-state feature: {k}"

    def test_feature_columns_contains_signal_keys(self):
        # Was a path-rule firing this bar?
        for k in ("trigger_stop_loss", "trigger_trailing_stop",
                 "trigger_single_day_loss", "any_trigger"):
            assert k in FEATURE_COLUMNS, f"missing signal feature: {k}"


class TestSnapshotLoggerBehavior:
    """Pin behavior — record / dump / empty cases."""

    def test_logger_constructs_with_empty_buffer(self):
        logger = SnapshotLogger()
        assert logger.n_rows() == 0

    def test_record_appends_one_row(self):
        logger = SnapshotLogger()
        row = {col: 0.0 for col in FEATURE_COLUMNS}
        row["ticker"] = "AAPL"
        row["date"]   = "2025-01-15"
        logger.record(row)
        assert logger.n_rows() == 1

    def test_record_appends_many_rows(self):
        logger = SnapshotLogger()
        for i in range(50):
            row = {col: float(i) for col in FEATURE_COLUMNS}
            row["ticker"] = f"T{i}"
            row["date"]   = "2025-01-15"
            logger.record(row)
        assert logger.n_rows() == 50

    def test_record_rejects_unknown_keys(self):
        logger = SnapshotLogger()
        row = {col: 0.0 for col in FEATURE_COLUMNS}
        row["ticker"] = "AAPL"
        row["date"]   = "2025-01-15"
        row["spurious_field"] = 999  # not in FEATURE_COLUMNS
        with pytest.raises(ValueError, match="unknown.*spurious"):
            logger.record(row)

    def test_record_rejects_missing_required_keys(self):
        logger = SnapshotLogger()
        # Drop a required field
        row = {col: 0.0 for col in FEATURE_COLUMNS if col != "ticker"}
        row["date"] = "2025-01-15"
        with pytest.raises(ValueError, match="missing.*ticker"):
            logger.record(row)

    def test_dump_to_parquet_writes_columns_and_rows(self, tmp_path):
        import pandas as pd  # noqa: PLC0415
        logger = SnapshotLogger()
        for i in range(3):
            row = {col: float(i) for col in FEATURE_COLUMNS}
            row["ticker"] = f"T{i}"
            row["date"]   = "2025-01-15"
            logger.record(row)
        out = tmp_path / "snapshots.parquet"
        logger.dump_to_parquet(out)
        assert out.exists()
        df = pd.read_parquet(out)
        assert len(df) == 3
        for col in FEATURE_COLUMNS:
            assert col in df.columns

    def test_dump_to_parquet_empty_buffer_writes_empty_frame(self, tmp_path):
        import pandas as pd  # noqa: PLC0415
        logger = SnapshotLogger()
        out = tmp_path / "snapshots.parquet"
        logger.dump_to_parquet(out)
        assert out.exists()
        df = pd.read_parquet(out)
        assert len(df) == 0
        # Schema should still match so downstream join works
        for col in FEATURE_COLUMNS:
            assert col in df.columns
