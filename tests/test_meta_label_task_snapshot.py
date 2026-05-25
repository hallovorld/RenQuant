"""TDD step 2 — SnapshotHoldingsTask unit tests.

Task contract:
  Reads ctx.holdings + ctx.prices + ctx.spy_returns + ctx.regime + ...
  Writes one row per held ticker to ctx.snapshot_logger.

Behavior:
  * No-op when ctx.snapshot_logger is None (training mode OFF)
  * No-op when ctx.holdings is empty (nothing to snapshot)
  * Records exactly len(ctx.holdings) rows per call
  * Each row has 'date' and 'ticker' set correctly
  * Computes position-state features from HoldingState (cum_pnl_pct,
    peak_gain_pct, drawdown_from_peak_pct)
  * Computes market features from ctx.spy_returns (slices p5d / p20d / p60d)
  * Maps regime to integer code
  * Detects trigger signals from ctx.exits (per-ticker exit_type lookup)
"""
from __future__ import annotations

import datetime
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.exits import HoldingState, ExitSignal  # noqa: E402
from kernel.meta_label.snapshot import SnapshotLogger  # noqa: E402
from kernel.meta_label.task_snapshot import SnapshotHoldingsTask  # noqa: E402


def _make_holding(*, entry_price=100.0, hwm=110.0, entry_date=None,
                   panel_score=0.5, entry_panel_score=0.6) -> HoldingState:
    return HoldingState(
        entry_price=entry_price,
        entry_date=entry_date or datetime.date(2025, 1, 1),
        high_watermark=hwm,
        panel_score=panel_score,
        entry_panel_score=entry_panel_score,
    )


def _make_ctx(*, holdings: dict, today: datetime.date,
              prices: dict, spy_returns: list,
              regime="BULL_CALM", confidence=0.8,
              hwm=100000.0, portfolio_value=95000.0,
              exits=None,
              snapshot_logger=None):
    return SimpleNamespace(
        today=today,
        holdings=holdings,
        prices=prices,
        spy_returns=spy_returns,
        regime=regime,
        confidence=confidence,
        hwm=hwm,
        portfolio_value=portfolio_value,
        exits=exits or [],
        candidates=[],
        config={},
        snapshot_logger=snapshot_logger,
    )


class TestSnapshotHoldingsTaskGuards:
    def test_noop_when_logger_is_none(self):
        ctx = _make_ctx(
            holdings={"AAPL": _make_holding()},
            today=datetime.date(2025, 1, 15),
            prices={"AAPL": 105.0},
            spy_returns=[0.001] * 80,
            snapshot_logger=None,
        )
        SnapshotHoldingsTask().run(ctx)
        # Nothing to assert on logger; just that it didn't crash.
        # If it ran without a logger it would have AttributeError'd —
        # the test passing means the guard worked.

    def test_noop_when_holdings_empty(self):
        logger = SnapshotLogger()
        ctx = _make_ctx(
            holdings={},
            today=datetime.date(2025, 1, 15),
            prices={},
            spy_returns=[0.001] * 80,
            snapshot_logger=logger,
        )
        SnapshotHoldingsTask().run(ctx)
        assert logger.n_rows() == 0

    def test_noop_when_logger_missing_attr(self):
        # Adapter may not have set ctx.snapshot_logger at all.
        ctx = SimpleNamespace(
            today=datetime.date(2025, 1, 15),
            holdings={"AAPL": _make_holding()},
            prices={"AAPL": 105.0},
            spy_returns=[0.001] * 80,
            regime="BULL_CALM",
            confidence=0.8,
            hwm=100000.0,
            portfolio_value=95000.0,
            exits=[],
            candidates=[],
            config={},
            # Note: no snapshot_logger attr at all
        )
        # Should not raise
        SnapshotHoldingsTask().run(ctx)


class TestSnapshotHoldingsTaskRecording:
    def test_records_one_row_per_holding(self):
        logger = SnapshotLogger()
        ctx = _make_ctx(
            holdings={
                "AAPL": _make_holding(entry_price=100.0, hwm=110.0),
                "MSFT": _make_holding(entry_price=200.0, hwm=220.0),
                "GOOG": _make_holding(entry_price=150.0, hwm=145.0),
            },
            today=datetime.date(2025, 1, 15),
            prices={"AAPL": 105.0, "MSFT": 215.0, "GOOG": 140.0},
            spy_returns=[0.001] * 80,
            snapshot_logger=logger,
        )
        SnapshotHoldingsTask().run(ctx)
        assert logger.n_rows() == 3

    def test_row_has_correct_ticker_and_date(self):
        logger = SnapshotLogger()
        today = datetime.date(2025, 1, 15)
        ctx = _make_ctx(
            holdings={"AAPL": _make_holding()},
            today=today,
            prices={"AAPL": 105.0},
            spy_returns=[0.001] * 80,
            snapshot_logger=logger,
        )
        SnapshotHoldingsTask().run(ctx)
        # Inspect via dump-to-DataFrame
        import pandas as pd
        rows = logger._rows  # noqa: SLF001  - test-only access
        assert rows[0]["ticker"] == "AAPL"
        assert rows[0]["date"] == "2025-01-15"


class TestSnapshotHoldingsTaskFeatures:
    def test_cum_pnl_pct_computed_from_entry_and_price(self):
        logger = SnapshotLogger()
        # entry=100, current=110 → cum_pnl_pct = +10%
        ctx = _make_ctx(
            holdings={"AAPL": _make_holding(entry_price=100.0, hwm=115.0)},
            today=datetime.date(2025, 1, 15),
            prices={"AAPL": 110.0},
            spy_returns=[0.001] * 80,
            snapshot_logger=logger,
        )
        SnapshotHoldingsTask().run(ctx)
        row = logger._rows[0]  # noqa: SLF001
        assert row["cum_pnl_pct"] == pytest.approx(0.10, abs=1e-9)

    def test_peak_gain_pct_uses_hwm(self):
        logger = SnapshotLogger()
        # entry=100, hwm=125 → peak_gain = +25%
        ctx = _make_ctx(
            holdings={"AAPL": _make_holding(entry_price=100.0, hwm=125.0)},
            today=datetime.date(2025, 1, 15),
            prices={"AAPL": 115.0},
            spy_returns=[0.001] * 80,
            snapshot_logger=logger,
        )
        SnapshotHoldingsTask().run(ctx)
        row = logger._rows[0]  # noqa: SLF001
        assert row["peak_gain_pct"] == pytest.approx(0.25, abs=1e-9)

    def test_drawdown_from_peak(self):
        logger = SnapshotLogger()
        # hwm=125, current=115 → DD from peak = (125-115)/125 = 0.08
        ctx = _make_ctx(
            holdings={"AAPL": _make_holding(entry_price=100.0, hwm=125.0)},
            today=datetime.date(2025, 1, 15),
            prices={"AAPL": 115.0},
            spy_returns=[0.001] * 80,
            snapshot_logger=logger,
        )
        SnapshotHoldingsTask().run(ctx)
        row = logger._rows[0]  # noqa: SLF001
        assert row["drawdown_from_peak_pct"] == pytest.approx(0.08, abs=1e-9)

    def test_regime_code_mapping(self):
        logger = SnapshotLogger()
        for regime, expected_code in [
            ("BULL_CALM", 0), ("BULL_VOLATILE", 1),
            ("CHOPPY", 2), ("BEAR", 3),
        ]:
            logger._rows.clear()  # noqa: SLF001
            ctx = _make_ctx(
                holdings={"AAPL": _make_holding()},
                today=datetime.date(2025, 1, 15),
                prices={"AAPL": 105.0},
                spy_returns=[0.001] * 80,
                regime=regime,
                snapshot_logger=logger,
            )
            SnapshotHoldingsTask().run(ctx)
            assert logger._rows[0]["regime_code"] == expected_code  # noqa: SLF001

    def test_trigger_features_from_ctx_exits(self):
        logger = SnapshotLogger()
        # AAPL has a stop_loss firing; MSFT has nothing
        sl_sig = ExitSignal(should_exit=True, reason="sl", exit_type="stop_loss")
        ctx = _make_ctx(
            holdings={
                "AAPL": _make_holding(entry_price=100.0, hwm=110.0),
                "MSFT": _make_holding(entry_price=200.0, hwm=220.0),
            },
            today=datetime.date(2025, 1, 15),
            prices={"AAPL": 80.0, "MSFT": 215.0},
            spy_returns=[0.001] * 80,
            exits=[("AAPL", sl_sig)],
            snapshot_logger=logger,
        )
        SnapshotHoldingsTask().run(ctx)
        rows_by_ticker = {r["ticker"]: r for r in logger._rows}  # noqa: SLF001
        assert rows_by_ticker["AAPL"]["trigger_stop_loss"] == 1
        assert rows_by_ticker["AAPL"]["any_trigger"] == 1
        assert rows_by_ticker["MSFT"]["trigger_stop_loss"] == 0
        assert rows_by_ticker["MSFT"]["any_trigger"] == 0

    def test_model_driven_exits_do_not_become_meta_label_triggers(self):
        logger = SnapshotLogger()
        model_sig = ExitSignal(
            should_exit=True,
            reason="model",
            exit_type="model_sell",
        )
        ctx = _make_ctx(
            holdings={"AAPL": _make_holding(entry_price=100.0, hwm=110.0)},
            today=datetime.date(2025, 1, 15),
            prices={"AAPL": 95.0},
            spy_returns=[0.001] * 80,
            exits=[("AAPL", model_sig)],
            snapshot_logger=logger,
        )
        SnapshotHoldingsTask().run(ctx)
        row = logger._rows[0]  # noqa: SLF001
        assert row["any_trigger"] == 0
        assert row["trigger_stop_loss"] == 0
        assert row["trigger_trailing_stop"] == 0
        assert row["trigger_single_day_loss"] == 0
        assert row["trigger_max_hold"] == 0

    def test_spy_returns_features(self):
        logger = SnapshotLogger()
        # 80 days of constant +0.1% / day. spy_5d_ret = 0.5%, spy_20d_ret = 2%, spy_60d_ret = 6%
        spy = [0.001] * 80
        ctx = _make_ctx(
            holdings={"AAPL": _make_holding()},
            today=datetime.date(2025, 1, 15),
            prices={"AAPL": 105.0},
            spy_returns=spy,
            snapshot_logger=logger,
        )
        SnapshotHoldingsTask().run(ctx)
        row = logger._rows[0]  # noqa: SLF001
        # Use cumulative product for accuracy
        import math
        ret_5  = math.prod(1 + r for r in spy[-5:])  - 1
        ret_20 = math.prod(1 + r for r in spy[-20:]) - 1
        ret_60 = math.prod(1 + r for r in spy[-60:]) - 1
        assert row["spy_5d_ret"]  == pytest.approx(ret_5,  abs=1e-6)
        assert row["spy_20d_ret"] == pytest.approx(ret_20, abs=1e-6)
        assert row["spy_60d_ret"] == pytest.approx(ret_60, abs=1e-6)

    def test_short_spy_returns_dont_crash(self):
        # Sim early bars have <60d spy_returns — must not crash
        logger = SnapshotLogger()
        ctx = _make_ctx(
            holdings={"AAPL": _make_holding()},
            today=datetime.date(2025, 1, 15),
            prices={"AAPL": 105.0},
            spy_returns=[0.001] * 3,  # only 3 days
            snapshot_logger=logger,
        )
        SnapshotHoldingsTask().run(ctx)
        row = logger._rows[0]  # noqa: SLF001
        # Should fill NaN, not crash
        assert math.isnan(row["spy_60d_ret"]) or row["spy_60d_ret"] is None

    def test_nan_inputs_dont_crash(self):
        # NaN price (data outage) → cum_pnl_pct = NaN, not exception
        logger = SnapshotLogger()
        ctx = _make_ctx(
            holdings={"AAPL": _make_holding(entry_price=100.0, hwm=110.0)},
            today=datetime.date(2025, 1, 15),
            prices={"AAPL": float("nan")},
            spy_returns=[0.001] * 80,
            snapshot_logger=logger,
        )
        SnapshotHoldingsTask().run(ctx)
        assert logger.n_rows() == 1
        row = logger._rows[0]  # noqa: SLF001
        assert math.isnan(row["cum_pnl_pct"])
