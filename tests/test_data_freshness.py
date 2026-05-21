"""Regression tests for the 2026-05-03 stale-data P0.

Before fix:
- ``LocalStore.has_range(symbol, start=None, end=None)`` returned True when
  cache stopped at Thursday close while wall-clock was Sunday — the
  ``tolerance_days=5`` legacy default plus the ``end=None`` short-circuit
  combined to silently accept a 3-trading-day-stale cache. Result: panel
  pipeline never refetched, model trained + inferred on Thursday close,
  6 live orders submitted to Alpaca on stale data Sunday evening.

Invariant: ``has_range`` and the inference data-freshness gate must REFUSE
any market data older than the last completed NYSE session as of the
reference timestamp. Before today's close, yesterday is enough; after
today's close, today's bar is required.
"""
from __future__ import annotations

import datetime as _dt
import sys
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.data import LocalStore  # noqa: E402


def _make_df(dates: list[str]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    return pd.DataFrame(
        {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1000},
        index=idx,
    )


class TestHasRangeNYSEFreshness(unittest.TestCase):
    """has_range must reject cache that doesn't include the last completed NYSE close."""

    def setUp(self) -> None:
        # tmp dir per test
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.store = LocalStore(data_dir=Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _save(self, sym: str, dates: list[str]) -> None:
        path = self.store._path(sym)  # noqa: SLF001
        path.parent.mkdir(parents=True, exist_ok=True)
        _make_df(dates).to_parquet(path)

    def test_stale_thursday_cache_on_sunday_is_rejected(self) -> None:
        """The actual 2026-05-03 bug — Thursday cache on Sunday must fail."""
        self._save("LITE", ["2026-04-29", "2026-04-30"])
        # ref = Sunday 2026-05-03 (3 calendar / 1 trading day after Friday close)
        self.assertFalse(
            self.store.has_range("LITE", end="2026-05-03"),
            "Thursday-only cache must be rejected when ref is Sunday post-Friday-close",
        )

    def test_friday_cache_on_sunday_is_accepted(self) -> None:
        self._save("LITE", ["2026-04-29", "2026-04-30", "2026-05-01"])
        self.assertTrue(
            self.store.has_range("LITE", end="2026-05-03"),
            "Cache that includes Friday close must pass on Sunday",
        )

    def test_friday_cache_on_monday_morning_is_accepted(self) -> None:
        """Monday 9 AM cron — Friday close is the last complete NYSE session."""
        self._save("LITE", ["2026-04-30", "2026-05-01"])
        self.assertTrue(
            self.store.has_range("LITE", end="2026-05-04"),
            "Friday cache must pass on Monday — Friday is last complete session",
        )

    def test_thursday_cache_on_friday_is_accepted(self) -> None:
        """Friday morning before close — Thursday is last complete session."""
        self._save("LITE", ["2026-04-29", "2026-04-30"])
        self.assertTrue(
            self.store.has_range("LITE", end="2026-05-01"),
            "Thursday cache passes on Friday — Thursday is last completed close",
        )

    def test_thursday_cache_after_friday_close_is_rejected(self) -> None:
        """Friday after close — Friday's session is now required."""
        self._save("LITE", ["2026-04-29", "2026-04-30"])
        self.assertFalse(
            self.store.has_range(
                "LITE",
                end=pd.Timestamp("2026-05-01 16:05", tz="America/New_York"),
            ),
            "Thursday cache must fail after Friday market close",
        )

    def test_friday_cache_after_friday_close_is_accepted(self) -> None:
        self._save("LITE", ["2026-04-29", "2026-04-30", "2026-05-01"])
        self.assertTrue(
            self.store.has_range(
                "LITE",
                end=pd.Timestamp("2026-05-01 16:05", tz="America/New_York"),
            ),
            "Friday cache must pass after Friday market close",
        )

    def test_end_none_uses_today_implicitly(self) -> None:
        """The actual buggy call site: fetch_ohlcv passed end=None."""
        # Build cache 5 trading days behind today
        today = _dt.date.today()
        old_date = (today - _dt.timedelta(days=10)).isoformat()
        self._save("LITE", [old_date])
        # No end given — must still detect staleness (this was the regression)
        self.assertFalse(
            self.store.has_range("LITE", end=None),
            "end=None must default to today — stale cache must not slip through",
        )

    def test_legacy_tolerance_days_parameter_still_works(self) -> None:
        """Backwards-compat: callers passing tolerance_days get legacy behaviour."""
        self._save("LITE", ["2026-04-25"])
        # tolerance_days=20 is so loose any cache passes
        self.assertTrue(
            self.store.has_range("LITE", end="2026-05-03", tolerance_days=20),
            "Explicit tolerance_days override should bypass NYSE check",
        )

    def test_missing_cache_returns_false(self) -> None:
        self.assertFalse(self.store.has_range("NEVER_FETCHED"))

    def test_start_check_unaffected(self) -> None:
        self._save("LITE", ["2026-04-29", "2026-04-30", "2026-05-01"])
        # cache starts at 2026-04-29 → start before that should fail
        self.assertFalse(
            self.store.has_range("LITE", start="2026-01-01", end="2026-05-03"),
        )


class TestDataFreshnessGateTask(unittest.TestCase):
    """The inference pipeline's hard gate against stale market data."""

    def setUp(self) -> None:
        from kernel.pipeline.task_data_freshness import DataFreshnessGateTask  # noqa: PLC0415,E402
        self.Task = DataFreshnessGateTask
        # minimal ctx-like stub
        self.ctx = type(
            "Ctx",
            (),
            {
                "today": _dt.date(2026, 5, 3),  # Sunday
                "run_timestamp": None,
                "ohlcv": {},
                "config": {},
                "counters": {},
            },
        )()

    def _ohlcv(self, sym_to_max_date: dict[str, str]) -> dict[str, pd.DataFrame]:
        out = {}
        for sym, max_d in sym_to_max_date.items():
            idx = pd.date_range(end=max_d, periods=10, freq="B")
            out[sym] = pd.DataFrame({"close": 1.0}, index=idx)
        return out

    def test_fresh_panel_with_friday_close_passes(self) -> None:
        self.ctx.ohlcv = self._ohlcv({
            "AAPL": "2026-05-01", "MSFT": "2026-05-01", "SPY": "2026-05-01",
        })
        self.Task().run(self.ctx)  # should not raise

    def test_stale_thursday_panel_on_sunday_raises(self) -> None:
        self.ctx.ohlcv = self._ohlcv({
            "AAPL": "2026-05-01", "MSFT": "2026-04-30", "SPY": "2026-05-01",
        })
        with self.assertRaises(RuntimeError) as cm:
            self.Task().run(self.ctx)
        self.assertIn("stale", str(cm.exception).lower())
        self.assertIn("MSFT", str(cm.exception))

    def test_panel_uniformly_one_day_stale_raises(self) -> None:
        """Production failure mode 2026-05-03: ALL tickers stop at Thursday."""
        self.ctx.ohlcv = self._ohlcv({
            f"T{i}": "2026-04-30" for i in range(10)
        })
        with self.assertRaises(RuntimeError):
            self.Task().run(self.ctx)

    def test_disabled_via_config_skips_gate(self) -> None:
        """Backtest / debug escape: data_freshness.enabled=false bypasses the gate."""
        self.ctx.config = {"data_freshness": {"enabled": False}}
        self.ctx.ohlcv = self._ohlcv({"AAPL": "2026-04-01"})  # ancient
        self.Task().run(self.ctx)  # must not raise when disabled

    def test_empty_ohlcv_skips_with_warning(self) -> None:
        """Empty ohlcv is downstream's problem; gate scope is staleness."""
        self.ctx.ohlcv = {}
        # Should not raise — gate logs warning and defers to downstream.
        self.Task().run(self.ctx)

    def test_ctx_today_governs_reference_date(self) -> None:
        """For a backtest at ctx.today=2024-06-15, panel up to 2024-06-14 passes."""
        self.ctx.today = _dt.date(2024, 6, 15)  # Saturday
        self.ctx.ohlcv = self._ohlcv({"AAPL": "2024-06-14"})  # Friday
        self.Task().run(self.ctx)  # passes — 2024-06-14 is last NYSE close

    def test_friday_after_close_requires_friday_bar(self) -> None:
        self.ctx.today = _dt.date(2026, 5, 1)
        self.ctx.run_timestamp = pd.Timestamp("2026-05-01 16:05", tz="America/New_York")
        self.ctx.ohlcv = self._ohlcv({"AAPL": "2026-04-30"})
        with self.assertRaises(RuntimeError):
            self.Task().run(self.ctx)

    def test_friday_before_close_accepts_thursday_bar(self) -> None:
        self.ctx.today = _dt.date(2026, 5, 1)
        self.ctx.run_timestamp = pd.Timestamp("2026-05-01 15:55", tz="America/New_York")
        self.ctx.ohlcv = self._ohlcv({"AAPL": "2026-04-30"})
        self.Task().run(self.ctx)


class TestInferencePipelineWiring(unittest.TestCase):
    """The gate is wired into both InferencePipeline and SellOnlyPipeline."""

    def test_inference_pipeline_imports_gate_before_regime(self) -> None:
        path = REPO / "backtesting" / "renquant_104" / "kernel" / "pipeline" / "pp_inference.py"
        src = path.read_text()
        # Source-level invariant: gate import + .run() must appear in the
        # InferencePipeline.run() body BEFORE RegimeJob().run(ctx).
        idx_inf = src.find("class InferencePipeline")
        idx_sell = src.find("class SellOnlyPipeline")
        self.assertGreater(idx_inf, 0)
        self.assertGreater(idx_sell, idx_inf)
        inf_body = src[idx_inf:idx_sell]
        idx_gate = inf_body.find("DataFreshnessGateTask().run(ctx)")
        idx_regime = inf_body.find("RegimeJob().run(ctx)")
        self.assertGreater(idx_gate, 0, "gate not wired into InferencePipeline")
        self.assertGreater(idx_regime, 0)
        self.assertLess(idx_gate, idx_regime,
                        "gate must run BEFORE RegimeJob")

    def test_sell_only_pipeline_also_has_gate(self) -> None:
        path = REPO / "backtesting" / "renquant_104" / "kernel" / "pipeline" / "pp_inference.py"
        src = path.read_text()
        idx_sell = src.find("class SellOnlyPipeline")
        sell_body = src[idx_sell:]
        idx_gate = sell_body.find("DataFreshnessGateTask().run(ctx)")
        idx_regime = sell_body.find("RegimeJob().run(ctx)")
        self.assertGreater(idx_gate, 0,
                           "gate not wired into SellOnlyPipeline")
        self.assertLess(idx_gate, idx_regime,
                        "gate must run BEFORE RegimeJob in sell-only too")


if __name__ == "__main__":
    unittest.main()
