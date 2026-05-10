"""Tests for kernel.typed_past — frozen Past + TypedTask Protocol.

Foundation tests for the multi-week migration to a typed Past-only data
contract on Tasks (cvxportfolio Estimator.values_in_time pattern).

Coverage:
  * Past is immutable (FrozenInstanceError on mutation)
  * slice_until rejects misconstructed snapshots and slices correctly
  * Equality based on (t, content)
  * Hash is stable on equal Pasts
  * TypedTaskAdapter wires a TypedTask into a legacy ctx-driven chain
  * §5.13.1: a real existing Task (DataFreshnessGateTask) migrated to
    TypedTask form produces the same behaviour as the legacy version
    when invoked through the adapter, exercising the real prod code
    paths (NYSE calendar lookup, not a synthetic short-circuit).
"""
from __future__ import annotations

import datetime as _dt
import sys
import types
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.typed_past import Past, TaskResult, TypedTask, TypedTaskAdapter  # noqa: E402
from kernel.typed_past.typed_data_freshness import TypedDataFreshnessGate  # noqa: E402


def _ohlcv_df(dates: list[str]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    return pd.DataFrame({"close": 1.0}, index=idx)


def _empty_source(t: str) -> dict:
    return {
        "ohlcv": pd.DataFrame(index=pd.DatetimeIndex([])),
        "fundamentals": pd.DataFrame(index=pd.DatetimeIndex([])),
        "regime_history": (),
        "holdings": {},
        "cash": 1000.0,
    }


class TestPastImmutability(unittest.TestCase):
    """Past is frozen — every attempt to mutate must raise."""

    def test_assignment_raises_frozen_instance_error(self) -> None:
        past = Past.slice_until("2026-05-01", _empty_source("2026-05-01"))
        with self.assertRaises(FrozenInstanceError):
            past.cash = 2000.0  # type: ignore[misc]

    def test_holdings_is_frozen_mapping(self) -> None:
        past = Past.slice_until("2026-05-01", _empty_source("2026-05-01"))
        # MappingProxyType refuses item assignment
        with self.assertRaises(TypeError):
            past.holdings["AAPL"] = 100  # type: ignore[index]

    def test_regime_history_is_tuple(self) -> None:
        past = Past.slice_until("2026-05-01", _empty_source("2026-05-01"))
        self.assertIsInstance(past.regime_history, tuple)


class TestSliceUntil(unittest.TestCase):
    """slice_until factory — rejects future rows; happy path slices to t."""

    def test_happy_path_slices_to_t(self) -> None:
        ohlcv = _ohlcv_df(["2026-04-28", "2026-04-29", "2026-04-30", "2026-05-01"])
        past = Past.slice_until(
            "2026-04-29",
            {
                "ohlcv": ohlcv,
                "fundamentals": pd.DataFrame(index=pd.DatetimeIndex([])),
                "regime_history": ("BULL_CALM",),
                "holdings": {"AAPL": 10},
                "cash": 5000.0,
            },
        )
        # only 04-28 and 04-29 should remain
        self.assertEqual(len(past.ohlcv), 2)
        self.assertEqual(past.ohlcv.index.max(), pd.Timestamp("2026-04-29"))
        self.assertEqual(past.cash, 5000.0)
        self.assertEqual(past.regime_history, ("BULL_CALM",))

    def test_direct_construction_rejects_future_rows(self) -> None:
        """If someone bypasses slice_until and passes a pre-filtered df with a
        post-t row, __post_init__ assertion must catch it."""
        ohlcv = _ohlcv_df(["2026-04-28", "2026-05-15"])  # 05-15 > t
        with self.assertRaises(AssertionError):
            Past(
                t=pd.Timestamp("2026-04-29"),
                ohlcv=ohlcv,
                fundamentals=pd.DataFrame(index=pd.DatetimeIndex([])),
                regime_history=(),
                holdings=types.MappingProxyType({}),
                cash=0.0,
            )

    def test_non_datetime_index_rejected(self) -> None:
        bad = pd.DataFrame({"close": [1.0, 2.0]}, index=[0, 1])
        with self.assertRaises(TypeError):
            Past.slice_until(
                "2026-04-29",
                {
                    "ohlcv": bad,
                    "fundamentals": pd.DataFrame(index=pd.DatetimeIndex([])),
                    "regime_history": (),
                    "holdings": {},
                    "cash": 0.0,
                },
            )


class TestPastEquality(unittest.TestCase):
    def test_equal_pasts_are_equal(self) -> None:
        s = _empty_source("2026-05-01")
        a = Past.slice_until("2026-05-01", s)
        b = Past.slice_until("2026-05-01", s)
        self.assertEqual(a, b)

    def test_different_t_not_equal(self) -> None:
        a = Past.slice_until("2026-05-01", _empty_source("2026-05-01"))
        b = Past.slice_until("2026-05-02", _empty_source("2026-05-02"))
        self.assertNotEqual(a, b)

    def test_hash_stable_on_equal_pasts(self) -> None:
        s = _empty_source("2026-05-01")
        a = Past.slice_until("2026-05-01", s)
        b = Past.slice_until("2026-05-01", s)
        self.assertEqual(hash(a), hash(b))


class TestTypedTaskProtocol(unittest.TestCase):
    """TypedTask Protocol — runtime-checkable; TypedDataFreshnessGate matches."""

    def test_typed_data_freshness_gate_satisfies_protocol(self) -> None:
        gate = TypedDataFreshnessGate()
        self.assertIsInstance(gate, TypedTask)

    def test_non_typed_task_rejected_by_adapter(self) -> None:
        with self.assertRaises(TypeError):
            TypedTaskAdapter(typed=object())  # type: ignore[arg-type]


class TestTypedTaskAdapterRealProdPath(unittest.TestCase):
    """§5.13.1 — exercise a TypedTask through the adapter using a ctx-shaped
    object that mirrors what InferencePipeline passes. Behaviour must match
    the legacy DataFreshnessGateTask invoked on the same inputs.
    """

    def setUp(self) -> None:
        from kernel.pipeline.task_data_freshness import DataFreshnessGateTask  # noqa: PLC0415,E402
        self.LegacyTask = DataFreshnessGateTask

    def _ctx(self, max_date: str, today: _dt.date) -> object:
        idx = pd.date_range(end=max_date, periods=10, freq="B")
        return type(
            "Ctx",
            (),
            {
                "today": today,
                "ohlcv": {
                    "AAPL": pd.DataFrame({"close": 1.0}, index=idx),
                    "MSFT": pd.DataFrame({"close": 2.0}, index=idx),
                },
                "config": {},
                "counters": {},
                "cash": 0.0,
                "holdings": {},
                "regime_counts": {},
            },
        )()

    def test_fresh_panel_passes_through_typed_adapter(self) -> None:
        ctx = self._ctx("2026-05-01", _dt.date(2026, 5, 3))  # Sun: last close Fri 5/1
        adapter = TypedTaskAdapter(TypedDataFreshnessGate())
        # Must not raise — fresh data
        ok = adapter.run(ctx)
        self.assertTrue(ok is True or ok is None)

    def test_stale_panel_raises_through_typed_adapter(self) -> None:
        # MSFT/AAPL both at 04-30 (Thu), today=Sun 5/3 → last close Fri 5/1 is missing
        ctx = self._ctx("2026-04-30", _dt.date(2026, 5, 3))
        adapter = TypedTaskAdapter(TypedDataFreshnessGate())
        with self.assertRaises(RuntimeError) as cm:
            adapter.run(ctx)
        self.assertIn("STALE", str(cm.exception).upper())

    def test_typed_and_legacy_agree_on_fresh_panel(self) -> None:
        """Same inputs → same go/no-go decision in legacy and typed forms.
        This is the §5.13.1 test that exercises BOTH code paths so a future
        regression in either implementation is caught."""
        ctx_legacy = self._ctx("2026-05-01", _dt.date(2026, 5, 3))
        ctx_typed = self._ctx("2026-05-01", _dt.date(2026, 5, 3))
        # Legacy
        self.LegacyTask().run(ctx_legacy)  # no raise
        # Typed
        TypedTaskAdapter(TypedDataFreshnessGate()).run(ctx_typed)  # no raise

    def test_typed_and_legacy_agree_on_stale_panel(self) -> None:
        ctx_legacy = self._ctx("2026-04-30", _dt.date(2026, 5, 3))
        ctx_typed = self._ctx("2026-04-30", _dt.date(2026, 5, 3))
        with self.assertRaises(RuntimeError):
            self.LegacyTask().run(ctx_legacy)
        with self.assertRaises(RuntimeError):
            TypedTaskAdapter(TypedDataFreshnessGate()).run(ctx_typed)


class TestTaskResult(unittest.TestCase):
    def test_default_continue_chain_true(self) -> None:
        r = TaskResult()
        self.assertTrue(r.continue_chain)
        self.assertEqual(dict(r.ctx_writes), {})

    def test_task_result_is_frozen(self) -> None:
        r = TaskResult()
        with self.assertRaises(FrozenInstanceError):
            r.continue_chain = False  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
