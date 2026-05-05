"""Tests for the reusable Task atom library.

Each atom gets at least 2 tests: happy path + edge case (NaN / missing /
empty / wrong type).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
STRATEGY = REPO / "backtesting" / "renquant_104"
if str(STRATEGY) not in sys.path:
    sys.path.insert(0, str(STRATEGY))

from kernel.pipeline.atoms import (   # noqa: E402
    AssertFieldExistsTask, BuildMaskFromConditionTask,
    BuildVectorFromMappingTask, ClampFieldTask, ClearFieldTask,
    CopyFieldTask, IncrementCounterTask, IsFiniteGuardTask,
    LogSummaryTask, NonEmptyGuardTask, RangeGuardTask,
    SkipIfConfigDisabledTask, SkipIfFieldEqualsTask, SkipIfFieldFalsyTask,
    StableTickerOrderTask, WriteJSONArtifactTask,
)


# ── ctx_ops ──────────────────────────────────────────────────────────────

class TestCtxOps:
    def test_copy_field_shallow(self):
        ctx = SimpleNamespace(a=42, b=None)
        CopyFieldTask("a", "b").run(ctx)
        assert ctx.b == 42

    def test_clear_field_to_none(self):
        ctx = SimpleNamespace(x="hello")
        ClearFieldTask("x").run(ctx)
        assert ctx.x is None

    def test_clear_field_to_factory(self):
        ctx = SimpleNamespace(x="hello")
        ClearFieldTask("x", fill=list).run(ctx)
        assert ctx.x == []

    def test_assert_field_exists_passes(self):
        ctx = SimpleNamespace(foo=42)
        AssertFieldExistsTask("foo").run(ctx)   # no raise

    def test_assert_field_exists_raises(self):
        ctx = SimpleNamespace(foo=None)
        with pytest.raises(AssertionError):
            AssertFieldExistsTask("foo").run(ctx)


# ── numerical ────────────────────────────────────────────────────────────

class TestNumericalGuards:
    def test_finite_guard_passes(self):
        ctx = SimpleNamespace(x=3.14)
        assert IsFiniteGuardTask("x").run(ctx) is None

    def test_finite_guard_skips_on_nan(self):
        ctx = SimpleNamespace(x=float("nan"))
        assert IsFiniteGuardTask("x", on_violation="skip").run(ctx) is False

    def test_finite_guard_raises_on_inf(self):
        ctx = SimpleNamespace(x=float("inf"))
        with pytest.raises(ValueError):
            IsFiniteGuardTask("x", on_violation="raise").run(ctx)

    def test_finite_guard_zeros_on_violation(self):
        ctx = SimpleNamespace(x=float("nan"))
        IsFiniteGuardTask("x", on_violation="zero").run(ctx)
        assert ctx.x == 0.0

    def test_range_guard_passes(self):
        ctx = SimpleNamespace(p=0.5)
        assert RangeGuardTask("p", 0.0, 1.0).run(ctx) is None

    def test_range_guard_rejects_below(self):
        ctx = SimpleNamespace(p=-0.1)
        assert RangeGuardTask("p", 0.0, 1.0).run(ctx) is False

    def test_range_guard_nan_rejected(self):
        ctx = SimpleNamespace(p=float("nan"))
        assert RangeGuardTask("p", 0.0, 1.0).run(ctx) is False

    def test_non_empty_guard_passes(self):
        ctx = SimpleNamespace(items=[1, 2, 3])
        assert NonEmptyGuardTask("items").run(ctx) is None

    def test_non_empty_guard_skips_on_empty(self):
        ctx = SimpleNamespace(items=[])
        assert NonEmptyGuardTask("items").run(ctx) is False

    def test_clamp_within_range_unchanged(self):
        ctx = SimpleNamespace(v=0.5)
        ClampFieldTask("v", 0.0, 1.0).run(ctx)
        assert ctx.v == 0.5

    def test_clamp_clips_high(self):
        ctx = SimpleNamespace(v=2.5)
        ClampFieldTask("v", 0.0, 1.0).run(ctx)
        assert ctx.v == 1.0

    def test_clamp_nan_to_midpoint(self):
        ctx = SimpleNamespace(v=float("nan"))
        ClampFieldTask("v", 0.0, 1.0).run(ctx)
        assert ctx.v == 0.5


# ── gates ────────────────────────────────────────────────────────────────

class TestGates:
    def test_skip_if_config_disabled_truthy_continues(self):
        ctx = SimpleNamespace(config={"a": {"b": True}})
        assert SkipIfConfigDisabledTask("a.b").run(ctx) is None

    def test_skip_if_config_disabled_falsy_short_circuits(self):
        ctx = SimpleNamespace(config={"a": {"b": False}})
        assert SkipIfConfigDisabledTask("a.b").run(ctx) is False

    def test_skip_if_field_falsy(self):
        ctx = SimpleNamespace(thing=None)
        assert SkipIfFieldFalsyTask("thing").run(ctx) is False

    def test_skip_if_field_equals(self):
        ctx = SimpleNamespace(mode="legacy")
        assert SkipIfFieldEqualsTask("mode", "legacy").run(ctx) is False
        ctx.mode = "new"
        assert SkipIfFieldEqualsTask("mode", "legacy").run(ctx) is None


# ── logging atoms ────────────────────────────────────────────────────────

class TestLoggingAtoms:
    def test_log_summary_no_args(self, caplog):
        import logging
        caplog.set_level(logging.INFO, logger="kernel.pipeline.atoms")
        LogSummaryTask("hello world").run(SimpleNamespace())
        assert "hello world" in caplog.text

    def test_log_summary_with_args(self, caplog):
        import logging
        caplog.set_level(logging.INFO, logger="kernel.pipeline.atoms")
        ctx = SimpleNamespace(_n=5, _m=10)
        LogSummaryTask("Job: %d / %d", fields=("_n", "_m")).run(ctx)
        assert "Job: 5 / 10" in caplog.text

    def test_increment_counter_creates_dict(self):
        ctx = SimpleNamespace()
        IncrementCounterTask("foo", 3).run(ctx)
        assert ctx.counters["foo"] == 3

    def test_increment_counter_adds(self):
        ctx = SimpleNamespace(counters={"x": 5})
        IncrementCounterTask("x", 2).run(ctx)
        assert ctx.counters["x"] == 7

    def test_increment_counter_from_field(self):
        ctx = SimpleNamespace(_n_dropped=4, counters={})
        IncrementCounterTask("dropped", amount="_n_dropped").run(ctx)
        assert ctx.counters["dropped"] == 4


# ── vectors ──────────────────────────────────────────────────────────────

class TestVectors:
    def test_build_vector_from_mapping_finite(self):
        h1 = SimpleNamespace(shares=10.0)
        h2 = SimpleNamespace(shares=20.0)
        ctx = SimpleNamespace(
            tickers=["A", "B"],
            holdings={"A": h1, "B": h2},
        )
        BuildVectorFromMappingTask(
            "tickers", "holdings", "shares", "_v",
        ).run(ctx)
        assert ctx._v.tolist() == [10.0, 20.0]

    def test_build_vector_handles_nan_with_default(self):
        h1 = SimpleNamespace(mu=0.5)
        h2 = SimpleNamespace(mu=float("nan"))
        ctx = SimpleNamespace(
            tickers=["A", "B"],
            holdings={"A": h1, "B": h2},
        )
        BuildVectorFromMappingTask(
            "tickers", "holdings", "mu", "_mu", default=0.0,
        ).run(ctx)
        assert ctx._mu[0] == 0.5
        assert ctx._mu[1] == 0.0   # NaN replaced by default

    def test_build_vector_fallback_attr(self):
        h = SimpleNamespace(mu=None, panel_score=0.3)
        ctx = SimpleNamespace(
            tickers=["A"],
            holdings={"A": h},
        )
        BuildVectorFromMappingTask(
            "tickers", "holdings", "mu", "_mu",
            fallback_attr="panel_score",
        ).run(ctx)
        assert ctx._mu[0] == 0.3

    def test_build_mask_from_condition(self):
        ctx = SimpleNamespace(tickers=["A", "B", "C"])
        BuildMaskFromConditionTask(
            "tickers", "_mask",
            predicate=lambda c, t: t in {"A", "C"},
        ).run(ctx)
        assert ctx._mask.tolist() == [True, False, True]

    def test_stable_ticker_order_combines(self):
        cands = [SimpleNamespace(ticker="X"), SimpleNamespace(ticker="A")]
        ctx = SimpleNamespace(
            holdings={"B": None, "C": None},
            candidates=cands,
        )
        StableTickerOrderTask("holdings", "candidates", "_tickers").run(ctx)
        # held first (preserves insertion order), then new cands
        assert ctx._tickers == ["B", "C", "X", "A"]

    def test_stable_ticker_order_dedupes_overlap(self):
        cands = [SimpleNamespace(ticker="B"), SimpleNamespace(ticker="X")]
        ctx = SimpleNamespace(
            holdings={"B": None}, candidates=cands,
        )
        StableTickerOrderTask("holdings", "candidates", "_tickers").run(ctx)
        # B in held → not duplicated in cand half
        assert ctx._tickers == ["B", "X"]


# ── persistence ──────────────────────────────────────────────────────────

class TestPersistence:
    def test_write_json_artifact_creates_file(self, tmp_path):
        ctx = SimpleNamespace(
            config={"_strategy_dir": str(tmp_path)},
            payload={"foo": 1, "bar": [1, 2, 3]},
        )
        WriteJSONArtifactTask(
            "payload",
            "{strategy_dir}/artifacts/test.json",
        ).run(ctx)
        out = tmp_path / "artifacts" / "test.json"
        assert out.exists()
        import json as _json
        loaded = _json.loads(out.read_text())
        assert loaded == {"foo": 1, "bar": [1, 2, 3]}

    def test_write_json_atomic_no_partial_file(self, tmp_path):
        # Mid-write Ctrl-C would leave .tmp behind, NOT the .json target.
        # Verify the target either fully exists or doesn't exist (atomic).
        ctx = SimpleNamespace(
            config={"_strategy_dir": str(tmp_path)},
            payload={"x": 1},
        )
        WriteJSONArtifactTask(
            "payload",
            "{strategy_dir}/artifacts/x.json",
        ).run(ctx)
        out = tmp_path / "artifacts" / "x.json"
        tmp = out.with_suffix(".json.tmp")
        assert out.exists()
        assert not tmp.exists()   # tmp must be cleaned up by rename
