"""R-03 regression guard — Hurst-Ooi-Pedersen 2017 12M SPY trend overlay.

Pins:
  (1) compute_spy_trend_return math: positive trend, negative trend,
      identity (no move), NaN/short series fail-open.
  (2) TrendOverlayTask: gated by config; only escalates hard_bear (never
      demotes); idempotent on already-True; skips on missing SPY OHLCV;
      respects threshold knob.
  (3) RegimeJob wiring: TrendOverlayTask sits between BEAROverrideTask
      and RegimeFinalizeTask (so RegimeFinalizeTask picks up the
      escalation via the canonical BEAR branch).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.pipeline.task_trend_overlay import (  # noqa: E402
    compute_spy_trend_return,
    TrendOverlayTask,
)
from kernel.pipeline.job_regime import RegimeJob  # noqa: E402
from kernel.pipeline.task_regime import (  # noqa: E402
    HurstTask, CUSUMTask, GMMTask, BEAROverrideTask, RegimeFinalizeTask,
)


def _spy_df(returns: list[float], start_price: float = 400.0) -> pd.DataFrame:
    """Build a SPY OHLCV-like frame from a sequence of daily returns."""
    closes = [start_price]
    for r in returns:
        closes.append(closes[-1] * (1.0 + r))
    return pd.DataFrame({"close": closes})


class TestComputeSpyTrendReturn:

    def test_positive_trend(self):
        # +0.1% daily for 252 days → ~28.7% cumulative.
        df = _spy_df([0.001] * 252)
        r = compute_spy_trend_return(df["close"], 252)
        assert r is not None
        assert r == pytest.approx(0.287, rel=0.05)

    def test_negative_trend(self):
        # -0.1% daily for 252 days → ~-22.3% cumulative.
        df = _spy_df([-0.001] * 252)
        r = compute_spy_trend_return(df["close"], 252)
        assert r is not None
        assert r == pytest.approx(-0.223, rel=0.05)

    def test_zero_trend(self):
        df = _spy_df([0.0] * 252)
        r = compute_spy_trend_return(df["close"], 252)
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_short_series_returns_none(self):
        df = _spy_df([0.001] * 100)
        assert compute_spy_trend_return(df["close"], 252) is None

    def test_none_series_returns_none(self):
        assert compute_spy_trend_return(None, 252) is None


def _ctx(*, ret_seq: list[float], cfg_overlay: dict | None = None,
         hard_bear: bool = False):
    cfg = {"regime": {}}
    if cfg_overlay is not None:
        cfg["regime"]["trend_overlay"] = cfg_overlay
    return SimpleNamespace(
        config=cfg,
        regime_state=SimpleNamespace(hard_bear=hard_bear),
        ohlcv={"SPY": _spy_df(ret_seq)} if ret_seq else {},
    )


class TestTrendOverlayTask:

    def test_disabled_block_is_noop(self):
        ctx = _ctx(ret_seq=[-0.001] * 252)  # bear trend
        TrendOverlayTask().run(ctx)
        assert ctx.regime_state.hard_bear is False

    def test_no_overlay_key_is_noop(self):
        ctx = _ctx(ret_seq=[-0.001] * 252, cfg_overlay=None)
        TrendOverlayTask().run(ctx)
        assert ctx.regime_state.hard_bear is False

    def test_negative_12m_return_triggers_hard_bear(self):
        ctx = _ctx(
            ret_seq=[-0.001] * 252,
            cfg_overlay={"enabled": True, "lookback_days": 252, "threshold": 0.0},
        )
        TrendOverlayTask().run(ctx)
        assert ctx.regime_state.hard_bear is True

    def test_positive_12m_return_does_not_trigger(self):
        ctx = _ctx(
            ret_seq=[0.001] * 252,
            cfg_overlay={"enabled": True, "lookback_days": 252, "threshold": 0.0},
        )
        TrendOverlayTask().run(ctx)
        assert ctx.regime_state.hard_bear is False

    def test_threshold_knob_respected(self):
        # ~28.7% cumulative; threshold +30% → still triggers.
        ctx = _ctx(
            ret_seq=[0.001] * 252,
            cfg_overlay={"enabled": True, "lookback_days": 252, "threshold": 0.30},
        )
        TrendOverlayTask().run(ctx)
        assert ctx.regime_state.hard_bear is True

    def test_already_hard_bear_is_preserved(self):
        # Even with a positive trend, an upstream True must not be demoted.
        ctx = _ctx(
            ret_seq=[0.001] * 252,
            cfg_overlay={"enabled": True, "lookback_days": 252, "threshold": 0.0},
            hard_bear=True,
        )
        TrendOverlayTask().run(ctx)
        assert ctx.regime_state.hard_bear is True

    def test_missing_spy_failopen(self):
        ctx = _ctx(
            ret_seq=[],
            cfg_overlay={"enabled": True, "lookback_days": 252, "threshold": 0.0},
        )
        TrendOverlayTask().run(ctx)
        assert ctx.regime_state.hard_bear is False

    def test_short_series_failopen(self):
        ctx = _ctx(
            ret_seq=[-0.001] * 100,  # only ~5 months
            cfg_overlay={"enabled": True, "lookback_days": 252, "threshold": 0.0},
        )
        TrendOverlayTask().run(ctx)
        assert ctx.regime_state.hard_bear is False


class TestRegimeJobWiring:
    """Pin the TrendOverlayTask position in RegimeJob.tasks."""

    def test_trend_overlay_between_bear_override_and_finalize(self):
        tasks = RegimeJob().tasks
        types = [type(t) for t in tasks]
        assert HurstTask in types
        assert CUSUMTask in types
        assert GMMTask in types
        assert BEAROverrideTask in types
        assert TrendOverlayTask in types
        assert RegimeFinalizeTask in types

        bear_idx     = types.index(BEAROverrideTask)
        overlay_idx  = types.index(TrendOverlayTask)
        finalize_idx = types.index(RegimeFinalizeTask)
        # Overlay must sit STRICTLY between BEAR and Finalize so the
        # canonical BEAR-resolution branch picks up the escalation.
        assert bear_idx < overlay_idx < finalize_idx
