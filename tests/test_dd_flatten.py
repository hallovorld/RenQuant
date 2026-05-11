"""S-2 regression guard — HARD FLATTEN at drawdown threshold.

Pins:
  (1) Disabled / no config / threshold=0 → no-op (golden preserved).
  (2) drawdown < flatten_pct → no-op.
  (3) drawdown >= flatten_pct → flatten signal for every untouched
      holding; pre-existing path-rule exits preserved.
  (4) Empty holdings or non-finite PV → no-op.
  (5) ctx.skip_buys forced True on flatten bar.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.exits import ExitSignal  # noqa: E402
from kernel.pipeline.task_dd_flatten import DrawdownFlattenTask  # noqa: E402


def _ctx(*, hwm=100.0, pv=100.0, holdings=None, exits=None, cfg_flat=None):
    cfg = {"risk": {}}
    if cfg_flat is not None:
        cfg["risk"]["drawdown_flatten"] = cfg_flat
    return SimpleNamespace(
        config=cfg,
        hwm=hwm,
        portfolio_value=pv,
        holdings=holdings or {},
        exits=exits if exits is not None else [],
        skip_buys=False,
    )


class TestDrawdownFlattenTask:

    def test_disabled_is_noop(self):
        ctx = _ctx(hwm=100.0, pv=50.0,
                   holdings={"AAPL": object(), "MSFT": object()})
        DrawdownFlattenTask().run(ctx)
        assert ctx.exits == []
        assert ctx.skip_buys is False

    def test_no_config_is_noop(self):
        ctx = _ctx(hwm=100.0, pv=50.0,
                   holdings={"AAPL": object()},
                   cfg_flat=None)
        DrawdownFlattenTask().run(ctx)
        assert ctx.exits == []

    def test_flatten_pct_zero_is_noop(self):
        ctx = _ctx(hwm=100.0, pv=50.0,
                   holdings={"AAPL": object()},
                   cfg_flat={"enabled": True, "flatten_pct": 0.0})
        DrawdownFlattenTask().run(ctx)
        assert ctx.exits == []

    def test_dd_below_threshold_is_noop(self):
        # 20% DD, threshold 25% → below
        ctx = _ctx(hwm=100.0, pv=80.0,
                   holdings={"AAPL": object(), "MSFT": object()},
                   cfg_flat={"enabled": True, "flatten_pct": 0.25})
        DrawdownFlattenTask().run(ctx)
        assert ctx.exits == []
        assert ctx.skip_buys is False

    def test_dd_at_threshold_flattens(self):
        # 25% DD, threshold 25% → fire
        ctx = _ctx(hwm=100.0, pv=75.0,
                   holdings={"AAPL": object(), "MSFT": object()},
                   cfg_flat={"enabled": True, "flatten_pct": 0.25})
        DrawdownFlattenTask().run(ctx)
        assert len(ctx.exits) == 2
        tickers = sorted([t for (t, _) in ctx.exits])
        assert tickers == ["AAPL", "MSFT"]
        for _, sig in ctx.exits:
            assert sig.should_exit is True
            assert sig.exit_type == "drawdown_flatten"
        assert ctx.skip_buys is True

    def test_dd_far_above_threshold_flattens(self):
        ctx = _ctx(hwm=100.0, pv=40.0,
                   holdings={"AAPL": object(), "MSFT": object(), "GOOG": object()},
                   cfg_flat={"enabled": True, "flatten_pct": 0.25})
        DrawdownFlattenTask().run(ctx)
        assert len(ctx.exits) == 3
        assert ctx.skip_buys is True

    def test_preserves_existing_path_rule_exits(self):
        # AAPL already has a trailing_stop; MSFT gets flatten.
        path = ExitSignal(should_exit=True, reason="trailing", exit_type="trailing_stop")
        ctx = _ctx(hwm=100.0, pv=70.0,
                   holdings={"AAPL": object(), "MSFT": object()},
                   exits=[("AAPL", path)],
                   cfg_flat={"enabled": True, "flatten_pct": 0.25})
        DrawdownFlattenTask().run(ctx)
        # AAPL kept its trailing_stop; MSFT got flatten.
        assert len(ctx.exits) == 2
        sigs = dict(ctx.exits)
        assert sigs["AAPL"].exit_type == "trailing_stop"
        assert sigs["MSFT"].exit_type == "drawdown_flatten"
        assert ctx.skip_buys is True

    def test_no_double_emit_for_path_ruled_ticker(self):
        path = ExitSignal(should_exit=True, reason="sl", exit_type="stop_loss")
        ctx = _ctx(hwm=100.0, pv=70.0,
                   holdings={"AAPL": object()},
                   exits=[("AAPL", path)],
                   cfg_flat={"enabled": True, "flatten_pct": 0.25})
        DrawdownFlattenTask().run(ctx)
        # AAPL keeps its stop_loss — only one exit entry.
        assert len(ctx.exits) == 1
        assert ctx.exits[0][1].exit_type == "stop_loss"

    def test_empty_holdings_is_noop(self):
        ctx = _ctx(hwm=100.0, pv=50.0,
                   holdings={},
                   cfg_flat={"enabled": True, "flatten_pct": 0.25})
        DrawdownFlattenTask().run(ctx)
        assert ctx.exits == []

    def test_nonfinite_pv_is_noop(self):
        ctx = _ctx(hwm=100.0, pv=float("nan"),
                   holdings={"AAPL": object()},
                   cfg_flat={"enabled": True, "flatten_pct": 0.25})
        DrawdownFlattenTask().run(ctx)
        assert ctx.exits == []

    def test_zero_hwm_is_noop(self):
        ctx = _ctx(hwm=0.0, pv=80.0,
                   holdings={"AAPL": object()},
                   cfg_flat={"enabled": True, "flatten_pct": 0.25})
        DrawdownFlattenTask().run(ctx)
        assert ctx.exits == []
