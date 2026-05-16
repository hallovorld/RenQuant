"""Phase 2D unit tests — short cover stop-loss + IRC §1233 marker.

NOT YET WIRED into the pipeline. Tests prove the tasks behave correctly
when called; wiring is a separate operator decision per
backtesting/renquant_104/kernel/pipeline/task_short_cover.py docstring.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _short(qty, entry):
    return SimpleNamespace(qty=qty, entry_price=entry)


def _ohlcv(price):
    return pd.DataFrame({"close": [price] * 5})


class TestShortCoverStopLoss:

    def test_short_underwater_15pct_triggers_cover(self):
        """Short entered at $100, current at $115 (15% loss for short)
        → cover at exactly threshold."""
        from kernel.pipeline.task_short_cover import ShortCoverStopLossTask
        ctx = SimpleNamespace(
            config={"risk": {"short_cover_stop_pct": 0.15}},
            short_holdings={"X": _short(qty=-10, entry=100.0)},
            ohlcv={"X": _ohlcv(115.0)},  # +15% from entry = -15% for short
            exits=[], counters={},
        )
        ShortCoverStopLossTask().run(ctx)
        assert len(ctx.exits) == 1
        ticker, sig = ctx.exits[0]
        assert ticker == "X"
        assert sig.reason == "short_cover_stop"
        assert sig.qty == 10  # POSITIVE (buy_to_close)
        assert sig.details["side"] == "buy_to_close"
        assert sig.details["loss_pct"] == 0.15
        assert sig.details["tax_holding_period"] == "ST_FORCED_§1233"
        assert ctx.counters["short_cover_stop_triggered"] == 1

    def test_short_within_threshold_no_cover(self):
        """Short at $100, current $110 (10% loss) — below 15% trigger.
        Should NOT fire."""
        from kernel.pipeline.task_short_cover import ShortCoverStopLossTask
        ctx = SimpleNamespace(
            config={"risk": {"short_cover_stop_pct": 0.15}},
            short_holdings={"X": _short(qty=-10, entry=100.0)},
            ohlcv={"X": _ohlcv(110.0)},
            exits=[], counters={},
        )
        ShortCoverStopLossTask().run(ctx)
        assert ctx.exits == []
        assert "short_cover_stop_triggered" not in ctx.counters

    def test_short_winning_no_cover(self):
        """Short at $100, current $80 (the short is +20% PROFIT) — no cover."""
        from kernel.pipeline.task_short_cover import ShortCoverStopLossTask
        ctx = SimpleNamespace(
            config={"risk": {"short_cover_stop_pct": 0.15}},
            short_holdings={"X": _short(qty=-10, entry=100.0)},
            ohlcv={"X": _ohlcv(80.0)},
            exits=[], counters={},
        )
        ShortCoverStopLossTask().run(ctx)
        assert ctx.exits == []

    def test_disabled_no_op(self):
        from kernel.pipeline.task_short_cover import ShortCoverStopLossTask
        ctx = SimpleNamespace(
            config={"risk": {"short_cover_stop_enabled": False}},
            short_holdings={"X": _short(qty=-10, entry=100.0)},
            ohlcv={"X": _ohlcv(120.0)},
            exits=[], counters={},
        )
        ShortCoverStopLossTask().run(ctx)
        assert ctx.exits == []

    def test_long_position_ignored(self):
        """A LONG position (qty > 0) in short_holdings (corrupted state)
        should be skipped, not generate a cover order."""
        from kernel.pipeline.task_short_cover import ShortCoverStopLossTask
        ctx = SimpleNamespace(
            config={"risk": {}},
            short_holdings={"X": _short(qty=+10, entry=100.0)},
            ohlcv={"X": _ohlcv(120.0)},
            exits=[], counters={},
        )
        ShortCoverStopLossTask().run(ctx)
        assert ctx.exits == []

    def test_multiple_shorts_partial_trigger(self):
        from kernel.pipeline.task_short_cover import ShortCoverStopLossTask
        ctx = SimpleNamespace(
            config={"risk": {"short_cover_stop_pct": 0.15}},
            short_holdings={
                "WIN":  _short(qty=-10, entry=100.0),  # 80 cur → +20% short profit
                "MEH":  _short(qty=-10, entry=100.0),  # 110 cur → -10% loss, no trig
                "LOSE": _short(qty=-10, entry=100.0),  # 120 cur → -20% loss, COVER
            },
            ohlcv={"WIN": _ohlcv(80.0), "MEH": _ohlcv(110.0), "LOSE": _ohlcv(120.0)},
            exits=[], counters={},
        )
        ShortCoverStopLossTask().run(ctx)
        assert len(ctx.exits) == 1
        assert ctx.exits[0][0] == "LOSE"


class TestIRC1233TaxMarker:

    def test_short_cover_marked_st(self):
        from kernel.pipeline.task_short_cover import IRC1233TaxMarkerTask
        trades = [
            {"ticker": "X", "side": "buy", "position_intent": "buy_to_close",
             "qty": 10, "filled_avg_price": 120.0},
            {"ticker": "Y", "side": "sell", "position_intent": "sell_to_close",
             "qty": 5, "filled_avg_price": 200.0},  # long sell — should NOT mark
            {"ticker": "Z", "side": "buy", "position_intent": "buy_to_open",
             "qty": 3, "filled_avg_price": 150.0},  # long open — should NOT mark
        ]
        ctx = SimpleNamespace(
            config={"tax": {}},
            realized_trades=trades, counters={},
        )
        IRC1233TaxMarkerTask().run(ctx)
        assert trades[0].get("tax_holding_period") == "ST_FORCED_§1233"
        assert "tax_holding_period" not in trades[1]
        assert "tax_holding_period" not in trades[2]
        assert ctx.counters["irc_1233_marker_applied"] == 1

    def test_disabled_no_op(self):
        from kernel.pipeline.task_short_cover import IRC1233TaxMarkerTask
        trades = [{"ticker": "X", "side": "buy", "position_intent": "buy_to_close"}]
        ctx = SimpleNamespace(
            config={"tax": {"irc_1233_marker_enabled": False}},
            realized_trades=trades, counters={},
        )
        IRC1233TaxMarkerTask().run(ctx)
        assert "tax_holding_period" not in trades[0]
