"""Phase 2D unit tests — short cover stop-loss + IRC §1233 marker."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import inspect

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _short(qty, entry):
    return SimpleNamespace(qty=qty, shares=qty, entry_price=entry)


def _ohlcv(price):
    return pd.DataFrame({"close": [price] * 5})


class TestShortCoverStopLoss:

    def test_short_underwater_15pct_triggers_cover(self):
        """Short entered at $100, current at $115 (15% loss for short)
        → cover at exactly threshold."""
        from kernel.pipeline.task_short_cover import ShortCoverStopLossTask
        ctx = SimpleNamespace(
            config={"risk": {"short_cover_stop_pct": 0.15}},
            holdings={"X": _short(qty=-10, entry=100.0)},
            ohlcv={"X": _ohlcv(115.0)},  # +15% from entry = -15% for short
            orders=[], counters={},
        )
        ShortCoverStopLossTask().run(ctx)
        assert len(ctx.orders) == 1
        order = ctx.orders[0]
        assert order["ticker"] == "X"
        assert order["detail"] == "short_cover_stop"
        assert order["shares"] == 10  # POSITIVE (buy_to_close)
        assert order["decision_inputs"]["side"] == "buy_to_close"
        assert order["decision_inputs"]["loss_pct"] == 0.15
        assert order["decision_inputs"]["tax_holding_period"] == "ST_FORCED_§1233"
        assert ctx.counters["short_cover_stop_triggered"] == 1

    def test_short_within_threshold_no_cover(self):
        """Short at $100, current $110 (10% loss) — below 15% trigger.
        Should NOT fire."""
        from kernel.pipeline.task_short_cover import ShortCoverStopLossTask
        ctx = SimpleNamespace(
            config={"risk": {"short_cover_stop_pct": 0.15}},
            holdings={"X": _short(qty=-10, entry=100.0)},
            ohlcv={"X": _ohlcv(110.0)},
            orders=[], counters={},
        )
        ShortCoverStopLossTask().run(ctx)
        assert ctx.orders == []
        assert "short_cover_stop_triggered" not in ctx.counters

    def test_short_winning_no_cover(self):
        """Short at $100, current $80 (the short is +20% PROFIT) — no cover."""
        from kernel.pipeline.task_short_cover import ShortCoverStopLossTask
        ctx = SimpleNamespace(
            config={"risk": {"short_cover_stop_pct": 0.15}},
            holdings={"X": _short(qty=-10, entry=100.0)},
            ohlcv={"X": _ohlcv(80.0)},
            orders=[], counters={},
        )
        ShortCoverStopLossTask().run(ctx)
        assert ctx.orders == []

    def test_disabled_no_op(self):
        from kernel.pipeline.task_short_cover import ShortCoverStopLossTask
        ctx = SimpleNamespace(
            config={"risk": {"short_cover_stop_enabled": False}},
            holdings={"X": _short(qty=-10, entry=100.0)},
            ohlcv={"X": _ohlcv(120.0)},
            orders=[], counters={},
        )
        ShortCoverStopLossTask().run(ctx)
        assert ctx.orders == []

    def test_long_position_ignored(self):
        """A LONG position (qty > 0) in short_holdings (corrupted state)
        should be skipped, not generate a cover order."""
        from kernel.pipeline.task_short_cover import ShortCoverStopLossTask
        ctx = SimpleNamespace(
            config={"risk": {}},
            holdings={"X": _short(qty=+10, entry=100.0)},
            ohlcv={"X": _ohlcv(120.0)},
            orders=[], counters={},
        )
        ShortCoverStopLossTask().run(ctx)
        assert ctx.orders == []

    def test_multiple_shorts_partial_trigger(self):
        from kernel.pipeline.task_short_cover import ShortCoverStopLossTask
        ctx = SimpleNamespace(
            config={"risk": {"short_cover_stop_pct": 0.15}},
            holdings={
                "WIN":  _short(qty=-10, entry=100.0),  # 80 cur → +20% short profit
                "MEH":  _short(qty=-10, entry=100.0),  # 110 cur → -10% loss, no trig
                "LOSE": _short(qty=-10, entry=100.0),  # 120 cur → -20% loss, COVER
            },
            ohlcv={"WIN": _ohlcv(80.0), "MEH": _ohlcv(110.0), "LOSE": _ohlcv(120.0)},
            orders=[], counters={},
        )
        ShortCoverStopLossTask().run(ctx)
        assert len(ctx.orders) == 1
        assert ctx.orders[0]["ticker"] == "LOSE"

    def test_legacy_short_holdings_surface_still_supported(self):
        from kernel.pipeline.task_short_cover import ShortCoverStopLossTask
        ctx = SimpleNamespace(
            config={"risk": {"short_cover_stop_pct": 0.15}},
            holdings={},
            short_holdings={"X": SimpleNamespace(qty=-10, entry_price=100.0)},
            ohlcv={"X": _ohlcv(120.0)},
            orders=[], counters={},
        )
        ShortCoverStopLossTask().run(ctx)
        assert ctx.orders[0]["ticker"] == "X"

    def test_inference_pipeline_wires_short_cover_before_buy_scan(self):
        from kernel.pipeline.pp_inference import InferencePipeline

        src = inspect.getsource(InferencePipeline.run)
        assert "ShortCoverStopLossTask().run(ctx)" in src
        assert src.index("ShortCoverStopLossTask().run(ctx)") < src.index(
            "Phase 2b (buy scan)"
        )


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
