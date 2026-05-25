"""Regression guards for inference exit-parameter wiring.

The exit engine already implements these knobs; the pipeline must pass them
from regime config into ``compute_exits`` or the knobs are silently dead.
"""
from __future__ import annotations

import sys
from pathlib import Path
import datetime as dt


REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from kernel.exits import HoldingState, compute_exits  # noqa: E402
from kernel.pipeline.context import InferenceContext  # noqa: E402
from kernel.pipeline.context import TickerInferenceContext  # noqa: E402
from kernel.pipeline.task_sell import EvaluateExitsTask  # noqa: E402
from kernel.pipeline.pp_inference import _build_exit_params, _make_sell_tctx  # noqa: E402


def test_build_exit_params_threads_all_regime_exit_knobs():
    regime_params = {
        "trailing_stop_trigger_pct": 0.20,
        "trailing_stop_trail_pct": 0.18,
        "stop_loss_pct": 0.15,
        "stop_n_sigma": 2.0,
        "atr_n_multiplier": 3.0,
        "max_single_day_loss_pct": 0.10,
        "sdl_n_sigma": 2.5,
        "sdl_skip_if_unrealized_above": 0.08,
        "take_profit_pct": 0.35,
        "stop_decay_days": 90,
        "stop_decay_floor": 0.07,
        "max_hold_days": 500,
    }

    params = _build_exit_params(
        regime_params,
        {
            "consecutive_sell_signals": 4,
            "min_hold_days": 30,
            "lt_hold_gate_days": 330,
            "lt_hold_min_gain": 0.12,
            "tax": {"long_term_threshold_days": 366},
        },
    )

    for key, value in regime_params.items():
        assert params[key] == value, f"{key} must be wired into exit_params"

    assert params["consecutive_sell_signals"] == 4
    assert params["min_hold_days"] == 30
    assert params["min_hold_profit_days"] == 0
    assert params["min_hold_loss_days"] == 0
    assert params["lt_hold_gate_days"] == 330
    assert params["lt_hold_min_gain"] == 0.12
    assert params["lt_hold_threshold_days"] == 366


def test_build_exit_params_threads_profit_loss_min_hold_knobs():
    params = _build_exit_params(
        {"max_hold_days": 500},
        {
            "min_hold_days": 5,
            "min_hold_profit_days": 20,
            "min_hold_loss_days": 15,
        },
    )

    assert params["min_hold_days"] == 5
    assert params["min_hold_profit_days"] == 20
    assert params["min_hold_loss_days"] == 15


def test_model_sell_uses_profit_loss_min_hold_without_blocking_hard_stops():
    today = dt.date(2026, 5, 22)
    state = HoldingState(
        entry_price=100.0,
        entry_date=today - dt.timedelta(days=10),
        high_watermark=110.0,
        sell_streak=2,
    )
    params = {
        "consecutive_sell_signals": 3,
        "min_hold_days": 5,
        "min_hold_profit_days": 20,
        "min_hold_loss_days": 15,
        "stop_loss_pct": 0.05,
    }

    sig, updated = compute_exits(110.0, today, "sell", state, params)

    assert not sig.should_exit
    assert updated.sell_streak == 2

    hard_sig, _ = compute_exits(94.0, today, "sell", updated, params)

    assert hard_sig.should_exit
    assert hard_sig.exit_type == "stop_loss"


def test_evaluate_exits_task_stamps_core_exit_source_task():
    today = dt.date(2026, 5, 22)
    tctx = TickerInferenceContext(
        ticker="AAPL",
        ohlcv={},
        model=None,
        config={},
        today=today,
        regime="BULL_CALM",
        regime_params={},
        exit_params={"stop_loss_pct": 0.05},
        holding=HoldingState(
            entry_price=100.0,
            entry_date=today - dt.timedelta(days=40),
            high_watermark=105.0,
        ),
        price=94.0,
    )

    EvaluateExitsTask().run(tctx)

    assert tctx.exit_signal.should_exit
    assert tctx.exit_signal.exit_type == "stop_loss"
    assert tctx.exit_signal.source_job == "TickerSellJob"
    assert tctx.exit_signal.source_task == "EvaluateExitsTask"
    assert tctx.exit_signal.order_source == "TickerSellJob.EvaluateExitsTask"


def test_make_sell_tctx_anchors_max_hold_to_entry_regime():
    """Time exits follow the entry thesis; current-regime risk exits still adapt."""
    ctx = InferenceContext(
        config={
            "regime_params": {
                "BULL_CALM": {"max_hold_days": 500, "stop_loss_pct": 0.15},
                "CHOPPY": {"max_hold_days": 40, "stop_loss_pct": 0.08},
            },
        },
        today=dt.date(2026, 5, 1),
        regime="CHOPPY",
        holdings={
            "AAPL": HoldingState(
                entry_price=100.0,
                entry_date=dt.date(2026, 1, 1),
                high_watermark=120.0,
                entry_regime="BULL_CALM",
            ),
        },
        prices={"AAPL": 115.0},
    )

    tctx = _make_sell_tctx(ctx, "AAPL")

    assert tctx.exit_params["max_hold_days"] == 500
    assert tctx.exit_params["max_hold_anchor_regime"] == "BULL_CALM"
    assert tctx.exit_params["stop_loss_pct"] == 0.08


def test_make_sell_tctx_can_anchor_stop_loss_to_entry_regime_when_enabled():
    """Explicit A/B mode keeps BULL_CALM cumulative stop no tighter than entry."""
    ctx = InferenceContext(
        config={
            "risk": {
                "stop_loss_anchor_policy": {
                    "mode": "max_entry_current",
                    "entry_regimes": ["BULL_CALM"],
                },
            },
            "regime_params": {
                "BULL_CALM": {"max_hold_days": 500, "stop_loss_pct": 0.15},
                "CHOPPY": {"max_hold_days": 40, "stop_loss_pct": 0.08},
            },
        },
        today=dt.date(2026, 5, 1),
        regime="CHOPPY",
        holdings={
            "AAPL": HoldingState(
                entry_price=100.0,
                entry_date=dt.date(2026, 1, 1),
                high_watermark=120.0,
                entry_regime="BULL_CALM",
            ),
        },
        prices={"AAPL": 115.0},
    )

    tctx = _make_sell_tctx(ctx, "AAPL")

    assert tctx.exit_params["stop_loss_pct"] == 0.15
    assert tctx.exit_params["stop_loss_anchor_policy"] == "max_entry_current"
    assert tctx.exit_params["stop_loss_anchor_regime"] == "BULL_CALM"
    assert tctx.exit_params["stop_loss_current_regime"] == "CHOPPY"
    assert tctx.exit_params["stop_loss_current_pct"] == 0.08
    assert tctx.exit_params["stop_loss_entry_regime"] == "BULL_CALM"
    assert tctx.exit_params["stop_loss_entry_pct"] == 0.15


def test_make_sell_tctx_anchors_model_sell_min_hold_to_entry_thesis_horizon():
    """Model-sell is a soft exit; BULL_CALM 60d thesis should survive
    current-regime relabeling until its configured horizon has elapsed.
    """
    ctx = InferenceContext(
        config={
            "min_hold_days": 5,
            "risk": {
                "panel_exit": {
                    "min_holding_days_by_regime": {"BULL_CALM": 60},
                },
            },
            "regime_params": {
                "BULL_CALM": {"max_hold_days": 500, "stop_loss_pct": 0.15},
                "CHOPPY": {"max_hold_days": 40, "stop_loss_pct": 0.08},
            },
        },
        today=dt.date(2026, 5, 1),
        regime="CHOPPY",
        holdings={
            "AAPL": HoldingState(
                entry_price=100.0,
                entry_date=dt.date(2026, 4, 1),
                high_watermark=120.0,
                entry_regime="BULL_CALM",
            ),
        },
        prices={"AAPL": 115.0},
    )

    tctx = _make_sell_tctx(ctx, "AAPL")

    assert tctx.exit_params["min_hold_days"] == 60
    assert tctx.exit_params["soft_exit_min_hold_anchor_regime"] == "BULL_CALM"
    assert tctx.exit_params["soft_exit_min_hold_days"] == 60


def test_make_sell_tctx_rejects_unknown_stop_anchor_policy():
    ctx = InferenceContext(
        config={
            "risk": {"stop_loss_anchor_policy": {"mode": "mystery"}},
            "regime_params": {
                "BULL_CALM": {"stop_loss_pct": 0.15},
                "CHOPPY": {"stop_loss_pct": 0.08},
            },
        },
        today=dt.date(2026, 5, 1),
        regime="CHOPPY",
        holdings={
            "AAPL": HoldingState(
                entry_price=100.0,
                entry_date=dt.date(2026, 1, 1),
                high_watermark=120.0,
                entry_regime="BULL_CALM",
            ),
        },
        prices={"AAPL": 115.0},
    )

    import pytest

    with pytest.raises(ValueError, match="stop_loss_anchor_policy"):
        _make_sell_tctx(ctx, "AAPL")


def test_evaluate_exits_stamps_applied_exit_params_on_signal():
    ctx = InferenceContext(
        config={
            "regime_params": {
                "BULL_CALM": {"max_hold_days": 1, "stop_loss_pct": 0.15},
                "CHOPPY": {"max_hold_days": 40, "stop_loss_pct": 0.08},
            },
        },
        today=dt.date(2026, 1, 5),
        regime="CHOPPY",
        holdings={
            "AAPL": HoldingState(
                entry_price=100.0,
                entry_date=dt.date(2026, 1, 1),
                high_watermark=100.0,
                entry_regime="BULL_CALM",
            ),
        },
        prices={"AAPL": 100.0},
    )
    tctx = _make_sell_tctx(ctx, "AAPL")

    EvaluateExitsTask().run(tctx)

    assert tctx.exit_signal is not None
    assert tctx.exit_signal.exit_type == "max_hold"
    assert tctx.exit_signal.exit_params["max_hold_days"] == 1
    assert tctx.exit_signal.exit_params["max_hold_anchor_regime"] == "BULL_CALM"
    assert tctx.exit_signal.exit_params["stop_loss_pct"] == 0.08


def test_evaluate_exits_stamps_stop_anchor_params_on_signal():
    ctx = InferenceContext(
        config={
            "risk": {
                "stop_loss_anchor_policy": {
                    "mode": "max_entry_current",
                    "entry_regimes": ["BULL_CALM"],
                },
            },
            "regime_params": {
                "BULL_CALM": {"max_hold_days": 1, "stop_loss_pct": 0.15},
                "CHOPPY": {"max_hold_days": 40, "stop_loss_pct": 0.08},
            },
        },
        today=dt.date(2026, 1, 5),
        regime="CHOPPY",
        holdings={
            "AAPL": HoldingState(
                entry_price=100.0,
                entry_date=dt.date(2026, 1, 1),
                high_watermark=100.0,
                entry_regime="BULL_CALM",
            ),
        },
        prices={"AAPL": 100.0},
    )
    tctx = _make_sell_tctx(ctx, "AAPL")

    EvaluateExitsTask().run(tctx)

    assert tctx.exit_signal is not None
    assert tctx.exit_signal.exit_params["stop_loss_pct"] == 0.15
    assert tctx.exit_signal.exit_params["stop_loss_anchor_policy"] == "max_entry_current"
