"""Regression guards for inference exit-parameter wiring.

The exit engine already implements these knobs; the pipeline must pass them
from regime config into ``compute_exits`` or the knobs are silently dead.
"""
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from kernel.pipeline.pp_inference import _build_exit_params  # noqa: E402


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
    assert params["lt_hold_gate_days"] == 330
    assert params["lt_hold_min_gain"] == 0.12
    assert params["lt_hold_threshold_days"] == 366
