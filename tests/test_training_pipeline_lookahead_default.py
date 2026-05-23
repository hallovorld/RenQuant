"""Regression guard for per-ticker tournament lookahead default.

The production config removed ``model_params.lookahead`` to avoid conflicting
with panel-LTR's 60-day horizon.  The short-horizon per-ticker tournament still
needs its legacy 5-day default instead of silently producing no feature frames.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def test_tournament_lookahead_defaults_to_legacy_five_days():
    from kernel.pipeline.pp_training import _model_params_for_tournament

    mp = _model_params_for_tournament({
        "model_params": {"threshold": 0.03},
        "panel_ltr": {"lookahead_days": 60},
    })

    assert mp["lookahead"] == 5
    assert mp["threshold"] == 0.03


def test_tournament_lookahead_can_be_explicitly_overridden():
    from kernel.pipeline.pp_training import _model_params_for_tournament

    mp = _model_params_for_tournament({
        "model_params": {"threshold": 0.03},
        "training": {"tournament_lookahead_days": 10},
    })

    assert mp["lookahead"] == 10
