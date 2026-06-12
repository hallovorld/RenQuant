"""G1 (2026-06-12): live.broker_side_stops.pct overrides the per-regime
intraday cap for broker-resident stops — the dead-box catastrophe line."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from adapters.runner import RunnerAdapter  # noqa: E402


def _ctx(cfg, regime="BULL_CALM"):
    return SimpleNamespace(config=cfg, regime=regime)


def test_catastrophe_pct_overrides_regime_cap():
    cfg = {"live": {"broker_side_stops": {"enabled": True, "pct": 0.20}},
           "regime_params": {"BULL_CALM": {"max_single_day_loss_pct": 0.06}}}
    assert RunnerAdapter._z9_stop_pct(_ctx(cfg)) == 0.20


def test_absent_key_keeps_legacy_behavior():
    cfg = {"live": {"broker_side_stops": {"enabled": True}},
           "regime_params": {"BULL_CALM": {"max_single_day_loss_pct": 0.06}}}
    assert RunnerAdapter._z9_stop_pct(_ctx(cfg)) == 0.06


def test_invalid_pct_falls_back():
    for bad in ("x", -0.2, 0, 1.5, None):
        cfg = {"live": {"broker_side_stops": {"pct": bad}},
               "regime_params": {"BULL_CALM": {"max_single_day_loss_pct": 0.08}}}
        assert RunnerAdapter._z9_stop_pct(_ctx(cfg)) == 0.08


def test_sigma_mode_zero_regime_cap_still_safe():
    # BULL_CALM sets max_single_day_loss_pct=0 (sigma mode); legacy path
    # returns 0 and the place-site clamps to 0.06 — with the catastrophe pct
    # set, we never hit that path at all.
    cfg = {"live": {"broker_side_stops": {"pct": 0.20}},
           "regime_params": {"BULL_CALM": {"max_single_day_loss_pct": 0}}}
    assert RunnerAdapter._z9_stop_pct(_ctx(cfg)) == 0.20
