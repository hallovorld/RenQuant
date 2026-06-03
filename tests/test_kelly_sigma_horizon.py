"""Kelly sigma-horizon rescale regression tests."""
from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))


@dataclass
class _Cand:
    ticker: str
    mu: float = 0.005
    sigma: float = 0.35
    kelly_target_pct: float | None = None


@dataclass
class _Hold:
    ticker: str
    mu: float = 0.005
    sigma: float = 0.35
    kelly_target_pct: float | None = None


def _ctx(*, sigma_horizon_days=None):
    kelly_cfg = {
        "enabled": True,
        "fractional": 0.5,
        "max_concentration": 1.0,
        "min_edge": 0.0,
    }
    if sigma_horizon_days is not None:
        kelly_cfg["sigma_horizon_days"] = sigma_horizon_days
    return types.SimpleNamespace(
        candidates=[_Cand("AAPL")],
        holdings={"MSFT": _Hold("MSFT")},
        config={
            "ranking": {"kelly_sizing": kelly_cfg},
            "regime_params": {"BULL_CALM": {"max_position_pct": 1.0}},
        },
        regime="BULL_CALM",
        confidence=1.0,
        counters={},
    )


def _run(ctx):
    from kernel.panel_pipeline.job_panel_scoring import ApplyKellySizingTask

    ApplyKellySizingTask().run(ctx)
    return ctx.candidates[0].kelly_target_pct, ctx.holdings["MSFT"].kelly_target_pct


def test_default_sigma_horizon_preserves_annualized_sigma_behavior() -> None:
    implicit_ctx = _ctx()
    explicit_ctx = _ctx(sigma_horizon_days=252)

    implicit_cand, implicit_hold = _run(implicit_ctx)
    explicit_cand, explicit_hold = _run(explicit_ctx)

    expected = 0.5 * (0.005 / (0.35**2))
    assert implicit_cand == pytest.approx(expected)
    assert implicit_hold == pytest.approx(expected)
    assert explicit_cand == pytest.approx(implicit_cand)
    assert explicit_hold == pytest.approx(implicit_hold)


def test_sigma_horizon_60_matches_mu_period_before_kelly_formula() -> None:
    legacy_ctx = _ctx(sigma_horizon_days=252)
    matched_ctx = _ctx(sigma_horizon_days=60)

    legacy_target, _ = _run(legacy_ctx)
    matched_target, matched_hold_target = _run(matched_ctx)

    expected_sigma_60d = 0.35 * math.sqrt(60.0 / 252.0)
    expected = 0.5 * (0.005 / (expected_sigma_60d**2))
    assert matched_target == pytest.approx(expected)
    assert matched_hold_target == pytest.approx(expected)
    assert matched_target == pytest.approx(legacy_target * (252.0 / 60.0))


def test_invalid_sigma_horizon_fails_closed_with_skip_reason() -> None:
    ctx = _ctx(sigma_horizon_days=0)

    cand_target, hold_target = _run(ctx)

    assert cand_target == 0.0
    assert hold_target == 0.0
    assert ctx._blocked_by_ticker["AAPL"] == "kelly_zero:sigma_horizon_invalid"
