"""Regression tests for trade-level WF acceptance gates."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from trade_monotonicity import evaluate_trade_monotonicity  # noqa: E402


def _round_trips(*, inverted: bool = False, n: int = 60) -> pd.DataFrame:
    scores = np.linspace(0.1, 0.9, n)
    pnl = np.linspace(-0.05, 0.08, n)
    if inverted:
        pnl = pnl[::-1]
    return pd.DataFrame({
        "status": ["closed"] * n,
        "entry_regime": ["BULL_CALM"] * n,
        "entry_rank_score": scores,
        "pnl_pct": pnl,
        "net_pnl_after_tax": pnl * 1000.0,
    })


def test_trade_monotonicity_passes_when_higher_scores_pay_more() -> None:
    report = evaluate_trade_monotonicity(
        _round_trips(inverted=False),
        min_n_per_regime=30,
        min_spearman=0.02,
    )
    assert report.passed is True
    assert report.regimes[0]["regime"] == "BULL_CALM"
    assert report.regimes[0]["spearman"] > 0.99
    assert report.regimes[0]["top_bottom_return_spread"] > 0


def test_trade_monotonicity_fails_when_rank_is_anti_predictive() -> None:
    report = evaluate_trade_monotonicity(
        _round_trips(inverted=True),
        min_n_per_regime=30,
        min_spearman=0.02,
    )
    assert report.passed is False
    assert "BULL_CALM" in report.reason
    assert report.regimes[0]["spearman"] < -0.99
    assert report.regimes[0]["top_bottom_return_spread"] < 0


def test_trade_monotonicity_fails_closed_for_tiny_regimes_by_default() -> None:
    report = evaluate_trade_monotonicity(
        _round_trips(inverted=True, n=10),
        min_n_per_regime=30,
    )
    assert report.passed is False
    assert "insufficient per-regime trade evidence" in report.reason
    assert report.regimes[0]["eligible"] is False


def test_trade_monotonicity_pass_open_is_explicit_diagnostic_only() -> None:
    report = evaluate_trade_monotonicity(
        _round_trips(inverted=True, n=10),
        min_n_per_regime=30,
        allow_pass_open=True,
    )
    assert report.passed is True
    assert "pass-open" in report.reason
    assert report.regimes[0]["eligible"] is False
