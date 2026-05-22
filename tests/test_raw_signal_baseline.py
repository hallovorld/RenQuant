from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parent.parent
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from eval_raw_signal_baseline import (  # noqa: E402
    after_tax_return,
    apply_score_control,
    cross_sectional_ic_by_date,
    event_study_topk,
    summarize_events,
)


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=5)
    return pd.DataFrame(
        {
            "A": [100.0, 101.0, 120.0, 121.0, 122.0],
            "B": [100.0, 99.0, 90.0, 91.0, 92.0],
            "C": [100.0, 100.0, 110.0, 111.0, 112.0],
            "SPY": [100.0, 101.0, 105.0, 106.0, 107.0],
        },
        index=dates,
    )


def _score_frame() -> pd.DataFrame:
    date = pd.Timestamp("2024-01-02")
    return pd.DataFrame(
        {
            "date": [date, date, date],
            "ticker": ["A", "B", "C"],
            "score": [0.9, -0.8, 0.1],
            "regime": ["HIGH_CALM", "HIGH_CALM", "HIGH_CALM"],
        }
    )


def test_event_study_topk_selects_ranked_winner_and_bottom() -> None:
    events = event_study_topk(
        _score_frame(),
        _prices(),
        spy_col="SPY",
        hold_days=2,
        top_k=1,
        bottom_k=1,
        tax_rate=0.50,
    )

    assert len(events) == 1
    row = events.iloc[0]
    assert row["top_tickers"] == ["A"]
    assert row["bottom_tickers"] == ["B"]
    assert row["top_return"] == pytest.approx(0.20)
    assert row["bottom_return"] == pytest.approx(-0.10)
    assert row["spy_return"] == pytest.approx(0.05)
    assert row["alpha_vs_spy"] == pytest.approx(0.15)
    assert row["top_after_tax_return"] == pytest.approx(0.10)
    assert row["regime"] == "HIGH_CALM"


def test_reverse_control_turns_winner_into_loser() -> None:
    reversed_scores = apply_score_control(
        _score_frame(),
        control="reverse",
        seed=1,
        shift_days=1,
    )
    events = event_study_topk(
        reversed_scores,
        _prices(),
        spy_col="SPY",
        hold_days=2,
        top_k=1,
        bottom_k=1,
        tax_rate=0.50,
    )

    assert events.iloc[0]["top_tickers"] == ["B"]
    assert events.iloc[0]["top_return"] == pytest.approx(-0.10)
    assert events.iloc[0]["long_short"] == pytest.approx(-0.30)


def test_time_shift_control_uses_stale_per_ticker_scores() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    frame = pd.DataFrame(
        {
            "date": [dates[0], dates[1], dates[2], dates[0], dates[1], dates[2]],
            "ticker": ["A", "A", "A", "B", "B", "B"],
            "score": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "regime": ["R", "R", "R", "R", "R", "R"],
        }
    )

    shifted = apply_score_control(frame, control="time_shift", seed=1, shift_days=1)

    got = shifted.sort_values(["ticker", "date"])[["ticker", "date", "score"]].reset_index(drop=True)
    assert got["score"].tolist() == [1.0, 2.0, 10.0, 20.0]
    assert got["date"].tolist() == [dates[1], dates[2], dates[1], dates[2]]


def test_after_tax_return_only_taxes_positive_returns() -> None:
    assert after_tax_return(0.20, 0.50) == pytest.approx(0.10)
    assert after_tax_return(-0.20, 0.50) == pytest.approx(-0.20)
    assert after_tax_return(0.0, 0.50) == pytest.approx(0.0)


def test_summarize_events_is_regime_first_with_pooled_secondary() -> None:
    events = event_study_topk(
        _score_frame(),
        _prices(),
        spy_col="SPY",
        hold_days=2,
        top_k=1,
        bottom_k=1,
        tax_rate=0.50,
    )
    summary = summarize_events(events, period_days=2)

    assert list(summary.keys()) == ["per_regime", "pooled"]
    assert "HIGH_CALM" in summary["per_regime"]
    assert summary["per_regime"]["HIGH_CALM"]["n"] == 1
    assert summary["pooled"]["n"] == 1


def test_summarize_events_empty_keeps_full_schema() -> None:
    summary = summarize_events(pd.DataFrame(), period_days=60)

    assert list(summary.keys()) == ["per_regime", "pooled"]
    assert summary["per_regime"] == {}
    assert summary["pooled"]["n"] == 0
    assert "mean_alpha_vs_spy" in summary["pooled"]
    assert "mean_long_short" in summary["pooled"]


def test_cross_sectional_ic_by_date_uses_rank_correlation() -> None:
    date = pd.Timestamp("2024-01-02")
    scores = pd.DataFrame(
        {
            "date": [date] * 5,
            "ticker": list("ABCDE"),
            "score": [1, 2, 3, 4, 5],
            "regime": ["HIGH_CALM"] * 5,
        }
    )
    labels = pd.DataFrame(
        {
            "date": [date] * 5,
            "ticker": list("ABCDE"),
            "fwd_60d_excess": [10, 20, 30, 40, 50],
        }
    )

    ic = cross_sectional_ic_by_date(scores, labels, label_col="fwd_60d_excess")

    assert len(ic) == 1
    assert ic.iloc[0]["ic"] == pytest.approx(1.0)
    assert ic.iloc[0]["regime"] == "HIGH_CALM"
