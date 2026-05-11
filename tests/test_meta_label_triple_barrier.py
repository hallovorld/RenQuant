"""TDD — Triple-barrier label generator.

Faithful port of López de Prado AFML 2018 ch.3 Snippet 3.4 (pp. 47-49)
``applyPtSlOnT1``, adapted for the meta-label-on-exit use case per
ch.20.

Algorithm pins:
  * Upper barrier:  entry_price × (1 + pt_mult × σ_daily)
  * Lower barrier:  entry_price × (1 - sl_mult × σ_daily)
  * Vertical:       entry_date + max_horizon_days (business days)
  * Label = +1 if upper hit first
  * Label = -1 if lower hit first
  * Label =  0 if neither hit before vertical (then optionally sign of
            terminal return)

For meta-labeling-on-EXIT, the "exit was profitable to make NOW vs
hold" decision is encoded as:
  * meta_label = 1 if lower hit first (the position kept falling →
                 the exit was correct)
  * meta_label = 0 if upper hit first OR vertical with positive
                 terminal return (the position recovered → exit was
                 a false positive; we should have held)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.meta_label.triple_barrier import (  # noqa: E402
    apply_triple_barrier,
    meta_label_for_exit_event,
)


def _close_series(prices: list[float], start: str = "2025-01-01") -> pd.Series:
    idx = pd.bdate_range(start=start, periods=len(prices))
    return pd.Series(prices, index=idx, name="close")


class TestApplyTripleBarrierAFML:
    """AFML ch.3 Snippet 3.4 contract pins."""

    def test_upper_barrier_hit_returns_plus_one(self):
        # Prices climb +1% per day → reaches +10% on bar 10 → upper hit
        s = _close_series([100.0 * (1.01 ** i) for i in range(30)])
        label, exit_date, exit_price = apply_triple_barrier(
            s, entry_idx=s.index[0], entry_price=100.0,
            pt_mult=10.0, sl_mult=10.0, sigma_daily=0.01,  # ±10% bands
            max_horizon_days=20,
        )
        assert label == +1
        assert exit_price >= 110.0

    def test_lower_barrier_hit_returns_minus_one(self):
        # Prices fall -1% per day → reaches -10% on bar 10 → lower hit
        s = _close_series([100.0 * (0.99 ** i) for i in range(30)])
        label, exit_date, exit_price = apply_triple_barrier(
            s, entry_idx=s.index[0], entry_price=100.0,
            pt_mult=10.0, sl_mult=10.0, sigma_daily=0.01,
            max_horizon_days=20,
        )
        assert label == -1
        assert exit_price <= 90.0

    def test_neither_barrier_hit_returns_terminal_sign(self):
        # Slow climb: +0.1%/day for 20 days → ~+2% at vertical, never hits
        # ±10% bands.
        s = _close_series([100.0 * (1.001 ** i) for i in range(30)])
        label, exit_date, exit_price = apply_triple_barrier(
            s, entry_idx=s.index[0], entry_price=100.0,
            pt_mult=10.0, sl_mult=10.0, sigma_daily=0.01,
            max_horizon_days=20,
        )
        # Vertical hit with positive terminal → label = 0 (or +1 if
        # ``return_terminal_sign`` set). Default 0.
        assert label == 0

    def test_terminal_sign_when_requested(self):
        s = _close_series([100.0 * (1.001 ** i) for i in range(30)])
        label, _, _ = apply_triple_barrier(
            s, entry_idx=s.index[0], entry_price=100.0,
            pt_mult=10.0, sl_mult=10.0, sigma_daily=0.01,
            max_horizon_days=20, return_terminal_sign=True,
        )
        # Returns +1 because terminal price is positive vs entry
        assert label == +1

    def test_upper_hits_first_when_both_would_hit(self):
        # Bumpy path: +5% then -15% (would hit lower) but upper at +5%
        # would have already fired.
        prices = [100.0, 102.0, 105.0, 95.0, 85.0]
        s = _close_series(prices)
        label, exit_date, exit_price = apply_triple_barrier(
            s, entry_idx=s.index[0], entry_price=100.0,
            pt_mult=5.0, sl_mult=5.0, sigma_daily=0.01,  # ±5% bands
            max_horizon_days=10,
        )
        assert label == +1
        assert exit_date == s.index[2]
        assert exit_price == 105.0

    def test_horizon_truncates_at_series_end(self):
        # Only 3 future bars; max_horizon_days=20 should not crash.
        s = _close_series([100.0, 101.0, 102.0])
        label, _, _ = apply_triple_barrier(
            s, entry_idx=s.index[0], entry_price=100.0,
            pt_mult=10.0, sl_mult=10.0, sigma_daily=0.01,
            max_horizon_days=20,
        )
        # Vertical (effectively last bar) — neither barrier hit
        assert label == 0

    def test_returns_none_when_entry_date_at_end(self):
        # No future data at all
        s = _close_series([100.0])
        result = apply_triple_barrier(
            s, entry_idx=s.index[0], entry_price=100.0,
            pt_mult=10.0, sl_mult=10.0, sigma_daily=0.01,
            max_horizon_days=20,
        )
        assert result is None

    def test_returns_none_when_entry_not_in_series(self):
        s = _close_series([100.0, 101.0, 102.0])
        result = apply_triple_barrier(
            s, entry_idx=pd.Timestamp("2099-01-01"),
            entry_price=100.0,
            pt_mult=10.0, sl_mult=10.0, sigma_daily=0.01,
            max_horizon_days=20,
        )
        assert result is None


class TestMetaLabelForExitEvent:
    """Meta-labeling adaptation per AFML ch.20:
    label = 1 if the path-rule exit was CORRECT (position kept falling);
    label = 0 if the exit was a FALSE POSITIVE (would have recovered).
    """

    def test_continued_loss_labels_exit_as_correct(self):
        # Price keeps falling — exit was the right call
        s = _close_series([100.0 * (0.99 ** i) for i in range(30)])
        label = meta_label_for_exit_event(
            s, event_idx=s.index[0], event_price=100.0,
            sigma_daily=0.01, fwd_window=20,
            pt_mult=10.0, sl_mult=10.0,
        )
        assert label == 1

    def test_recovery_labels_exit_as_false_positive(self):
        # Price recovers — exit was a false positive
        s = _close_series([100.0 * (1.01 ** i) for i in range(30)])
        label = meta_label_for_exit_event(
            s, event_idx=s.index[0], event_price=100.0,
            sigma_daily=0.01, fwd_window=20,
            pt_mult=10.0, sl_mult=10.0,
        )
        assert label == 0

    def test_flat_path_labels_zero_no_strong_signal(self):
        # Slight positive drift, neither barrier hit → "exit was wrong
        # (should have held)" per the asymmetric default
        s = _close_series([100.0 * (1.0005 ** i) for i in range(30)])
        label = meta_label_for_exit_event(
            s, event_idx=s.index[0], event_price=100.0,
            sigma_daily=0.01, fwd_window=20,
            pt_mult=10.0, sl_mult=10.0,
        )
        # Terminal positive → label = 0 (would have recovered)
        assert label == 0

    def test_no_future_data_returns_none(self):
        s = _close_series([100.0])
        label = meta_label_for_exit_event(
            s, event_idx=s.index[0], event_price=100.0,
            sigma_daily=0.01, fwd_window=20,
            pt_mult=10.0, sl_mult=10.0,
        )
        assert label is None
