"""Regression tests for P-BROKER-FILL-FRESHNESS (audit finding 9).

2026-06-02 daily decision-tree audit found ``monitor_state.last_fill_date``
was 2026-05-27 — strategy had been dormant (0 runner-driven broker fills)
for 5 trading days. That was nowhere in the preflight surface; the
operator only saw it by reading the state file.

This task adds a preflight check that surfaces the streak with three
verdicts (HARD PASS / SOFT WARN / HARD FAIL) gated by configurable
thresholds.
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY) not in sys.path:
    sys.path.insert(0, str(_STRATEGY))

from kernel.preflight_pipeline.ctx import PreflightContext  # noqa: E402
from kernel.preflight_pipeline.tasks.broker_fill_freshness import (  # noqa: E402
    BrokerFillFreshnessTask,
)


class _BrokerWithFills:
    """Fake broker exposing ``get_filled_orders``. Returns fills shaped
    like the umbrella ``live/alpaca_broker.py::get_filled_orders`` output."""

    def __init__(self, fills):
        self._fills = fills

    def get_filled_orders(self, after=None):
        return list(self._fills)


class _BrokerWithoutFills:
    """Fake broker with no ``get_filled_orders`` attr (sim broker shape)."""
    pass


class _RaisingBroker:
    def get_filled_orders(self, after=None):
        raise RuntimeError("broker API offline")


def _ctx(broker=None, cfg=None):
    return PreflightContext(
        config=cfg or {},
        strategy_dir=Path("/tmp"),
        broker=broker,
    )


def test_no_broker_dry_run_soft_pass():
    result = BrokerFillFreshnessTask().check(_ctx(broker=None))
    assert result.ok
    assert result.severity == "soft"
    assert "dry-run" in result.message.lower()


def test_broker_lacks_get_filled_orders_soft_pass():
    result = BrokerFillFreshnessTask().check(_ctx(broker=_BrokerWithoutFills()))
    assert result.ok
    assert result.severity == "soft"
    assert "get_filled_orders" in result.message


def test_broker_raises_soft_pass_no_fail_close():
    """Transient API errors must not fail the entire preflight."""
    result = BrokerFillFreshnessTask().check(_ctx(broker=_RaisingBroker()))
    assert result.ok
    assert result.severity == "soft"
    assert "broker.get_filled_orders failed" in result.message


def test_recent_fill_hard_pass():
    """Fill today → HARD PASS with 0 trading-day streak."""
    today = _dt.date.today()
    broker = _BrokerWithFills([
        {"order_id": "o1", "symbol": "AAPL", "action": "BUY",
         "filled_at": today.isoformat() + "T15:30:00+00:00"},
    ])
    result = BrokerFillFreshnessTask().check(_ctx(broker=broker))
    assert result.ok
    assert result.severity == "hard"
    assert "last runner fill" in result.message


def test_streak_above_warn_below_hard_soft_warn():
    """Build a fill 8 trading days ago; warn cap default = 5, hard = 20.
    Streak ≥ 5 < 20 → SOFT WARN."""
    today = _dt.date.today()
    # 14 calendar days = ~10 trading days (rough; depends on weekends)
    old = today - _dt.timedelta(days=14)
    broker = _BrokerWithFills([
        {"order_id": "o1", "symbol": "AAPL", "action": "SELL",
         "filled_at": old.isoformat() + "T15:30:00+00:00"},
    ])
    result = BrokerFillFreshnessTask().check(_ctx(broker=broker))
    # The streak should be > 5 (warn) but < 20 (hard) — soft warn
    assert result.severity == "soft"
    assert result.ok  # soft warn still "passes" per PreflightCheck contract
    assert "trading days" in result.message
    assert "warn cap" in result.message


def test_no_fills_in_window_hard_fail():
    """Empty fill history AND hard cap exceeded → HARD FAIL."""
    broker = _BrokerWithFills([])
    result = BrokerFillFreshnessTask().check(_ctx(broker=broker))
    # No fills → streak == 120 (capped at lookback) > hard cap 20
    assert not result.ok
    assert result.severity == "hard"
    assert "120 trading days" in result.message
    assert "hard cap" in result.message
    assert "Strategy is dormant" in result.message


def test_thresholds_configurable_via_cfg():
    """Operator can tune warn/hard caps via monitoring.fill_freshness_*."""
    today = _dt.date.today()
    old = today - _dt.timedelta(days=14)
    broker = _BrokerWithFills([
        {"order_id": "o1", "symbol": "AAPL", "action": "SELL",
         "filled_at": old.isoformat() + "T15:30:00+00:00"},
    ])
    cfg = {
        "monitoring": {
            # Make hard fire immediately — streak likely ≥ 5
            "fill_freshness_warn_after_trading_days": 1,
            "fill_freshness_hard_after_trading_days": 3,
        },
    }
    result = BrokerFillFreshnessTask().check(_ctx(broker=broker, cfg=cfg))
    # Streak ~10 trading days > 3 hard cap → HARD FAIL
    assert not result.ok
    assert result.severity == "hard"


def test_fill_filtered_out_if_bad_iso_string():
    """Malformed filled_at row is skipped (not parsed), but a good one wins."""
    today = _dt.date.today()
    broker = _BrokerWithFills([
        {"order_id": "bad", "filled_at": "not-an-iso-string"},
        {"order_id": "good", "filled_at": today.isoformat() + "T15:30:00+00:00",
         "symbol": "AAPL"},
    ])
    result = BrokerFillFreshnessTask().check(_ctx(broker=broker))
    assert result.ok
    assert result.severity == "hard"
