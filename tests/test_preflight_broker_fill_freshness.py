"""Regression tests for P-BROKER-FILL-FRESHNESS (audit finding 9 + codex PR #84).

The check distinguishes runner-driven activity from any-source broker
fills. Codex PR #84 review caught that the first iteration was reading
`broker.get_filled_orders()` (any source) — a manual close or Z9-only
fill would falsely reset the streak.

The fix reads ``monitor_state.last_activity_date`` from the persisted
state file (set by ``MonitorIdleStreakTask`` from ``ctx.orders`` /
``ctx.exits`` — runner emissions only). The companion ``runner.py``
change stops the previous override from clobbering this field with
broker-truth fills.

These tests pin BOTH the happy path and the codex regression: a
broker-side external fill must NOT reset the streak when no runner
activity has happened.
"""
from __future__ import annotations

import datetime as _dt
import json
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


def _write_state(tmp_path, broker_name, monitor_state, **extra):
    """Write a minimal live_state.{broker}.json that matches the canonical
    path resolved by ``kernel.state_paths.resolve_live_state_read``."""
    state_path = tmp_path / f"live_state.{broker_name}.json"
    state = {"monitor_state": monitor_state, **extra}
    state_path.write_text(json.dumps(state))
    return state_path


def _ctx(tmp_path, broker_name="alpaca", cfg=None):
    return PreflightContext(
        config=cfg or {},
        strategy_dir=tmp_path,
        broker=object(),  # presence required only when broker_name truthy
        broker_name=broker_name,
    )


# ── soft-pass paths (no fail-close in degenerate contexts) ────────────────

def test_no_broker_name_dry_run_soft_pass(tmp_path):
    ctx = PreflightContext(config={}, strategy_dir=tmp_path, broker_name=None)
    result = BrokerFillFreshnessTask().check(ctx)
    assert result.ok
    assert result.severity == "soft"
    assert "dry-run" in result.message.lower()


def test_state_file_absent_soft_pass(tmp_path):
    """First-run scenario — no state file exists yet."""
    result = BrokerFillFreshnessTask().check(_ctx(tmp_path))
    assert result.ok
    assert result.severity == "soft"
    assert "absent" in result.message.lower()


def test_state_file_unparseable_soft_pass(tmp_path):
    state_path = tmp_path / "live_state.alpaca.json"
    state_path.write_text("{not valid json")
    result = BrokerFillFreshnessTask().check(_ctx(tmp_path))
    assert result.ok
    assert result.severity == "soft"
    assert "unparseable" in result.message.lower()


# ── happy + warn + hard paths ─────────────────────────────────────────────

def test_recent_runner_activity_hard_pass(tmp_path):
    """Runner activity today → HARD PASS, 0-streak."""
    today = _dt.date.today()
    _write_state(tmp_path, "alpaca", {
        "last_activity_date": today.isoformat(),
        "first_trade_date": "2026-04-01",
    })
    result = BrokerFillFreshnessTask().check(_ctx(tmp_path))
    assert result.ok
    assert result.severity == "hard"
    assert "last runner-driven activity" in result.message


def test_streak_between_warn_and_hard_soft_warn(tmp_path):
    """Activity 14 days ago (~10 trading) — warn=5 hard=20 → SOFT WARN."""
    today = _dt.date.today()
    old = today - _dt.timedelta(days=14)
    _write_state(tmp_path, "alpaca", {
        "last_activity_date": old.isoformat(),
        "first_trade_date": "2026-04-01",
    })
    result = BrokerFillFreshnessTask().check(_ctx(tmp_path))
    assert result.severity == "soft"
    assert result.ok
    assert "trading days" in result.message
    assert "warn cap" in result.message


def test_no_runner_activity_recorded_hard_fail(tmp_path):
    """Empty monitor_state → strategy has never traded → HARD FAIL."""
    _write_state(tmp_path, "alpaca", {})  # no last_activity_date, no first_trade_date
    result = BrokerFillFreshnessTask().check(_ctx(tmp_path))
    assert not result.ok
    assert result.severity == "hard"
    assert "never emitted" in result.message.lower() or "no runner-driven" in result.message.lower()


def test_streak_above_hard_fails(tmp_path):
    """Last runner activity 40 days ago (~28 trading) > hard cap 20 → HARD FAIL."""
    today = _dt.date.today()
    old = today - _dt.timedelta(days=40)
    _write_state(tmp_path, "alpaca", {
        "last_activity_date": old.isoformat(),
        "first_trade_date": "2026-04-01",
    })
    result = BrokerFillFreshnessTask().check(_ctx(tmp_path))
    assert not result.ok
    assert result.severity == "hard"
    assert "Strategy is dormant" in result.message
    assert "hard cap" in result.message


def test_thresholds_configurable_via_cfg(tmp_path):
    """Operator can tighten the hard cap via monitoring config."""
    today = _dt.date.today()
    old = today - _dt.timedelta(days=14)  # ~10 trading days
    _write_state(tmp_path, "alpaca", {
        "last_activity_date": old.isoformat(),
        "first_trade_date": "2026-04-01",
    })
    cfg = {
        "monitoring": {
            "fill_freshness_warn_after_trading_days": 1,
            "fill_freshness_hard_after_trading_days": 3,
        },
    }
    result = BrokerFillFreshnessTask().check(_ctx(tmp_path, cfg=cfg))
    assert not result.ok
    assert result.severity == "hard"


# ── codex PR #84 regression: external fill does NOT reset the streak ──────

def test_external_fill_yesterday_runner_stale_still_warns(tmp_path):
    """The exact scenario codex flagged in PR #84.

    Broker has a fresh ``last_fill_date`` (e.g. external/manual close
    or Z9 stop fired yesterday) — but ``last_activity_date`` is stale.
    The check MUST surface the runner-driven streak, NOT pass on the
    broker any-source field.
    """
    today = _dt.date.today()
    runner_old = today - _dt.timedelta(days=40)
    _write_state(tmp_path, "alpaca", {
        # Broker-truth field is fresh — would PASS under the old impl
        "last_fill_date": today.isoformat(),
        # Runner-truth field is stale — what we should actually read
        "last_activity_date": runner_old.isoformat(),
        "first_trade_date": "2026-04-01",
        "no_trade_streak": 0,            # broker source — irrelevant
        "no_trade_streak_source": "broker_filled_orders",
    })
    result = BrokerFillFreshnessTask().check(_ctx(tmp_path))
    assert not result.ok, (
        "external/Z9 fill must NOT reset runner-driven streak; "
        f"got {result}"
    )
    assert result.severity == "hard"
    assert "dormant" in result.message.lower()


def test_falls_back_to_first_trade_date_when_last_activity_missing(tmp_path):
    """If ``last_activity_date`` is absent (e.g. older state file shape),
    fall back to ``first_trade_date`` rather than fail-closing."""
    today = _dt.date.today()
    first = today - _dt.timedelta(days=2)
    _write_state(tmp_path, "alpaca", {
        "first_trade_date": first.isoformat(),
        # No last_activity_date
    })
    result = BrokerFillFreshnessTask().check(_ctx(tmp_path))
    # 2 calendar days ≈ 0-2 trading days < warn cap 5 → HARD PASS
    assert result.ok
    assert result.severity == "hard"


def test_unparseable_activity_date_soft_pass(tmp_path):
    """Defensive: bad ISO string → soft pass, don't fail-close."""
    _write_state(tmp_path, "alpaca", {
        "last_activity_date": "not-a-real-date",
    })
    result = BrokerFillFreshnessTask().check(_ctx(tmp_path))
    assert result.ok
    assert result.severity == "soft"
    assert "unparseable" in result.message.lower()
