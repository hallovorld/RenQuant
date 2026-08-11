"""P-BROKER-CONNECT bounded retry, fail-closed preserved.

Both the runtime path (``BrokerConnectTask``) and the legacy bridge
(``_check_broker_connect``) share one ``_attempt_broker_connect`` body; these
tests cover both so the twin implementations cannot drift.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from kernel import preflight
from kernel.preflight import _attempt_broker_connect, _check_broker_connect
from kernel.preflight_pipeline.tasks.broker import BrokerConnectTask


class _FlakyBroker:
    """Fail the first ``fail_times`` attempts, then return an account value."""

    def __init__(self, fail_times: int, *, equity: float = 10000.0, where: str = "connect"):
        self.fail_times = fail_times
        self.equity = equity
        self.where = where
        self.attempts = 0

    def connect(self):
        self.attempts += 1
        if self.where == "connect" and self.attempts <= self.fail_times:
            raise ConnectionError("Read timed out. (read timeout=None)")

    def get_account_value(self):
        if self.where == "account" and self.attempts <= self.fail_times:
            raise ConnectionError("Read timed out. (read timeout=None)")
        return self.equity


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Record backoff sleeps instead of waiting in deterministic unit tests."""
    sleeps: list[float] = []
    monkeypatch.setattr(preflight.time, "sleep", lambda seconds: sleeps.append(seconds))
    return sleeps


def test_success_first_attempt_has_no_count_suffix(_no_real_sleep):
    broker = _FlakyBroker(fail_times=0)
    result = _check_broker_connect(broker)
    assert result.severity == "hard" and result.ok is True
    assert result.message == "broker connected, equity=$10000.00"
    assert broker.attempts == 1
    assert _no_real_sleep == []


def test_succeeds_after_transient_failures_names_attempt_count(_no_real_sleep):
    broker = _FlakyBroker(fail_times=2)
    result = _check_broker_connect(broker, max_attempts=3, backoff_seconds=2.0)
    assert result.severity == "hard" and result.ok is True
    assert "after 3 attempts" in result.message
    assert "equity=$10000.00" in result.message
    assert broker.attempts == 3
    assert _no_real_sleep == [2.0, 2.0]


def test_get_account_value_failure_is_also_retried(_no_real_sleep):
    broker = _FlakyBroker(fail_times=1, where="account")
    result = _check_broker_connect(broker, max_attempts=3)
    assert result.ok is True
    assert "after 2 attempts" in result.message
    assert broker.attempts == 2


def test_fails_closed_after_all_attempts_exhausted(_no_real_sleep):
    broker = _FlakyBroker(fail_times=999)
    result = _check_broker_connect(broker, max_attempts=3, backoff_seconds=2.0)
    assert result.severity == "hard" and result.ok is False
    assert "broker connect failed after 3 attempts" in result.message
    assert "Read timed out" in result.message
    assert broker.attempts == 3
    assert _no_real_sleep == [2.0, 2.0]


def test_none_broker_is_soft_skip(_no_real_sleep):
    result = _check_broker_connect(None)
    assert result.severity == "soft" and result.ok is True
    assert "dry-run" in result.message


def test_runtime_task_recovers_on_bounded_retry(_no_real_sleep):
    broker = _FlakyBroker(fail_times=2)
    result = BrokerConnectTask().check(SimpleNamespace(broker=broker))
    assert result.name == "P-BROKER-CONNECT"
    assert result.severity == "hard" and result.ok is True
    assert "after 3 attempts" in result.message
    assert broker.attempts == 3


def test_runtime_task_fails_closed_when_outage_persists(_no_real_sleep):
    broker = _FlakyBroker(fail_times=999)
    result = BrokerConnectTask().check(SimpleNamespace(broker=broker))
    assert result.severity == "hard" and result.ok is False
    assert "broker connect failed after 3 attempts" in result.message
    assert broker.attempts == 3


def test_both_entry_points_route_through_the_shared_body():
    assert "_attempt_broker_connect" in BrokerConnectTask.check.__code__.co_names
    result = _attempt_broker_connect(_FlakyBroker(fail_times=0))
    assert result.ok is True and result.name == "P-BROKER-CONNECT"
