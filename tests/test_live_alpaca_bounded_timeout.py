"""The stack that trades must not hang on an account read.

`live/runner.py` does `from .alpaca_broker import AlpacaBroker`, so THIS module
is the order path -- renquant-execution#41's identical fix does not reach it
(the pair is a deliberate `diverged_pin`). The alpaca-py SDK exposes no timeout
knob: `RESTClient.__init__` has no `timeout` parameter, so every account read
inherits requests' default `timeout=None` and can hang on the OS TCP timeout.
That is the 2026-08-11 07:00 P-BROKER-CONNECT abort.

Every assertion below fails on the pre-fix module.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from live.alpaca_broker import (
    _BROKER_CONNECT_TIMEOUT_SECONDS,
    _BROKER_READ_TIMEOUT_SECONDS,
    AlpacaBroker,
)

EXPECTED = (_BROKER_CONNECT_TIMEOUT_SECONDS, _BROKER_READ_TIMEOUT_SECONDS)


class _Session:
    """Stand-in for the SDK's requests.Session, carrying transport state."""

    def __init__(self):
        self.calls = []
        # State a replace-the-session fix would silently drop.
        self.proxies = {"https": "http://corp-proxy:3128"}
        self.verify = "/etc/ssl/corp-ca.pem"
        self.headers = {"X-Seeded": "1"}

    def request(self, *args, **kwargs):
        self.calls.append(kwargs.get("timeout", "ABSENT"))
        return SimpleNamespace(status_code=200)


def _broker():
    b = AlpacaBroker.__new__(AlpacaBroker)
    session = _Session()
    b._trading_client = SimpleNamespace(_session=session)
    return b, session


def test_preflight_read_injects_a_bounded_timeout():
    b, session = _broker()
    with b._bounded_account_timeout():
        session.request("GET", "/v2/account")
    assert session.calls == [EXPECTED], (
        f"account read went out with timeout={session.calls!r}"
    )


def test_order_path_outside_the_context_stays_unbounded():
    """The whole point of scoping it: order submission is untouched."""
    b, session = _broker()
    with b._bounded_account_timeout():
        session.request("GET", "/v2/account")
    session.request("POST", "/v2/orders")          # order submission
    assert session.calls == [EXPECTED, "ABSENT"], (
        "order submission must not inherit the preflight timeout"
    )


def test_session_object_and_its_transport_state_survive():
    """Wrap, not replace -- a fresh session would drop proxies/verify/headers."""
    b, session = _broker()
    before = (id(session), dict(session.proxies), session.verify, dict(session.headers))
    with b._bounded_account_timeout():
        pass
    after = (
        id(b._trading_client._session), dict(session.proxies),
        session.verify, dict(session.headers),
    )
    assert before == after


def test_restoration_leaves_no_shadowing_instance_attr():
    """`request` is a CLASS method here, so teardown must `del`, not assign.

    Asserting on identity would be wrong: a bound method is rebuilt on every
    attribute access, so `session.request is original` fails even when the
    restore is correct. `vars()` is the honest probe.
    """
    b, session = _broker()
    assert "request" not in vars(session), "precondition: class-level method"
    with b._bounded_account_timeout():
        assert "request" in vars(session), "wrapper should shadow during the read"
    assert "request" not in vars(session), "temporary wrapper leaked"
    session.request("POST", "/v2/orders")
    assert session.calls == ["ABSENT"], "restored request must be unbounded"


def test_restoration_preserves_a_pre_existing_instance_attr():
    """The other branch: if the SDK already set an instance-level `request`,
    teardown must put THAT back rather than deleting it and exposing the class
    method."""
    b, session = _broker()
    sentinel_calls = []

    def _own(*args, **kwargs):
        sentinel_calls.append(kwargs.get("timeout", "ABSENT"))

    session.request = _own                      # now an instance attribute
    with b._bounded_account_timeout():
        session.request("GET", "/v2/account")
    assert session.request is _own, "pre-existing instance attr was not restored"
    assert sentinel_calls == [EXPECTED], "wrapper must delegate to the original"
    session.request("POST", "/v2/orders")
    assert sentinel_calls == [EXPECTED, "ABSENT"]


def test_restores_even_when_the_body_raises():
    b, session = _broker()
    with pytest.raises(ValueError):
        with b._bounded_account_timeout():
            raise ValueError("read blew up")
    assert "request" not in vars(session), "wrapper leaked after an exception"


def test_caller_supplied_timeout_is_respected():
    b, session = _broker()
    with b._bounded_account_timeout():
        session.request("GET", "/v2/account", timeout=(1.0, 2.0))
    assert session.calls == [(1.0, 2.0)]


@pytest.mark.parametrize(
    "client",
    [SimpleNamespace(), SimpleNamespace(_session=None),
     SimpleNamespace(_session=SimpleNamespace(request="not-callable"))],
    ids=["no_session_attr", "session_is_none", "request_not_callable"],
)
def test_unusable_session_raises_instead_of_degrading(client):
    """A silent unbounded fallback would defeat the fast-fail contract."""
    b = AlpacaBroker.__new__(AlpacaBroker)
    b._trading_client = client
    with pytest.raises(RuntimeError, match="bounded account-read timeout"):
        with b._bounded_account_timeout():
            pass


def test_the_sdk_still_offers_no_timeout_knob():
    """If this ever fails, the SDK gained a timeout and the wrap can retire."""
    import inspect

    from alpaca.common.rest import RESTClient

    assert "timeout" not in inspect.signature(RESTClient.__init__).parameters
