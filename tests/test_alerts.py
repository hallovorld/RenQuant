from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def test_duplicate_same_key_suppresses_second_send(tmp_path, monkeypatch):
    from live.alerts import AlertEvent, post_ntfy_alert

    monkeypatch.delenv("RENQUANT_NO_NOTIFY", raising=False)
    monkeypatch.setenv("RENQUANT_NTFY_BACKOFF_SECONDS", "0")
    state_path = tmp_path / "alert_state.json"
    event = AlertEvent(
        taxonomy="DECISION",
        title="RenQuant decision",
        body="no trade",
        key="same-cycle",
        cooldown_seconds=3600,
    )

    with patch("urllib.request.urlopen", return_value=SimpleNamespace(read=lambda: b"ok")) as m:
        assert post_ntfy_alert("https://ntfy.sh/test", event, state_path=state_path)
        assert not post_ntfy_alert("https://ntfy.sh/test", event, state_path=state_path)

    assert m.call_count == 1


def test_different_key_sends(tmp_path, monkeypatch):
    from live.alerts import AlertEvent, post_ntfy_alert

    monkeypatch.delenv("RENQUANT_NO_NOTIFY", raising=False)
    monkeypatch.setenv("RENQUANT_NTFY_BACKOFF_SECONDS", "0")
    state_path = tmp_path / "alert_state.json"

    with patch("urllib.request.urlopen", return_value=SimpleNamespace(read=lambda: b"ok")) as m:
        assert post_ntfy_alert(
            "https://ntfy.sh/test",
            AlertEvent("DECISION", "A", "body", key="k1", cooldown_seconds=3600),
            state_path=state_path,
        )
        assert post_ntfy_alert(
            "https://ntfy.sh/test",
            AlertEvent("DECISION", "A", "body", key="k2", cooldown_seconds=3600),
            state_path=state_path,
        )

    assert m.call_count == 2


def test_force_trade_alerts_are_never_suppressed(tmp_path, monkeypatch):
    from live.alerts import AlertEvent, post_ntfy_alert

    monkeypatch.delenv("RENQUANT_NO_NOTIFY", raising=False)
    monkeypatch.setenv("RENQUANT_NTFY_BACKOFF_SECONDS", "0")
    state_path = tmp_path / "alert_state.json"
    event = AlertEvent(
        taxonomy="TRADE",
        title="RenQuant trade",
        body="BUY AAPL",
        key="same-trade-key",
        cooldown_seconds=3600,
        force=True,
    )

    with patch("urllib.request.urlopen", return_value=SimpleNamespace(read=lambda: b"ok")) as m:
        assert post_ntfy_alert("https://ntfy.sh/test", event, state_path=state_path)
        assert post_ntfy_alert("https://ntfy.sh/test", event, state_path=state_path)

    assert m.call_count == 2


def test_renquant_no_notify_prevents_network(tmp_path, monkeypatch):
    from live.alerts import AlertEvent, post_ntfy_alert

    monkeypatch.setenv("RENQUANT_NO_NOTIFY", "1")
    with patch("urllib.request.urlopen") as m:
        ok = post_ntfy_alert(
            "https://ntfy.sh/test",
            AlertEvent("DECISION", "A", "body", key="k", cooldown_seconds=3600),
            state_path=tmp_path / "alert_state.json",
        )

    assert not ok
    m.assert_not_called()
