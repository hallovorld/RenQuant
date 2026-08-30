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


def test_pytest_alert_log_is_isolated(tmp_path, monkeypatch):
    import live.alerts as alerts

    monkeypatch.setattr(alerts, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_alerts.py::test_x (call)")
    monkeypatch.delenv("RENQUANT_ALERT_STATE_PATH", raising=False)
    monkeypatch.delenv("RENQUANT_NO_NOTIFY", raising=False)
    monkeypatch.setenv("RENQUANT_NTFY_BACKOFF_SECONDS", "0")

    with patch("urllib.request.urlopen", return_value=SimpleNamespace(read=lambda: b"ok")):
        ok = alerts.post_ntfy_alert(
            "https://ntfy.sh/test",
            alerts.AlertEvent("DECISION", "A", "body", key="k", cooldown_seconds=3600),
        )

    assert ok
    alert_dir = tmp_path / "logs" / "alerts"
    assert not (alert_dir / "alert_log.jsonl").exists()
    assert list(alert_dir.glob("pytest-*.jsonl"))


def test_no_notify_uses_resolved_state_log(tmp_path, monkeypatch):
    from live.alerts import AlertEvent, post_ntfy_alert

    state_path = tmp_path / "alert_state.json"
    monkeypatch.setenv("RENQUANT_NO_NOTIFY", "1")

    with patch("urllib.request.urlopen") as m:
        ok = post_ntfy_alert(
            "https://ntfy.sh/test",
            AlertEvent("DECISION", "A", "body", key="k", cooldown_seconds=3600),
            state_path=state_path,
        )

    assert not ok
    m.assert_not_called()
    assert (tmp_path / "alert_state.jsonl").exists()


# ── Header encoding (orch#886) ──────────────────────────────────────────────
# live/alerts.py is declared BYTE-IDENTICAL to renquant-execution's twin and
# routes its Title header through renquant_common.notify.encode_header (#585).
# These tests pin that a non-ASCII title (emoji / CJK / em dash) is RFC 2047
# wrapped on BOTH transports rather than raising UnicodeEncodeError and
# dropping the whole alert — the 2026-07-27/28 "rq105 DOWN" loss mode.

_NON_ASCII_TITLE = "🚨 rq104 假想前10 — 2026-07-28"


def test_alerts_uses_the_shared_renquant_common_encoder():
    import live.alerts as alerts
    from renquant_common.notify import encode_header

    assert alerts.encode_header is encode_header


def test_non_ascii_title_is_rfc2047_wrapped_on_urllib(tmp_path, monkeypatch):
    import base64

    from live.alerts import AlertEvent, post_ntfy_alert

    monkeypatch.delenv("RENQUANT_NO_NOTIFY", raising=False)
    monkeypatch.setenv("RENQUANT_NTFY_BACKOFF_SECONDS", "0")
    event = AlertEvent("ACTION_REQUIRED", _NON_ASCII_TITLE, "body 正文", force=True)

    with patch("urllib.request.urlopen", return_value=SimpleNamespace(read=lambda: b"ok")) as m:
        assert post_ntfy_alert("https://ntfy.sh/test", event, state_path=tmp_path / "s.json")

    m.assert_called_once()
    req = m.call_args[0][0]
    title = req.get_header("Title")
    assert title.startswith("=?UTF-8?B?") and title.endswith("?=")
    title.encode("latin-1")  # what urllib does on the wire — must not raise
    assert base64.b64decode(title[len("=?UTF-8?B?"):-2]).decode("utf-8") == _NON_ASCII_TITLE
    assert req.data == "body 正文".encode("utf-8")


def test_ascii_title_passes_through_unchanged(tmp_path, monkeypatch):
    from live.alerts import AlertEvent, post_ntfy_alert

    monkeypatch.delenv("RENQUANT_NO_NOTIFY", raising=False)
    monkeypatch.setenv("RENQUANT_NTFY_BACKOFF_SECONDS", "0")
    event = AlertEvent("TRADE", "RENQUANT-104 [full] TRADE: BUY NVDA x5", "b", force=True)

    with patch("urllib.request.urlopen", return_value=SimpleNamespace(read=lambda: b"ok")) as m:
        assert post_ntfy_alert("https://ntfy.sh/test", event, state_path=tmp_path / "s.json")

    assert m.call_args[0][0].get_header("Title") == "RENQUANT-104 [full] TRADE: BUY NVDA x5"


def test_non_ascii_title_is_rfc2047_wrapped_on_the_curl_fallback(tmp_path, monkeypatch):
    import live.alerts as alerts

    monkeypatch.delenv("RENQUANT_NO_NOTIFY", raising=False)
    monkeypatch.delenv("RENQUANT_NTFY_DISABLE_CURL_FALLBACK", raising=False)
    monkeypatch.setenv("RENQUANT_NTFY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("RENQUANT_NTFY_RETRIES", "1")
    event = alerts.AlertEvent("ACTION_REQUIRED", _NON_ASCII_TITLE, "b", force=True)

    with patch("urllib.request.urlopen", side_effect=OSError("down")), \
            patch.object(alerts.subprocess, "run") as curl:
        assert alerts.post_ntfy_alert("https://ntfy.sh/test", event, state_path=tmp_path / "s.json")

    curl.assert_called_once()
    argv = curl.call_args[0][0]
    header = argv[argv.index("-H") + 1]
    assert header.startswith("Title: =?UTF-8?B?")
    header.encode("ascii")
