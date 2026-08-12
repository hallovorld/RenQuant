"""`live/alerts.py` twin contract: ntfy header values must be transport-safe.

Committed oracle for the `encode_header` sync (execution#40 -> the live twin).
Every assertion here FAILS on the pre-sync copy that passed a raw
``event.title`` into an HTTP header, so this module distinguishes the bug from
the fix instead of merely exercising the path.

Why a header value is not a free-form string: per RFC 7230 header field values
are ISO-8859-1, and ``http.client`` enforces that by encoding them as latin-1.
A title carrying U+2212 (``live/stream_watchdog.py`` writes one) therefore
raises ``UnicodeEncodeError`` on the urllib path and reaches ntfy as mojibake
through the curl fallback. ``encode_header`` emits RFC 2047 encoded-words,
which are pure ASCII.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# U+2212 MINUS SIGN — the character actually present in a live alert title,
# not a synthetic one. See live/stream_watchdog.py.
NON_ASCII_TITLE = "WATCHDOG NVDA −4.1%"


def _event(title):
    from live.alerts import AlertEvent

    return AlertEvent(
        taxonomy="DECISION",
        title=title,
        body="regression probe",
        key=f"header-encoding-{title}",
        cooldown_seconds=0,
    )


def _quiet(monkeypatch):
    monkeypatch.delenv("RENQUANT_NO_NOTIFY", raising=False)
    monkeypatch.setenv("RENQUANT_NTFY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("RENQUANT_NTFY_RETRIES", "1")


def test_urllib_title_header_is_latin1_safe(tmp_path, monkeypatch):
    """The header the urllib path builds must survive http.client's encoder.

    This is the assertion that fails pre-sync: the raw title is not latin-1
    encodable, so the real send raises before a byte leaves the process.
    """
    from live.alerts import post_ntfy_alert

    _quiet(monkeypatch)
    seen = {}

    def _capture(req, *a, **kw):
        seen["title"] = req.get_header("Title")
        return SimpleNamespace(read=lambda: b"ok")

    with patch("urllib.request.urlopen", side_effect=_capture):
        assert post_ntfy_alert(
            "https://ntfy.sh/test", _event(NON_ASCII_TITLE),
            state_path=tmp_path / "s.json",
        )

    title = seen["title"]
    assert title is not None, "no Title header was built"
    # The contract, stated as the transport states it.
    title.encode("latin-1")
    assert title.isascii(), f"Title header is not ASCII: {title!r}"
    assert title != NON_ASCII_TITLE, "raw title reached the header unencoded"


def test_curl_fallback_title_is_latin1_safe(tmp_path, monkeypatch):
    """The curl fallback must not smuggle raw UTF-8 into a header either.

    Pre-sync this path 'succeeds' while delivering mojibake, which is the more
    dangerous failure of the two: it looks delivered.
    """
    from live.alerts import post_ntfy_alert

    _quiet(monkeypatch)
    monkeypatch.delenv("RENQUANT_NTFY_DISABLE_CURL_FALLBACK", raising=False)
    argv = {}

    def _run(cmd, *a, **kw):
        argv["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    with patch("urllib.request.urlopen", side_effect=OSError("forced urllib failure")), \
         patch("subprocess.run", side_effect=_run):
        post_ntfy_alert(
            "https://ntfy.sh/test", _event(NON_ASCII_TITLE),
            state_path=tmp_path / "s.json",
        )

    cmd = argv.get("cmd")
    assert cmd is not None, "curl fallback never ran"
    titles = [a for a in cmd if isinstance(a, str) and a.startswith("Title: ")]
    assert titles, f"no Title header in curl argv: {cmd!r}"
    value = titles[0][len("Title: "):]
    value.encode("latin-1")
    assert value.isascii(), f"curl Title is not ASCII: {value!r}"


def test_ascii_title_is_not_mangled(tmp_path, monkeypatch):
    """Guards the other direction: encoding must be identity on ASCII.

    Every alert title the wired path produces today is ASCII, so if this ever
    fails the sync changed live behaviour rather than preserving it.
    """
    from live.alerts import post_ntfy_alert

    _quiet(monkeypatch)
    plain = "RQ104 [full] TRADE"
    seen = {}

    def _capture(req, *a, **kw):
        seen["title"] = req.get_header("Title")
        return SimpleNamespace(read=lambda: b"ok")

    with patch("urllib.request.urlopen", side_effect=_capture):
        assert post_ntfy_alert(
            "https://ntfy.sh/test", _event(plain), state_path=tmp_path / "s.json",
        )

    assert seen["title"] == plain, (
        f"ASCII title was altered: {seen['title']!r} != {plain!r}"
    )


_PINNED_TWIN = (
    Path(__file__).resolve().parent.parent
    / ".subrepo_runtime" / "repos" / "renquant-execution"
    / "src" / "renquant_execution" / "alerts.py"
)


@pytest.mark.skipif(
    not _PINNED_TWIN.exists(),
    reason=(
        "pinned subrepo assembly absent (fresh clone / CI) — the byte-identity "
        "leg runs on the deploy machine, where the twin that trades lives. The "
        "behavioural tests above hold the contract everywhere."
    ),
)
def test_live_twin_is_byte_identical_to_the_pinned_execution_copy():
    """Structural leg: the manifest declares this pair byte-identical."""
    live = Path(__file__).resolve().parent.parent / "live" / "alerts.py"
    assert live.read_bytes() == _PINNED_TWIN.read_bytes(), (
        "live/alerts.py has diverged from the PINNED renquant-execution copy; "
        "a fix landed on one stack only (audit C1-a)."
    )
