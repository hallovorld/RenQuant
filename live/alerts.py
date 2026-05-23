"""Small alert helper for live operational notifications.

The live runner must never roll back trading state because ntfy is down, but
operator alerts also need basic hygiene: retry transient failures, suppress
duplicate low-information messages, and leave a local audit trail explaining
what was sent or suppressed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import time
import urllib.request


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AlertEvent:
    taxonomy: str
    title: str
    body: str
    key: str | None = None
    priority: str = "default"
    cooldown_seconds: int = 0
    force: bool = False


def stable_alert_key(*parts: object) -> str:
    """Build a compact stable key from non-price alert dimensions."""
    raw = json.dumps([str(p) for p in parts], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def post_ntfy_alert(
    url: str,
    event: AlertEvent,
    *,
    logger: logging.Logger | None = None,
    state_path: Path | None = None,
) -> bool:
    """Best-effort ntfy publish with persisted duplicate suppression.

    `force=True` bypasses dedupe and is intended for actual trades, failed
    exits, and other actionable broker-state alerts.
    """
    logger = logger or logging.getLogger("live.alerts")
    if os.environ.get("RENQUANT_NO_NOTIFY") == "1":
        _append_alert_log(event, "suppressed_env", state_path=state_path)
        logger.info("ntfy suppressed by RENQUANT_NO_NOTIFY: %s", event.title)
        return False

    state_path = state_path or _state_path()
    event_id = _event_id(event)
    now = time.time()
    state = _load_state(state_path)

    if _should_suppress(event, event_id, state, now):
        _append_alert_log(event, "suppressed_duplicate", state_path=state_path)
        logger.info("ntfy duplicate suppressed: %s key=%s", event.title, event_id)
        return False

    ok = _send_ntfy(url, event, logger)
    _append_alert_log(event, "sent" if ok else "failed", state_path=state_path)
    if ok and event_id and event.cooldown_seconds > 0:
        state.setdefault("events", {})[event_id] = {
            "sent_at": now,
            "title": event.title,
            "taxonomy": event.taxonomy,
        }
        _write_state(state_path, state)
    return ok


def _event_id(event: AlertEvent) -> str | None:
    if not event.key:
        return None
    taxonomy = event.taxonomy.strip().upper() or "INFO"
    return f"{taxonomy}:{event.key}"


def _state_path() -> Path:
    raw = os.environ.get("RENQUANT_ALERT_STATE_PATH")
    if raw:
        return Path(raw)
    pytest_name = os.environ.get("PYTEST_CURRENT_TEST")
    if pytest_name:
        digest = hashlib.sha256(pytest_name.encode("utf-8")).hexdigest()[:16]
        return REPO_ROOT / "logs" / "alerts" / f"pytest-{digest}-{os.getpid()}.json"
    return REPO_ROOT / "logs" / "alerts" / "alert_state.json"


def _log_path(state_path: Path | None) -> Path:
    if state_path is not None:
        return state_path.with_name("alert_log.jsonl")
    return REPO_ROOT / "logs" / "alerts" / "alert_log.jsonl"


def _load_state(path: Path) -> dict:
    try:
        if not path.exists():
            return {"version": 1, "events": {}}
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            return {"version": 1, "events": {}}
        raw.setdefault("version", 1)
        raw.setdefault("events", {})
        return raw
    except Exception:
        return {"version": 1, "events": {}}


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _append_alert_log(event: AlertEvent, status: str, *, state_path: Path | None) -> None:
    path = _log_path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": time.time(),
        "status": status,
        "taxonomy": event.taxonomy,
        "title": event.title,
        "key": event.key,
        "priority": event.priority,
        "force": event.force,
    }
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _should_suppress(event: AlertEvent, event_id: str | None, state: dict, now: float) -> bool:
    if event.force or not event_id or event.cooldown_seconds <= 0:
        return False
    last = (state.get("events", {}) or {}).get(event_id) or {}
    try:
        sent_at = float(last.get("sent_at", 0.0))
    except (TypeError, ValueError):
        return False
    return now - sent_at < float(event.cooldown_seconds)


def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    return min(max(val, lo), hi)


def _env_float(name: str, default: float, *, lo: float, hi: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except ValueError:
        return default
    return min(max(val, lo), hi)


def _send_ntfy(url: str, event: AlertEvent, logger: logging.Logger) -> bool:
    attempts = _env_int("RENQUANT_NTFY_RETRIES", 3, lo=1, hi=5)
    timeout = _env_float("RENQUANT_NTFY_TIMEOUT_SECONDS", 5.0, lo=1.0, hi=30.0)
    backoff = _env_float("RENQUANT_NTFY_BACKOFF_SECONDS", 1.0, lo=0.0, hi=10.0)
    body_bytes = event.body.encode("utf-8")
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(
                url,
                data=body_bytes,
                method="POST",
                headers={"Title": event.title, "Priority": event.priority},
            )
            urllib.request.urlopen(req, timeout=timeout).read()
            logger.info("ntfy sent: %s | %s", event.title, event.body)
            return True
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < attempts:
                logger.warning(
                    "ntfy publish attempt %d/%d failed (%s); retrying",
                    attempt, attempts, exc,
                )
                if backoff > 0:
                    time.sleep(backoff * attempt)

    if os.environ.get("RENQUANT_NTFY_DISABLE_CURL_FALLBACK") != "1":
        try:
            subprocess.run(
                [
                    "curl", "-fsS",
                    "--connect-timeout", str(min(timeout, 10.0)),
                    "--max-time", str(max(timeout + 5.0, 10.0)),
                    "-H", f"Title: {event.title}",
                    "-H", f"Priority: {event.priority}",
                    "--data-binary", "@-",
                    url,
                ],
                input=body_bytes,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                timeout=max(timeout + 7.0, 12.0),
            )
            logger.info("ntfy sent via curl fallback: %s | %s", event.title, event.body)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("ntfy curl fallback FAILED (%s)", exc)

    logger.warning(
        "ntfy publish FAILED after %d urllib attempt(s) (%s) - cycle still committed: %s",
        attempts, last_exc, event.body,
    )
    return False
