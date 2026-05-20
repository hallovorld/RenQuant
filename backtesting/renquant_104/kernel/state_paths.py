"""Broker-isolated state file paths.

The 2026-04-27 incident: a ``--broker paper`` smoke run wrote synthetic
positions into ``live_state.json``; the next ``--broker alpaca`` invocation
read those fake positions instead of the real Alpaca account state. This
module centralises path construction so each broker (paper / alpaca-paper /
alpaca / ibkr) gets its own state file and runs.db.

Naming convention
-----------------
- ``live_state.alpaca.json``        — real Alpaca live account
- ``live_state.alpaca_paper.json``  — Alpaca paper account
- ``live_state.paper.json``         — in-process simulation
- ``live_state.ibkr.json``          — future
- ``data/runs.alpaca.db``           — analogous SQLite store

The single legacy ``live_state.json`` (pre-isolation) is treated as a
read-only fallback when a broker-specific file does not exist yet —
useful during the migration window. New writes always go to the
broker-specific path; legacy is never overwritten.
"""
from __future__ import annotations

from pathlib import Path

# VAL-1 (2026-04-28 self-audit): allowlist of broker tags. Centralised here
# so a typo / malicious caller cannot path-traverse via broker_name. Add
# new brokers here when wiring them through live/runner.py.
ALLOWED_BROKERS: frozenset[str] = frozenset({
    "paper", "alpaca", "alpaca-paper",
    "alpaca_paper",     # post-replace form (allow both)
    "alpaca-shorts",
    "alpaca_shorts",
    "alpaca_shadow",    # 2026-05-19 full-e2e shadow via ReadOnlyBrokerWrapper
    "ibkr",
})


def _safe_broker(broker_name: str | None) -> str:
    """Sanitise broker name for use in a filename.

    Replaces '-' with '_' (some downstream tools mishandle hyphens in stems)
    and validates against ALLOWED_BROKERS. Returns ``"unknown"`` for
    None / empty (treated as a non-fatal fallback). Raises ValueError when
    the broker_name is non-empty but not in the allowlist — this prevents
    a directory-traversal attack via crafted broker_name.
    """
    if not broker_name:
        return "unknown"
    if broker_name not in ALLOWED_BROKERS:
        raise ValueError(
            f"Unknown broker_name {broker_name!r}; expected one of "
            f"{sorted(ALLOWED_BROKERS)}. Add new brokers to ALLOWED_BROKERS "
            f"in kernel/state_paths.py before using."
        )
    return broker_name.replace("-", "_")


def live_state_path(strategy_dir: Path | str, broker_name: str | None) -> Path:
    """Return the broker-isolated live_state.json path."""
    return Path(strategy_dir) / f"live_state.{_safe_broker(broker_name)}.json"


def live_state_legacy_path(strategy_dir: Path | str) -> Path:
    """Pre-isolation single-file path. Read-only fallback during migration."""
    return Path(strategy_dir) / "live_state.json"


def resolve_live_state_read(
    strategy_dir: Path | str, broker_name: str | None
) -> tuple[Path, bool]:
    """Pick the live_state file to read from.

    Returns ``(path, is_legacy_fallback)``. Caller should log a warning
    when ``is_legacy_fallback`` is True so the operator knows to verify.
    """
    primary = live_state_path(strategy_dir, broker_name)
    if primary.exists():
        return primary, False
    legacy = live_state_legacy_path(strategy_dir)
    if legacy.exists():
        return legacy, True
    return primary, False  # neither exists; caller will handle missing


def runs_db_path(base_path: Path | str, broker_name: str | None) -> Path:
    """Append broker tag before the .db suffix.

    e.g. ``data/runs.db`` + ``alpaca`` → ``data/runs.alpaca.db``.

    Idempotent: if the base path already ends with ``.{broker}.db`` (i.e.
    a previously-tagged path is passed in by mistake), the broker tag is
    NOT doubled. ``data/runs.alpaca.db`` + ``alpaca`` → ``data/runs.alpaca.db``.
    """
    p = Path(base_path)
    safe = _safe_broker(broker_name)
    # TEST-1 idempotence: detect existing broker tag in stem (e.g.
    # "runs.alpaca" → don't append again). The stem is the last
    # extension-less component; we check whether it already ends in
    # ``.{safe}``.
    if p.stem.endswith(f".{safe}"):
        return p
    return p.with_stem(f"{p.stem}.{safe}")


def runs_db_legacy_path(base_path: Path | str) -> Path:
    """Pre-isolation runs.db path. Used by analytics tools that span all brokers."""
    return Path(base_path)
