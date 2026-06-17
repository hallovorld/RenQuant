"""Live-state load/save — runner.py decomposition slice 1 (state_store).

EXTRACTED 2026-06-13 from adapters/runner.py (eng plan S2 item 5:
"Decompose runner.py (state_store / broker_sync / order_emit /
reporting)"). Behavior-identical move; the sim replay does NOT cover
the live adapter, so the gate is the state-contract test suites plus a
line-faithful move (every log message, fallback order, and error class
preserved verbatim).

Two functions, the exact code that lived inline in the runner:

  * load_live_state  — JSON-first, RESTORE-FROM-DB (#144) fallback with
    14-day age cap, hot-cache write-back, empty-state last resort.
  * save_live_state_atomic — LS-ATOM tmp+rename (audit 2026-04-25):
    a SIGKILL mid-write leaves the prior complete snapshot, never a
    truncated file.

Typed live-state (s1-wire-live-state-v2)
----------------------------------------
OPT-IN, default OFF. When ``RQ_LIVE_STATE_V2=1`` (or config
``live_state.typed_v2: true``), both helpers route the live-state dict
through ``kernel.live_state_v2.LiveStateV2`` — a lossless typed model whose
``parse(...).to_v1_dict()`` is the byte-for-byte inverse of the v1 flat
dict. The on-disk JSON is therefore IDENTICAL with the flag on or off; the
flag only changes whether the in-process state is validated/normalised
through the typed schema, so adding a per-holding field becomes a one-line
change. With the flag OFF the code below is exactly the verbatim runner
extraction — zero behaviour change.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("adapters.runner")  # same logger as the runner — log
                                           # contract unchanged by the move


def _typed_v2_enabled(config: dict | None) -> bool:
    """Opt-in flag for the typed LiveStateV2 path. Default OFF.

    Either env ``RQ_LIVE_STATE_V2=1`` (matches the repo's ``RQ_*`` flag
    convention) or config ``live_state.typed_v2: true`` turns it on. Any
    other value — including unset — leaves the plain-dict path unchanged.
    """
    if os.environ.get("RQ_LIVE_STATE_V2") == "1":
        return True
    if config:
        return bool((config.get("live_state") or {}).get("typed_v2", False))
    return False


def _roundtrip_through_v2(state: dict) -> dict:
    """parse(state).to_v1_dict() — the lossless typed normalisation.

    Returns a v1 dict byte-identical to the input under ``json.dumps``. Kept
    defensive: if the typed model is unavailable for any reason, fall back to
    the original dict so the flag can never break a live write.
    """
    try:
        from kernel.live_state_v2 import LiveStateV2  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - import-guard belt & braces
        log.warning(
            "LIVE-STATE-V2: typed model unavailable (%s) — using plain dict",
            exc,
        )
        return state
    return LiveStateV2.parse(state).to_v1_dict()


def load_live_state(state_file: Path, config: dict, strategy_dir) -> dict:
    """JSON-first live-state load with the #144 DB fallback chain."""
    state: dict = {}
    json_loaded = False
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text()) or {}
            json_loaded = True
            # s1-wire-live-state-v2 (opt-in, default OFF): normalise the
            # loaded dict through the typed model. Lossless, so the in-memory
            # state is unchanged when the flag is on; it only validates the
            # shape via the schema before the runner consumes it.
            if _typed_v2_enabled(config):
                state = _roundtrip_through_v2(state)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning(
                "live_state read failed (%s) — falling back to db",
                exc,
            )
    if not json_loaded:
        try:
            from kernel.persistence import (  # noqa: PLC0415
                get_connection, load_latest_live_state,
            )
            conn = get_connection(config, strategy_dir=strategy_dir)
            strategy_name = config.get("_strategy_name", "renquant_104")
            # max_age_days=14 — defensive: don't resurrect ancient state
            # (e.g. from a 6-month-old test db). 14d aligns with the
            # max plausible gap before a sim/restore is needed.
            db_state = load_latest_live_state(
                conn, strategy=strategy_name, max_age_days=14,
            )
            if db_state:
                log.warning(
                    "RESTORE-FROM-DB (#144): live_state.json missing/"
                    "corrupt — restored from live_state_snapshots "
                    "(strategy=%s). Writing JSON cache now.",
                    strategy_name,
                )
                state = db_state
                # Write the recovered state back to JSON so subsequent
                # bars see a hot cache (no need to re-query db).
                try:
                    state_file.write_text(json.dumps(state, default=str))
                except OSError as exc:
                    log.warning(
                        "RESTORE-FROM-DB: JSON write-back failed (%s) "
                        "— state recovered in-memory only", exc,
                    )
        except Exception as exc:
            log.warning(
                "RESTORE-FROM-DB: db load failed (%s) — proceeding "
                "with empty state", exc,
            )
    return state


def save_live_state_atomic(
    state_file: Path, state: dict, config: dict | None = None
) -> None:
    """LS-ATOM write: tmp + atomic rename (audit fix, 2026-04-25).

    ``config`` is optional and defaults to None so the existing
    ``save_live_state_atomic(state_file, state)`` call site is unchanged. It
    only feeds the opt-in typed-v2 flag (env ``RQ_LIVE_STATE_V2=1`` still
    works without it).
    """
    # s1-wire-live-state-v2 (opt-in, default OFF): route the dict through the
    # lossless typed model before serialising. parse(state).to_v1_dict() is
    # byte-identical to state, so the on-disk JSON is unchanged — the typed
    # path only guarantees the schema-validated shape is what gets written.
    if _typed_v2_enabled(config):
        state = _roundtrip_through_v2(state)
    tmp_path = state_file.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, indent=2))
    tmp_path.replace(state_file)
