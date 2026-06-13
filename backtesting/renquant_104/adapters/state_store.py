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
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("live.runner")  # same logger as the runner — log
                                        # contract unchanged by the move


def load_live_state(state_file: Path, config: dict, strategy_dir) -> dict:
    """JSON-first live-state load with the #144 DB fallback chain."""
    state: dict = {}
    json_loaded = False
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text()) or {}
            json_loaded = True
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


def save_live_state_atomic(state_file: Path, state: dict) -> None:
    """LS-ATOM write: tmp + atomic rename (audit fix, 2026-04-25)."""
    tmp_path = state_file.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, indent=2))
    tmp_path.replace(state_file)
