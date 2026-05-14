"""LEAN-specific leakage guard for the panel scorer artifact.

Defends the LEAN backtest path (`main.py:Initialize`) against the same
look-ahead leakage class that affects sim. Per CLAUDE.md §5.13.5 (single
source of truth), this peeks at the panel artifact JSON to extract
`trained_date`, then routes through `assert_no_leakage` from
`leakage_guard.py`. Adding a parallel implementation requires deleting
this one first.

Per §5.13.3, the regression invariant lives in
`tests/test_lean_guard.py::TestLeanGuardRegression` — pin it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .leakage_guard import assert_no_leakage


def _read_trained_date(artifact_full: Path) -> str | None:
    """Open artifact JSON and extract `trained_date` field, or None."""
    if not artifact_full.exists():
        return None
    try:
        meta = json.loads(artifact_full.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return meta.get("trained_date")


def assert_lean_panel_no_leakage(
    *,
    config: dict[str, Any],
    strategy_dir: Path,
    is_live_mode: bool,
) -> None:
    """Raise ValueError if the panel artifact's trained_date >= backtest_end.

    Skips silently (no raise) when:
      - LEAN is in LiveMode (no backtest window applies)
      - panel_scoring is disabled in config
      - artifact file does not exist (LoadScorerTask will fail later with
        a clearer message)
      - artifact JSON is malformed or has no `trained_date` metadata
        (legacy artifact)
      - config has no `backtest_end`

    Mirrors the SimAdapter `_assert_legacy_no_leakage` check (P2,
    `adapters/sim.py`). Wired into `main.py:Initialize` after
    `_load_all_models()`.
    """
    if is_live_mode:
        return

    panel_cfg = config.get("ranking", {}).get("panel_scoring", {})
    if not panel_cfg.get("enabled", True):
        return

    # §5.13.14: require explicit artifact_path. A sim/research LEAN config
    # that forgot to override panel_ltr.artifact_path used to read the
    # prod artifact's trained_date and validate against THIS sim's
    # backtest_end — silently misleading.
    artifact_rel = panel_cfg.get("artifact_path")
    if not artifact_rel:
        import logging as _logging  # noqa: PLC0415
        _logging.getLogger("kernel.walk_forward.lean_guard").warning(
            "assert_lean_no_leakage: panel_scoring.enabled=true but no "
            "artifact_path set — skipping leakage guard. Set artifact_path "
            "explicitly to re-enable the trained_date check."
        )
        return
    artifact_full = Path(strategy_dir) / artifact_rel

    trained_date = _read_trained_date(artifact_full)
    if trained_date is None:
        return

    backtest_end = config.get("backtest_end")
    if backtest_end is None:
        return

    assert_no_leakage(
        trained_date,
        backtest_end,
        context="LEAN backtest panel scorer",
    )
