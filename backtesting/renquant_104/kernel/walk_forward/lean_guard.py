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


def _read_artifact_metadata(artifact_full: Path) -> dict[str, Any] | None:
    """Open artifact JSON and return metadata, or None if load must defer."""
    if not artifact_full.exists():
        return None
    try:
        return json.loads(artifact_full.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _selection_anchor(meta: dict[str, Any]) -> Any:
    contract = meta.get("training_contract") or {}
    split_ranges = (
        meta.get("split_date_ranges")
        or contract.get("split_date_ranges")
        or {}
    )
    validation_end = (
        (split_ranges.get("val") or {}).get("end")
        if isinstance(split_ranges, dict) else None
    )
    return (
        meta.get("effective_selection_cutoff_date")
        or contract.get("effective_selection_cutoff_date")
        or validation_end
        or meta.get("effective_train_cutoff_date")
        or contract.get("effective_train_cutoff_date")
        or meta.get("train_cutoff_date")
        or meta.get("cutoff_date")
        or meta.get("trained_date")
        or contract.get("trained_date")
    )


def assert_lean_panel_no_leakage(
    *,
    config: dict[str, Any],
    strategy_dir: Path,
    is_live_mode: bool,
) -> None:
    """Raise if the panel artifact could have seen a backtest bar's label.

    Skips silently (no raise) when:
      - LEAN is in LiveMode (no backtest window applies)
      - panel_scoring is disabled in config
      - artifact file does not exist (LoadScorerTask will fail later with
        a clearer message)
      - artifact JSON is malformed
      - config has no `backtest_start` and no `backtest_end`

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

    meta = _read_artifact_metadata(artifact_full)
    if meta is None:
        return
    trained_date = (
        meta.get("trained_date")
        or (meta.get("training_contract") or {}).get("trained_date")
    )
    if trained_date is None:
        raise ValueError(
            "LEAN backtest panel scorer artifact is missing trained_date "
            f"metadata: {artifact_full}. Historical validation cannot prove "
            "this static scorer is point-in-time."
        )

    sim_first_bar = config.get("backtest_start") or config.get("backtest_end")
    if sim_first_bar is None:
        return

    assert_no_leakage(
        _selection_anchor(meta),
        sim_first_bar,
        context="LEAN backtest panel scorer",
        lookahead_days=int(meta.get("lookahead_days", 0) or 0),
    )
