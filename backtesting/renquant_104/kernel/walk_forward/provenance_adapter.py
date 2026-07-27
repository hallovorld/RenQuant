"""Sim-side construction of the WF provenance sink (pipeline#215/#216).

This is the UMBRELLA half of the ``wf_sim_provenance.v1`` contract
(``renquant-pipeline doc/design/2026-07-27-wf-sim-provenance-contract.md``):
the record constructors, digest grammar, PIT check, and the JSONL sink all
live in ``renquant_pipeline.kernel.walk_forward.provenance`` (single owner —
never re-implemented here). This module only decides:

* WHERE the JSONL lands — ``<sim repo root>/data/wf_provenance/<sim_run_id>
  .jsonl``, rooted on THIS module's own tree (``__file__``), i.e. the
  checkout the sim's code actually runs from. Never ``strategy_dir`` (under
  ``snapshot=True`` that is a throwaway tmpdir deleted at sim end — evidence
  must outlive the snapshot) and never a hardcoded live-tree path.
* WHICH revision pins are captured — the checkouts the sim actually imports
  from (module resolution first, sibling-checkout default otherwise).

Pin-vintage honesty: when the pinned ``renquant-pipeline`` predates
pipeline#216 the provenance module does not exist; this helper returns
``None`` with a LOUD warning and the sim runs exactly as before (no emit).
The sim therefore only starts emitting provenance once the pipeline pin
advances past #216 — that pin advance ships with the rerun batch, not here.

Live-surface delta: ZERO. Only ``sim.runner.run_backtest`` calls this, and
only when ``walkforward.enabled`` is true. The daily/live path never
constructs a sink.
"""
from __future__ import annotations

import datetime as _dt
import importlib
import logging
import uuid
from pathlib import Path

log = logging.getLogger("kernel.walk_forward.provenance_adapter")

#: repo-name -> importable module the sim actually runs that repo's code from.
_PIN_MODULES = {
    "pipeline": "renquant_pipeline",
    "common": "renquant_common",
    "model": "renquant_model",
    "artifacts": "renquant_artifacts",
}

#: repo-name -> sibling checkout dirname (fallback when not importable).
_PIN_SIBLINGS = {
    "pipeline": "renquant-pipeline",
    "common": "renquant-common",
    "model": "renquant-model",
    "artifacts": "renquant-artifacts",
    "backtesting": "renquant-backtesting",
}


def sim_repo_root() -> Path:
    """The umbrella tree THIS code runs from (worktree-safe, snapshot-safe)."""
    # <root>/backtesting/renquant_104/kernel/walk_forward/provenance_adapter.py
    return Path(__file__).resolve().parents[4]


def _module_repo_path(module_name: str) -> "Path | None":
    """Directory of an imported package — ``git -C`` walks up to its repo."""
    try:
        mod = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - absence is an expected state
        return None
    file = getattr(mod, "__file__", None)
    if not file:
        return None
    return Path(file).resolve().parent


def revision_pin_paths(repo_root: "Path | None" = None) -> dict[str, Path]:
    """The six-repo map for ``capture_revision_pins`` (design §2.1 code group).

    ``umbrella`` is this tree; the subrepos use the checkout the sim
    actually imports from when importable (editable installs / PYTHONPATH),
    else the sibling-checkout default. ``capture_revision_pins`` is
    best-effort — a missing path pins as ``None``, never raises.
    """
    root = Path(repo_root) if repo_root is not None else sim_repo_root()
    paths: dict[str, Path] = {"umbrella": root}
    for name in ("pipeline", "model", "backtesting", "common", "artifacts"):
        module_name = _PIN_MODULES.get(name)
        resolved = _module_repo_path(module_name) if module_name else None
        if resolved is None:
            resolved = root.parent / _PIN_SIBLINGS[name]
        paths[name] = resolved
    return paths


def mint_sim_run_id() -> str:
    """One identity per ``run_backtest`` call (all bars share it).

    ``ctx.run_id`` is per-BAR (``<date>-sim-<uuid8>``) and keys the
    ``score_observation_key`` instead; the JSONL filename needs the
    per-SIM identity the two-phase records are keyed on.
    """
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"wfsim-{stamp}-{uuid.uuid4().hex[:8]}"


def build_wf_provenance_sink(
    *,
    seed: "int | None" = None,
    sim_run_id: "str | None" = None,
    data_root: "Path | str | None" = None,
):
    """Construct the JSONL provenance sink, or ``None`` pre-#216 pin.

    Returns ``renquant_pipeline...provenance.JsonlProvenanceSink`` writing
    ``<data_root>/data/wf_provenance/<sim_run_id>.jsonl`` with the revision
    pins + seed captured at sim start (design §2.3/§2.4). ``data_root``
    exists for tests; the default is this checkout's root.
    """
    try:
        from renquant_pipeline.kernel.walk_forward.provenance import (  # noqa: PLC0415
            PROVENANCE_DIRNAME,
            JsonlProvenanceSink,
            capture_revision_pins,
        )
    except ImportError:
        log.warning(
            "WF provenance sink UNAVAILABLE: the pinned renquant-pipeline "
            "predates pipeline#216 (no kernel.walk_forward.provenance). "
            "Sim runs WITHOUT sim-time provenance emit; its scores stay "
            "Phase-A-inadmissible until the pipeline pin advances."
        )
        return None
    root = Path(data_root) if data_root is not None else sim_repo_root()
    directory = root / "data" / PROVENANCE_DIRNAME
    run_id = sim_run_id or mint_sim_run_id()
    # Pins are always captured against the CODE tree (sim_repo_root), not
    # data_root — a test redirecting the JSONL must not change what code
    # identity gets recorded.
    pins = capture_revision_pins(revision_pin_paths())
    sink = JsonlProvenanceSink(
        run_id, directory, seed=seed, revision_pins=pins,
    )
    log.info(
        "WF provenance sink: sim_run_id=%s path=%s seed=%s",
        run_id, sink.path, seed,
    )
    return sink
