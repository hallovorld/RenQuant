"""Software-stop registry wiring for the live runner — the NEUTRAL ROOT.

Umbrella-side glue only (no capability logic lives here — RenQuant#440):

* the registry itself (schema, tagging, ``from_config``) belongs to
  ``renquant_pipeline.software_stops`` (renquant-pipeline#167);
* the registry LOCATION convention belongs to
  ``renquant_orchestrator.software_stops_registry_contract`` (orch#481 /
  #1078): the neutral, host-scoped runtime-state root
  ``~/.renquant/runtime/software-stops`` (override:
  ``RENQUANT_RUNTIME_STATE_ROOT``), which is the ``--data-root`` the
  liveness pager (``renquant_execution.software_stops_liveness``) and the
  orchestrator seeder / readiness classifier resolve against.

Before 2026-08-29 ``RunnerAdapter.__init__`` called ``from_config`` without
``repo_root``, so the pipeline's relative default ``registry_path``
(``data/rq105/software_stops.json``) resolved against the process cwd — the
umbrella checkout (``scripts/intraday_sell_104.sh`` does ``cd "$REPO_DIR"``)
— while the checker and the seeder looked under the neutral root. A writer
stamping one path and a watchdog reading another is a dark pager. This
module makes the writer resolve the SAME root the checker uses, and it
never falls back to the cwd: if the orchestrator contract module is not
importable the registry is not constructed at all (``None`` => the stage-0
capability gate stays unarmed, fractional entries fail closed) and one
ERROR line says why.

Ordering (Codex review on RenQuant#613): the pipeline-owned enabled gate is
read FIRST — ``execution.software_stops.enabled`` absent/false returns
``None`` before anything else happens: no orchestrator import, no neutral
root resolution, no log line, no disk access. Only an ENABLED layer imports
the contract, resolves the root and calls
``SoftwareStopRegistry.from_config(..., repo_root=root)``. The gate is a
mirror of ``from_config``'s own read (``software_stops.py`` ``from_config``:
``((config or {}).get("execution") or {}).get("software_stops") or {}`` then
``.get("enabled", False)`` truthiness); ``from_config`` still re-applies its
own gate afterwards, so the pipeline stays authoritative — the mirror can
only make this wiring MORE inert, never less.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

# Same logger as RunnerAdapter so the existing "software-stop registry
# construction FAILED" line keeps its name for anyone grepping run logs.
log = logging.getLogger("adapters.runner")

CONTRACT_MODULE = "renquant_orchestrator.software_stops_registry_contract"


class NeutralRootUnavailable(RuntimeError):
    """The orchestrator LOCATION contract is not importable.

    Raised INSTEAD of resolving a registry path against the cwd. The
    message names the missing module and states that no fallback is taken.
    """


def software_stops_enabled(config: "dict | None") -> bool:
    """The pipeline-owned enabled gate, read the pipeline's way.

    ``renquant_pipeline.software_stops`` exposes no public enabled-check, so
    this is a verbatim mirror of ``SoftwareStopRegistry.from_config``'s first
    three lines: the same key path ``execution.software_stops.enabled`` with
    ``or {}`` at every level and ``.get("enabled", False)`` truthiness.
    Pure: reads the dict only — imports nothing, logs nothing, touches no
    disk. Parity with the real ``from_config`` is pinned by
    ``tests/test_software_stops_neutral_root.py``.
    """
    ss_cfg = ((config or {}).get("execution") or {}).get("software_stops") or {}
    return bool(ss_cfg.get("enabled", False))


def software_stops_neutral_root() -> Path:
    """``<runtime-state root>/software-stops`` per the orchestrator contract.

    Resolution is the contract's own (override -> ``RENQUANT_RUNTIME_STATE_ROOT``
    -> ``~/.renquant/runtime``); this function adds nothing and caches
    nothing. Raises :class:`NeutralRootUnavailable` when the contract module
    cannot be imported — never returns a cwd-relative or repo-relative path.
    """
    try:
        from renquant_orchestrator.software_stops_registry_contract import (  # noqa: PLC0415
            runtime_state_root,
            software_stops_registry_root,
        )
    except ImportError as exc:
        raise NeutralRootUnavailable(
            f"neutral runtime-state root UNAVAILABLE: {CONTRACT_MODULE} is not "
            f"importable ({type(exc).__name__}: {exc}). REFUSING to resolve the "
            "software-stop registry against the process cwd — the liveness "
            "checker and the registry seeder resolve against the neutral root, "
            "and a writer on a different path is a dark pager. Put the pinned "
            "renquant-orchestrator checkout's src on PYTHONPATH."
        ) from exc
    return software_stops_registry_root(runtime_state_root())


def build_software_stop_registry(config: "dict | None", broker_name: str | None) -> Any:
    """Flag-gated registry construction at the neutral root, fail-closed.

    * ``execution.software_stops.enabled`` absent/false -> ``None``
      immediately: nothing imported, resolved, logged or written (the
      pre-2026-08-29 inert path, preserved byte-for-byte).
    * enabled -> import the orchestrator LOCATION contract (unimportable ->
      ONE ERROR line + ``None``; no cwd fallback), resolve the neutral root,
      ``SoftwareStopRegistry.from_config(config, broker_name=…, repo_root=root)``;
      any construction failure -> ONE ERROR line + ``None``.

    ``None`` is what ``commit_contract.software_stops_armed`` reads as "not
    armed": fractional entries stay blocked by the stage-0 capability gate.
    """
    if not software_stops_enabled(config):
        return None
    try:
        # 2026-07-04: relocated to renquant_pipeline.software_stops
        # (renquant-pipeline#167) -- new capability logic belongs in
        # an owning repo, not the umbrella (RenQuant#440 review).
        from renquant_pipeline.software_stops import SoftwareStopRegistry  # noqa: PLC0415

        root = software_stops_neutral_root()
        registry = SoftwareStopRegistry.from_config(
            config, broker_name=broker_name, repo_root=root,
        )
    except Exception as exc:  # noqa: BLE001 — fail closed, never raise into the runner
        log.error(
            "software-stop registry construction FAILED: %s — layer "
            "NOT armed; fractional entries remain fail-closed by the "
            "stage-0 capability gate.", exc,
        )
        return None
    if registry is None:  # the pipeline gate is authoritative; defer to it
        return None
    log.info(
        "software-stop registry resolved under the NEUTRAL root %s "
        "(broker=%s, path=%s) — the path the liveness checker and the "
        "orchestrator seeder/readiness classifier read.",
        root, broker_name, getattr(registry, "path", "?"),
    )
    return registry
