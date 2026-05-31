"""PreflightContext — read-mostly state shared across PreflightTasks.

Mirrors the kwargs the legacy ``_check_*`` functions take, packed as a single
dataclass so Tasks have a typed interface. Plus a ``results`` list each Task
appends to (analogous to ``run_preflight``'s local ``results`` list).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Re-export legacy PreflightCheck (the result record) so consumers don't need
# two imports. Migration target: lift PreflightCheck into this module once
# the legacy preflight.py retires.
from kernel.preflight import PreflightCheck  # noqa: PLC0415  (legacy bridge)


@dataclass
class PreflightContext:
    """Inputs available to every PreflightTask.

    Keep this dataclass narrow: anything a Task needs must be passed in
    explicitly. No global imports; no environment lookups inside Tasks.
    """

    config: dict
    strategy_dir: Path
    broker: Any = None
    broker_name: str | None = None
    run_mode: str | None = None
    results: list[PreflightCheck] = field(default_factory=list)

    def append(self, check: PreflightCheck) -> None:
        """Record a check result on the context."""
        self.results.append(check)
