"""Typed Past-only data contract for Tasks (cvxportfolio Estimator pattern).

Goal: make peek-ahead architecturally impossible. Tasks that subclass
``TypedTask`` receive a frozen ``Past`` snapshot pre-sliced to cursor ``t``.
Any DataFrame inside ``Past`` has had rows > t stripped, and the dataclass is
``frozen=True`` so a Task literally cannot mutate it.

This is the foundation. Migration of existing 100+ Tasks is multi-week —
see ``MIGRATION.md``. This module ships:

  * ``Past``                 — frozen dataclass, the snapshot
  * ``TypedTask``            — Protocol with ``values_in_time(t, past)``
  * ``TaskResult``           — typed return for ``values_in_time``
  * ``TypedTaskAdapter``     — bridge so a TypedTask can be used as a
                                legacy ``Task.run(ctx)`` step (for
                                incremental migration without big-bang).

References:
  * cvxportfolio.Estimator.values_in_time —
    https://www.cvxportfolio.com/api_documentation/estimator.html
  * RenQuant CLAUDE.md §1c (≤50 LOC tasks), §5.13.1 (real prod path),
    §5.13.10 (no defensive ``if X is not None`` dead code).
"""
from __future__ import annotations

from .past import Past
from .estimator import TaskResult, TypedTask, TypedTaskAdapter

__all__ = ["Past", "TypedTask", "TypedTaskAdapter", "TaskResult"]
