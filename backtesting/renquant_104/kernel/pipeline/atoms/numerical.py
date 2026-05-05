"""Numerical guard atoms — finite / range / non-empty / clamp.

The recurring NaN-slip bug class (gate fails False on NaN comparison)
is fixed once and for all by composing these atoms instead of writing
ad-hoc `if x is None or x <= 0` conditions in every Task.
"""
from __future__ import annotations

import math
from typing import Any

from ..pipeline import Task
from .ctx_ops import _get_path, _set_path


class IsFiniteGuardTask(Task):
    """Check `ctx.<field>` is finite. On violation, choose action.

    on_violation:
      "skip"    — return False, short-circuit the Job chain
      "raise"   — raise ValueError
      "zero"    — overwrite with 0.0
      "default" — overwrite with self.default
    """

    def __init__(
        self,
        field: str,
        on_violation: str = "skip",
        default: Any = 0.0,
    ):
        if on_violation not in {"skip", "raise", "zero", "default"}:
            raise ValueError(f"on_violation={on_violation!r}")
        self.field = field
        self.on_violation = on_violation
        self.default = default

    @property
    def name(self) -> str:
        return f"IsFiniteGuard({self.field}, {self.on_violation})"

    def run(self, ctx) -> bool | None:
        v = _get_path(ctx, self.field)
        try:
            ok = v is not None and math.isfinite(float(v))
        except (TypeError, ValueError):
            ok = False
        if ok:
            return
        if self.on_violation == "skip":
            return False
        if self.on_violation == "raise":
            raise ValueError(f"{self.field}={v!r} non-finite")
        _set_path(ctx, self.field, 0.0 if self.on_violation == "zero" else self.default)


class RangeGuardTask(Task):
    """Verify `lo <= ctx.<field> <= hi`. Same on_violation semantics."""

    def __init__(
        self, field: str, lo: float, hi: float,
        on_violation: str = "skip",
    ):
        self.field, self.lo, self.hi = field, lo, hi
        self.on_violation = on_violation

    @property
    def name(self) -> str:
        return f"RangeGuard({self.field}, [{self.lo},{self.hi}])"

    def run(self, ctx) -> bool | None:
        v = _get_path(ctx, self.field)
        try:
            f = float(v)
            ok = math.isfinite(f) and (self.lo <= f <= self.hi)
        except (TypeError, ValueError):
            ok = False
        if ok:
            return
        if self.on_violation == "raise":
            raise ValueError(
                f"{self.field}={v!r} outside [{self.lo}, {self.hi}]"
            )
        return False if self.on_violation == "skip" else None


class NonEmptyGuardTask(Task):
    """Verify `ctx.<field>` is a non-empty container."""

    def __init__(self, field: str, on_violation: str = "skip"):
        self.field, self.on_violation = field, on_violation

    @property
    def name(self) -> str:
        return f"NonEmptyGuard({self.field})"

    def run(self, ctx) -> bool | None:
        v = _get_path(ctx, self.field)
        ok = v is not None and hasattr(v, "__len__") and len(v) > 0
        if ok:
            return
        if self.on_violation == "raise":
            raise ValueError(f"{self.field} is empty / None")
        return False


class ClampFieldTask(Task):
    """Clamp `ctx.<field>` to [lo, hi] in place. NaN → midpoint."""

    def __init__(self, field: str, lo: float, hi: float):
        self.field, self.lo, self.hi = field, lo, hi

    @property
    def name(self) -> str:
        return f"Clamp({self.field}, [{self.lo},{self.hi}])"

    def run(self, ctx) -> bool | None:
        v = _get_path(ctx, self.field)
        try:
            f = float(v)
            if not math.isfinite(f):
                f = 0.5 * (self.lo + self.hi)
            f = max(self.lo, min(self.hi, f))
        except (TypeError, ValueError):
            f = 0.5 * (self.lo + self.hi)
        _set_path(ctx, self.field, f)


__all__ = [
    "IsFiniteGuardTask",
    "RangeGuardTask",
    "NonEmptyGuardTask",
    "ClampFieldTask",
]
