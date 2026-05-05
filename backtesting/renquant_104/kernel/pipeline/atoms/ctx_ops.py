"""Atoms that read, copy, move, or clear ctx fields by name.

User mandate (2026-05-04 §1c): atoms are small, parameterized, reusable
across Jobs. Field names are passed in — no hard-coded ctx fields here.

Each atom reads/writes nested fields via `_get_path` / `_set_path` so
expressions like `"holdings.AAPL.shares"` or `"_qp_solution.delta_w"`
work uniformly.
"""
from __future__ import annotations

from typing import Any

from ..pipeline import Task


def _get_path(obj: Any, path: str, default: Any = None) -> Any:
    """Resolve `a.b.c` against an object — supports attrs and dict keys."""
    if not path:
        return obj
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default if part == path.split(".")[-1] else None)
        else:
            cur = getattr(cur, part, default if part == path.split(".")[-1] else None)
    return cur


def _set_path(obj: Any, path: str, value: Any) -> None:
    """Set `a.b.c = value`. Creates intermediate attrs only on the leaf parent
    if it's a SimpleNamespace-like object — otherwise raises."""
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        if isinstance(cur, dict):
            cur = cur.setdefault(part, {})
        else:
            cur = getattr(cur, part)
    if isinstance(cur, dict):
        cur[parts[-1]] = value
    else:
        setattr(cur, parts[-1], value)


class CopyFieldTask(Task):
    """Copy `ctx.<src>` → `ctx.<dst>` (shallow). Useful between Jobs that
    need the same value under different names without coupling."""

    def __init__(self, src: str, dst: str):
        self.src, self.dst = src, dst

    @property
    def name(self) -> str:
        return f"CopyField({self.src}→{self.dst})"

    def run(self, ctx) -> bool | None:
        v = _get_path(ctx, self.src)
        _set_path(ctx, self.dst, v)


class ClearFieldTask(Task):
    """Set `ctx.<field>` = None or a fresh empty container."""

    def __init__(self, field: str, fill: Any = None):
        self.field, self.fill = field, fill

    @property
    def name(self) -> str:
        return f"ClearField({self.field})"

    def run(self, ctx) -> bool | None:
        _set_path(ctx, self.field, self.fill() if callable(self.fill) else self.fill)


class AssertFieldExistsTask(Task):
    """Raise AssertionError if `ctx.<field>` is None or missing.

    Useful as a contract check between Jobs — fails loud rather than
    letting downstream Tasks see a None they can't act on.
    """

    def __init__(self, field: str, message: str | None = None):
        self.field = field
        self.message = message

    @property
    def name(self) -> str:
        return f"AssertFieldExists({self.field})"

    def run(self, ctx) -> bool | None:
        v = _get_path(ctx, self.field)
        if v is None:
            raise AssertionError(
                self.message or f"required ctx field {self.field!r} is None"
            )


__all__ = [
    "_get_path", "_set_path",
    "CopyFieldTask", "ClearFieldTask", "AssertFieldExistsTask",
]
