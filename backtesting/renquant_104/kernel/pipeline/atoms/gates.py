"""Gate atoms — short-circuit a Job chain based on config or ctx state.

Each Job in renquant_104 has a `should_skip(ctx)` hook, but in-chain
gates are useful for finer control (e.g. skip remaining tasks after
a guard fires). These atoms let a Job declare its skip conditions
declaratively without writing custom skip logic.
"""
from __future__ import annotations

from typing import Any

from ..pipeline import Task
from .ctx_ops import _get_path


class SkipIfConfigDisabledTask(Task):
    """Return False (stop chain) if `ctx.config.<dotted_path>` is falsy.

    Default-aware: missing keys read as None, which is falsy.
    """

    def __init__(self, config_path: str, default: Any = None):
        self.config_path = config_path
        self.default = default

    @property
    def name(self) -> str:
        return f"SkipIfConfigDisabled({self.config_path})"

    def run(self, ctx) -> bool | None:
        cfg = getattr(ctx, "config", {}) or {}
        v = _get_path(cfg, self.config_path, self.default)
        if not v:
            return False


class SkipIfFieldFalsyTask(Task):
    """Return False if `ctx.<field>` is falsy (None, 0, empty)."""

    def __init__(self, field: str):
        self.field = field

    @property
    def name(self) -> str:
        return f"SkipIfFieldFalsy({self.field})"

    def run(self, ctx) -> bool | None:
        v = _get_path(ctx, self.field)
        if not v:
            return False


class SkipIfFieldEqualsTask(Task):
    """Return False if `ctx.<field>` equals `value`."""

    def __init__(self, field: str, value: Any):
        self.field, self.value = field, value

    @property
    def name(self) -> str:
        return f"SkipIfFieldEquals({self.field}, {self.value!r})"

    def run(self, ctx) -> bool | None:
        v = _get_path(ctx, self.field)
        if v == self.value:
            return False


__all__ = [
    "SkipIfConfigDisabledTask",
    "SkipIfFieldFalsyTask",
    "SkipIfFieldEqualsTask",
]
