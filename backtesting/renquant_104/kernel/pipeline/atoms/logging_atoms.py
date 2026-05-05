"""Logging + counter atoms — uniform log lines and ctx.counters increments.

Replaces ad-hoc `log.info("FooTask: did X with %s", ...)` patterns
with declarative atoms a Job can compose.
"""
from __future__ import annotations

import logging

from ..pipeline import Task
from .ctx_ops import _get_path


class LogSummaryTask(Task):
    """Emit one log line at the chosen level, formatted from ctx fields.

    template uses %-style placeholders; fields is a tuple of dotted ctx paths
    whose resolved values get plugged in.

    log.info("Job: %d / %d done", x, y)
    LogSummaryTask("Job: %d / %d done", fields=("_n_done", "_n_total"))
    """

    def __init__(
        self,
        template: str,
        fields: tuple[str, ...] = (),
        level: str = "info",
        logger: str = "kernel.pipeline.atoms",
    ):
        self.template = template
        self.fields = fields
        self.level = level
        self._log = logging.getLogger(logger)

    @property
    def name(self) -> str:
        return f"LogSummary({self.level})"

    def run(self, ctx) -> bool | None:
        try:
            args = tuple(_get_path(ctx, f) for f in self.fields)
            line = self.template % args if args else self.template
        except Exception as exc:
            line = f"LogSummaryTask: format failed — {exc}"
        getattr(self._log, self.level, self._log.info)(line)


class IncrementCounterTask(Task):
    """ctx.counters[key] += amount (or amount from a ctx field)."""

    def __init__(
        self,
        counter_key: str,
        amount: int | float | str = 1,
    ):
        self.counter_key = counter_key
        self.amount = amount

    @property
    def name(self) -> str:
        return f"IncrementCounter({self.counter_key})"

    def run(self, ctx) -> bool | None:
        if not hasattr(ctx, "counters"):
            ctx.counters = {}
        if isinstance(self.amount, str):
            inc = _get_path(ctx, self.amount, 0)
        else:
            inc = self.amount
        try:
            inc_n = int(inc) if isinstance(inc, (int, float)) else 0
        except Exception:
            inc_n = 0
        ctx.counters[self.counter_key] = (
            ctx.counters.get(self.counter_key, 0) + inc_n
        )


__all__ = ["LogSummaryTask", "IncrementCounterTask"]
