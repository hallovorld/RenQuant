"""PreflightTask / PreflightJob / PreflightPipeline base classes.

Built on top of the canonical ``kernel.pipeline.Task`` ABC so the Preflight
layer shares the project-wide ${1c}-aligned vocabulary.
"""
from __future__ import annotations

import logging
from abc import abstractmethod

from kernel.pipeline.pipeline import Task

from .ctx import PreflightContext

log = logging.getLogger("preflight_pipeline")


class PreflightTask(Task):
    """One preflight gate.

    Subclasses set ``check_name`` (e.g. "P-STATE-FILE") and implement
    ``check(ctx) -> PreflightCheck``. The Task's ``run`` wraps ``check``
    with the contract:
      - any unexpected exception → fail-closed (or sell-only soft if
        ``ctx.run_mode`` permits — handled by PreflightPipeline, not here)
      - the produced PreflightCheck is appended to ``ctx.results``
      - the ``run`` return value follows the canonical Task contract:
        ``False`` means "stop chain" (used only for hard-fail short-circuit
        in strict mode if a future refactor wants it). Default is ``True``
        so the orchestrator (PreflightPipeline) decides global pass/fail.

    The check_name is also the symbolic gate id (matches legacy log output).
    """

    check_name: str = ""

    @abstractmethod
    def check(self, ctx: PreflightContext):  # -> PreflightCheck
        """Subclass impl. Must return a PreflightCheck.

        No side effects on ctx beyond the result append done in ``run``.
        """

    def run(self, ctx: PreflightContext) -> bool | None:
        # Legacy parity: append the result and log a marker line, regardless of
        # outcome. Strict-mode hard-fail enforcement is done by PreflightPipeline
        # AFTER all checks have run (matches run_preflight semantics).
        from kernel.preflight import (  # noqa: PLC0415  (legacy bridge)
            PreflightCheck,
            _is_sell_only_run,
        )
        try:
            result = self.check(ctx)
        except Exception as exc:  # noqa: BLE001  legacy parity
            sell_only = _is_sell_only_run(ctx.run_mode)
            result = PreflightCheck(
                self.check_name,
                "soft" if sell_only else "hard",
                True if sell_only else False,
                f"check raised unexpectedly: {exc}; "
                + (
                    "sell-only risk exits are allowed"
                    if sell_only else
                    "full/buy preflight fails closed"
                ),
            )
        ctx.append(result)
        marker = "✓" if result.ok else "✗"
        log.info(
            "preflight %s %-22s [%s] %s",
            marker, result.name, result.severity.upper(), result.message,
        )
        return True


class PreflightJob:
    """Group of related PreflightTasks. Runs every Task in declaration order.

    Unlike inference Jobs there's no ``should_skip`` analogue here — every
    preflight gate runs every time. Even if one fails, the rest are evaluated
    so the operator sees the full slate of failures.
    """

    tasks: list[PreflightTask] = []

    def run(self, ctx: PreflightContext) -> None:
        for task in self.tasks:
            task.run(ctx)


class PreflightPipeline:
    """Orchestrates PreflightJobs and enforces strict-mode hard-fail.

    Behaviour mirrors ``kernel.preflight.run_preflight``:
      - runs every Job → every Task → every check, appending to ctx.results
      - at the end, if any HARD check failed and strict=True, raise
        PreflightFailed
    """

    def __init__(self, jobs: list[PreflightJob]) -> None:
        self.jobs = jobs

    def run(self, ctx: PreflightContext, *, strict: bool = True) -> list:
        from kernel.preflight import PreflightFailed  # noqa: PLC0415

        ctx.results = []
        for job in self.jobs:
            job.run(ctx)
        hard_failures = [r for r in ctx.results if r.severity == "hard" and not r.ok]
        if hard_failures and strict:
            raise PreflightFailed(hard_failures)
        return ctx.results
