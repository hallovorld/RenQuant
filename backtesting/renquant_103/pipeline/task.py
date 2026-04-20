"""Parallel task runner using ThreadPoolExecutor."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class TaskResult:
    name: str
    result: Any
    error: Exception | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def run_tasks(
    tasks: list[tuple[str, Callable]],
    max_workers: int = 8,
) -> list[TaskResult]:
    """Run (name, fn) pairs in parallel; return results in submission order.

    Each *fn* is called with no arguments.  Exceptions are captured in
    TaskResult.error rather than re-raised so all tasks complete before
    the caller inspects errors.
    """
    if not tasks:
        return []

    n = min(max_workers, len(tasks))
    results: list[TaskResult | None] = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=n) as pool:
        future_to_idx = {
            pool.submit(fn): (i, name)
            for i, (name, fn) in enumerate(tasks)
        }
        for future in as_completed(future_to_idx):
            i, name = future_to_idx[future]
            try:
                results[i] = TaskResult(name=name, result=future.result())
            except Exception as exc:
                results[i] = TaskResult(name=name, result=None, error=exc)

    return results  # type: ignore[return-value]
