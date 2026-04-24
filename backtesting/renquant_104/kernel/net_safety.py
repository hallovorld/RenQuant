"""Network-call safety primitives.

Problem this solves (2026-04-23 incident): `daily_104.sh` hung for 10+
minutes in PanelDataJob's fetch phase (yfinance `.info` / `.earnings_dates`
calls), accumulating 19 CLOSE_WAIT sockets to Yahoo. yfinance reuses
a requests Session with NO timeout — when Yahoo slow-drips a response,
the socket waits forever. Two consecutive e2e runs hit this, blocking
live trading decisions.

This module wraps any network-touching callable with:

  1. Hard per-call timeout (default 20 s) via concurrent.futures.
     Gives up cleanly, returns None, lets caller move on.
  2. A fresh daemon thread per call, so a stuck call can't outlive the
     parent process (previously a shared ThreadPoolExecutor with
     non-daemon threads kept pytest-xdist workers alive).
  3. Structured logging with a label so hangs are diagnosable.
  4. Optional global "fetch budget" — cap cumulative fetch time per
     task. Once the budget is consumed, remaining calls short-circuit
     to None immediately (no more network attempts this task).

Usage:

    from kernel.net_safety import call_with_timeout, FetchBudget

    # Per-call timeout
    info = call_with_timeout(
        lambda: yf.Ticker(sym).info,
        timeout_sec = 15.0,
        label       = f"yf.info({sym})",
    )
    if info is None:
        # timeout or exception — already logged, just skip
        return

    # Optional: wrap a whole loop with a budget so one bad day can't
    # derail the entire PanelDataJob
    budget = FetchBudget(total_sec=120.0, label="LoadFundamentals")
    for sym in symbols:
        if budget.exhausted():
            break
        info = call_with_timeout(lambda: yf.Ticker(sym).info,
                                  timeout_sec=15.0, label=f"yf.info({sym})",
                                  budget=budget)
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import TimeoutError as _FutTimeout
from typing import Any, Callable

log = logging.getLogger("kernel.net_safety")


# 2026-04-24: one daemon thread per call (no shared pool). Python's
# ThreadPoolExecutor spawns NON-daemon threads, and a stuck worker kept
# pytest-xdist processes alive for the full sleep duration. A per-call
# daemon thread is simpler and ensures the interpreter can exit even
# when a network call is still blocked inside yfinance / OpenBB.
# Overhead: ~10 µs per thread spawn — negligible vs 20 s network calls.


class FetchBudget:
    """Time budget for a batch of network calls.

    Call `.exhausted()` before each call; once True, the caller should
    short-circuit to None rather than issue more network I/O.
    """
    def __init__(self, total_sec: float = 180.0, label: str = "batch"):
        self.total_sec = float(total_sec)
        self.label     = label
        self.consumed  = 0.0

    def charge(self, seconds: float) -> None:
        self.consumed += float(seconds)

    def exhausted(self) -> bool:
        if self.consumed >= self.total_sec:
            log.warning(
                "[%s] fetch budget exhausted: consumed %.1fs ≥ %.1fs — "
                "short-circuiting remaining calls",
                self.label, self.consumed, self.total_sec,
            )
            return True
        return False

    def remaining(self) -> float:
        return max(0.0, self.total_sec - self.consumed)


def call_with_timeout(
    fn: Callable[..., Any],
    *args: Any,
    timeout_sec: float = 20.0,
    label: str = "",
    budget: "FetchBudget | None" = None,
    **kwargs: Any,
) -> Any:
    """Invoke `fn(*args, **kwargs)` with a hard timeout.

    Returns the function's result, or `None` on timeout / exception /
    exhausted budget. Never raises — the goal is that a misbehaving
    network call can't block the calling task.

    If `budget` is provided, the call's actual duration is charged to
    the budget (via `budget.charge`), and if the budget is already
    exhausted the call short-circuits to `None` immediately.
    """
    if budget is not None and budget.exhausted():
        return None

    effective_timeout = timeout_sec
    if budget is not None:
        # Don't exceed remaining budget
        effective_timeout = min(timeout_sec, max(1.0, budget.remaining()))

    t0 = time.monotonic()
    # Per-call daemon thread — see module docstring. Stops the interpreter
    # from hanging at shutdown if `fn` is stuck inside a network library.
    import threading as _th  # noqa: PLC0415
    result_box: dict = {}

    def _runner():
        try:
            result_box["value"] = fn(*args, **kwargs)
        except Exception as exc:      # noqa: BLE001
            result_box["exc"] = exc

    worker = _th.Thread(
        target=_runner, name=f"net-safe-{label or 'anon'}", daemon=True,
    )
    worker.start()
    worker.join(timeout=effective_timeout)

    try:
        if worker.is_alive():
            raise _FutTimeout
        if "exc" in result_box:
            raise result_box["exc"]
        result = result_box.get("value")
        elapsed = time.monotonic() - t0
        if budget is not None:
            budget.charge(elapsed)
        if elapsed > timeout_sec * 0.8:
            # Borderline slow — flag even on success so we can tune
            log.info("[%s] slow: %.1fs (budget=%.1fs)",
                     label or "call_with_timeout", elapsed, timeout_sec)
        return result
    except _FutTimeout:
        elapsed = time.monotonic() - t0
        if budget is not None:
            budget.charge(elapsed)
        log.warning(
            "[%s] TIMEOUT after %.1fs (limit=%.1fs) — abandoning; "
            "daemon thread may still run briefly in background",
            label or "call_with_timeout", elapsed, effective_timeout,
        )
        return None
    except Exception as exc:
        elapsed = time.monotonic() - t0
        if budget is not None:
            budget.charge(elapsed)
        log.warning("[%s] FAILED after %.1fs — %s",
                    label or "call_with_timeout", elapsed, exc)
        return None


__all__ = ["call_with_timeout", "FetchBudget"]
