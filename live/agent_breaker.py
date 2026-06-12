"""G2 agent breaker — adapter-level order admission caps (Week-0 disaster guard).

Design: doc/research/2026-06-12-engineering-architecture-deep-plan.md
(renquant-orchestrator) §0 Week-0, §II.2 layer L4/L6 boundary, §III.4
"Disaster guards"; prototype with property proofs:
scripts/engineering/agent_breaker_prototype.py (epic graduation, PR #112).

Sits BELOW all pipeline logic, wired into AlpacaBroker order submission:
a runaway agent/pipeline loop cannot exceed a hard daily order count, a
hard daily notional, or a manual TRADING_OFF file — no matter what any
upstream layer decides. State is process-local by design: the caps bound a
single runaway PROCESS; cross-process daily totals are bounded by
(processes × cap), still finite and small.
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

log = logging.getLogger("live.agent_breaker")

TRADING_OFF_FLAG = Path("/Users/renhao/git/github/RenQuant/TRADING_OFF")
MAX_ORDERS_PER_DAY = 25
MAX_NOTIONAL_PER_DAY = 5_000.0


class BreakerTripped(RuntimeError):
    """Raised before submission. Callers must treat as fatal for the order
    and MUST NOT retry-loop (design §III.5 margin framework: rejection is a
    de-risk signal)."""


class AgentBreaker:
    def __init__(self, *, max_orders_per_day: int = MAX_ORDERS_PER_DAY,
                 max_notional_per_day: float = MAX_NOTIONAL_PER_DAY,
                 off_flag: Path = TRADING_OFF_FLAG):
        self.max_orders = int(max_orders_per_day)
        self.max_notional = float(max_notional_per_day)
        self.off_flag = off_flag
        self._day: dt.date | None = None
        self._orders = 0
        self._notional = 0.0

    def _roll(self, today: dt.date) -> None:
        if self._day != today:
            self._day, self._orders, self._notional = today, 0, 0.0

    def admit(self, *, symbol: str, notional: float | None,
              today: dt.date | None = None) -> None:
        """Call exactly once immediately before each broker submission.

        notional=None (price unknown, e.g. market order without a reference
        price) still consumes an order slot; notional accounting skips it.
        """
        if self.off_flag.exists():
            raise BreakerTripped(
                f"manual TRADING_OFF present at {self.off_flag} — all order "
                f"submission disabled (delete the file to re-enable)")
        if today is None:
            from live.clock import trading_date  # noqa: PLC0415

            today = trading_date()  # P0.3: exchange date, not midnight-PT
        self._roll(today)
        if self._orders + 1 > self.max_orders:
            raise BreakerTripped(
                f"G2 daily order cap {self.max_orders} reached "
                f"(symbol={symbol}) — runaway guard")
        if notional is not None and self._notional + abs(notional) > self.max_notional:
            raise BreakerTripped(
                f"G2 daily notional cap ${self.max_notional:,.0f} would be "
                f"exceeded (${self._notional:,.0f} + ${abs(notional):,.0f}, "
                f"symbol={symbol})")
        self._orders += 1
        if notional is not None:
            self._notional += abs(notional)
        log.debug("G2 admit %s: orders=%d/%d notional=%.0f/%.0f",
                  symbol, self._orders, self.max_orders,
                  self._notional, self.max_notional)
