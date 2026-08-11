"""P-BROKER-CONNECT — connect + get_account_value works."""
from __future__ import annotations

from kernel.preflight import (  # noqa: PLC0415 (legacy bridge)
    PreflightCheck,
    _attempt_broker_connect,
)

from ..base import PreflightTask
from ..ctx import PreflightContext


class BrokerConnectTask(PreflightTask):
    """P-BROKER-CONNECT — broker.connect() + broker.get_account_value() succeed.

    Behavior parity with ``kernel.preflight._check_broker_connect`` — the two
    share the single ``_attempt_broker_connect`` body so they cannot drift:
      - ctx.broker is None → soft pass ("dry-run; skip")
      - connect/get_account_value are retried a BOUNDED number of times with a
        short backoff; the first success → HARD PASS (message includes equity,
        and the attempt count when it took more than one)
      - only after ALL attempts fail → HARD FAIL (fail-closed; message names
        the attempt count)
    """

    check_name = "P-BROKER-CONNECT"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        if ctx.broker is None:
            return PreflightCheck(
                self.check_name, "soft", True, "no broker (dry-run); skip",
            )
        return _attempt_broker_connect(ctx.broker)
