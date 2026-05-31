"""P-BROKER-CONNECT — connect + get_account_value works."""
from __future__ import annotations

from kernel.preflight import PreflightCheck  # noqa: PLC0415 (legacy bridge)

from ..base import PreflightTask
from ..ctx import PreflightContext


class BrokerConnectTask(PreflightTask):
    """P-BROKER-CONNECT — broker.connect() + broker.get_account_value() succeed.

    Behavior parity with ``kernel.preflight._check_broker_connect``:
      - ctx.broker is None → soft pass ("dry-run; skip")
      - any exception during connect/get_account_value → HARD FAIL
      - both succeed → HARD PASS, message includes equity
    """

    check_name = "P-BROKER-CONNECT"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        if ctx.broker is None:
            return PreflightCheck(
                self.check_name, "soft", True, "no broker (dry-run); skip",
            )
        try:
            ctx.broker.connect()
            eq = float(ctx.broker.get_account_value())
            return PreflightCheck(
                self.check_name, "hard", True,
                f"broker connected, equity=${eq:.2f}",
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightCheck(
                self.check_name, "hard", False,
                f"broker connect failed: {exc}",
            )
