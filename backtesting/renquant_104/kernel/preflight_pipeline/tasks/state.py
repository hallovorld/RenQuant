"""P-STATE-FILE — live_state.{broker}.json parses (or absent)."""
from __future__ import annotations

import json

from kernel.preflight import PreflightCheck  # noqa: PLC0415 (legacy bridge)

from ..base import PreflightTask
from ..ctx import PreflightContext


class StateFileTask(PreflightTask):
    """P-STATE-FILE — live_state.{broker}.json parses (or is absent for first run).

    Behavior parity with ``kernel.preflight._check_state_file``:
      - no broker_name → soft pass ("dry-run; skip")
      - state_paths module unavailable → soft pass ("skip")
      - file absent → soft pass ("first run?")
      - file present + unparseable → HARD FAIL
      - file present + parses → HARD PASS
    """

    check_name = "P-STATE-FILE"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        if not ctx.broker_name:
            return PreflightCheck(
                self.check_name, "soft", True, "no broker_name (dry-run); skip",
            )
        try:
            from kernel.state_paths import resolve_live_state_read  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            return PreflightCheck(
                self.check_name, "soft", True,
                f"state_paths unavailable: {exc}; skip",
            )
        p, _used_legacy = resolve_live_state_read(ctx.strategy_dir, ctx.broker_name)
        if not p.exists():
            return PreflightCheck(
                self.check_name, "soft", True,
                f"state file absent at {p.name} (first run?)",
            )
        try:
            json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            return PreflightCheck(
                self.check_name, "hard", False,
                f"state file unreadable {p.name}: {exc}",
            )
        return PreflightCheck(
            self.check_name, "hard", True, f"loaded {p.name}",
        )
