"""SoftwareStopExitTask — the sell-only loop's software-stop pass.

S-FRAC stage 3 core (sprint D2). Design: renquant-orchestrator
doc/design/2026-07-02-s-frac-fractional-v2.md §3.2.3 — "the sell-only
loop evaluates registry entries each pass exactly like its existing
stop-loss rule and emits a fractional market SELL on breach".

The registry (adapters/software_stops.SoftwareStopRegistry) is attached
to ctx by RunnerAdapter.make_context as ``ctx.software_stops``; it is
None unless ``execution.software_stops.enabled`` — in which case this
task is a no-op (flag-off byte-inert).

Placement in SellOnlyPipeline: AFTER MetaLabelVetoTask and
LimitSellsPerBarTask, by design not by accident. The software stop is
the loop-resident mirror of a broker-resident Z9 GTC stop, and a broker
stop can neither be vetoed by the meta-label model nor capped by the
per-bar sell limit — so its mirror is not either. Defense in depth: the
``software_stop`` exit type is also in the exit-type taxonomy's
PANEL_VETO_BYPASS / PER_BAR_CAP_EXEMPT sets (kernel/exit_types.py), so
even a re-ordering could not silently subject it to those gates.
"""
from __future__ import annotations

import logging

from kernel.exits import ExitSignal
from kernel.pipeline.pipeline import Task

log = logging.getLogger("kernel.pipeline")

EXIT_TYPE_SOFTWARE_STOP = "software_stop"


class SoftwareStopExitTask(Task):
    """Evaluate the software-stop registry against this pass's prices and
    queue a market exit for the FULL registered qty on every breach."""

    def should_skip(self, ctx) -> bool:  # noqa: ANN001
        return getattr(ctx, "software_stops", None) is None

    def run(self, ctx) -> "bool | None":  # noqa: ANN001
        registry = getattr(ctx, "software_stops", None)
        if registry is None:
            return True
        armed_probe = getattr(registry, "is_armed", None)
        if not callable(armed_probe) or armed_probe() is not True:
            # Corrupt / mis-armed layer: registered stops cannot be
            # checked. New fractional entries are already blocked by the
            # stage-0 capability gate (software_stops_armed is False);
            # existing ones are an operator page (watchdog exit 2), and
            # this log fires every 12-minute pass until resolved.
            log.error(
                "SOFTWARE-STOP layer present but NOT armed — registered "
                "stops NOT evaluated this pass; new fractional entries are "
                "blocked by the stage-0 capability gate. Run "
                "scripts/check_software_stops_liveness.py.",
            )
            return True
        intents = registry.evaluate(getattr(ctx, "prices", None) or {})
        for intent in intents:
            symbol = intent["symbol"]
            # Full registered qty. commit()'s is_full_liquidate_signal
            # treats quantity >= held as full liquidation (wash-sale
            # stamp + Z9 cancel + registry deregister); a registered qty
            # below the held qty sells exactly the protected quantity.
            ctx.exits.append((
                symbol,
                ExitSignal(
                    should_exit=True,
                    reason=intent["reason"],
                    exit_type=EXIT_TYPE_SOFTWARE_STOP,
                    quantity=float(intent["qty"]),
                ),
            ))
        if intents:
            log.warning(
                "SOFTWARE-STOP pass: %d breach exit(s) queued: %s",
                len(intents), ", ".join(i["symbol"] for i in intents),
            )
        return True
