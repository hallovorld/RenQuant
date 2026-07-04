"""Z9 broker-side stop orders — runner.py decomposition slice 3 (order_emit).

EXTRACTED 2026-06-13 from adapters/runner.py (eng plan S2 item 5:
"Decompose runner.py: state_store / broker_sync / order_emit /
reporting"). The protective-order emission path — placement, never-loosen
replacement, cancellation. Line-faithful, test-gated (the sim replay does
not cover the live adapter); self-deps parameterized (broker, the
stop_orders cache dict mutated in place), same logger.

Z9 invariant (G1, 2026-06-12): broker-resident GTC stops are the only
protection that survives a dead box, so the stop distance is a FAR
catastrophe line, never the 6% intraday cap (see z9_stop_pct).
"""
from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger("live.runner")  # same logger as the runner —
                                        # log contract unchanged by the move


def z9_enabled(broker: Any, ctx: Any) -> bool:
    cfg = ctx.config.get("live", {}).get("broker_side_stops", {})
    if not cfg.get("enabled", False):
        return False
    if not getattr(broker, "supports_broker_side_stops", lambda: False)():
        log.debug("Z9: broker %s does not support broker-side stops — skip",
                  type(broker).__name__)
        return False
    return True


def z9_stop_pct(ctx: Any) -> float:
    """Broker-side stop distance.

    2026-06-12 G1 (dead-box catastrophe line): when
    ``live.broker_side_stops.pct`` is set, it OVERRIDES the per-regime
    intraday cap. Rationale: every in-process stop (SDL/trailing/
    protection) dies with this machine; the broker-resident GTC stop is
    the only protection that survives a dead box. It must therefore be a
    FAR catastrophe line (e.g. 0.20), not the 6% intraday cap — a 6%
    broker stop on a 119%-vol name whipsaws on noise (the NVTS-class
    winner-crystallization pathology the sigma-aware SDL exists to avoid).
    Legacy behavior (per-regime max_single_day_loss_pct, default 6%) is
    unchanged when the key is absent.
    """
    z9_cfg = ctx.config.get("live", {}).get("broker_side_stops", {})
    pct = z9_cfg.get("pct")
    if pct is not None:
        try:
            pct_f = float(pct)
            if 0.0 < pct_f < 1.0:
                return pct_f
        except (TypeError, ValueError):
            pass
    regime_p = ctx.config.get("regime_params", {}).get(ctx.regime, {})
    return float(regime_p.get("max_single_day_loss_pct", 0.06))


def place_or_replace_stop(
    broker: Any, stop_orders: dict, ticker: str, qty: float,
    reference_price: float, today_str: str, ctx_pct: float = 0.06,
    software_stops: Any = None,
) -> None:
    """Place a stop at reference × (1 - pct). If a stop already exists for
    this ticker, cancel it first; the new stop_price is the MIN of
    (existing, new) so we never loosen. Mutates ``stop_orders`` in place.
    """
    # 2026-05-09 audit fix (Z9-NaN): pre-fix, NaN qty / reference_price
    # slipped past `<= 0` (NaN comparisons return False) → target=NaN
    # → broker.place_stop_order crashed inside int(qty). Same QTY-NaN
    # pattern as the exit-side audit fix. Now: explicit isfinite guard.
    if (not math.isfinite(qty) or qty <= 0
            or not math.isfinite(reference_price) or reference_price <= 0):
        log.warning(
            "Z9: skipping stop for %s — non-finite or non-positive "
            "qty=%s reference_price=%s", ticker, qty, reference_price,
        )
        return
    # S-FRAC stage 0 (design 2026-07-02 §2.2.2): qty-aware stop routing.
    # This is the consumer of the per-quantity capability signature that
    # v1 (renquant-execution#19 round 2) built but never wired — the
    # capability is re-evaluated HERE, at placement time, against the
    # CURRENT held quantity (never a cached pre-restart value). A
    # fractional qty cannot ride a broker-resident GTC stop at this
    # broker (fractional = TIF DAY only, §4); it routes to the stage-3
    # software-stop registry, and with stage 3 absent it is loudly
    # UNPROTECTABLE — never place a silently-truncated whole-share stop.
    # Entries can't reach this state (fail-closed upstream in commit);
    # this guards externally-acquired fractional positions.
    from adapters.commit_contract import fmt_qty, route_stop_protection  # noqa: PLC0415

    route = route_stop_protection(broker, ticker, qty, software_stops)
    if route == "software":
        # S-FRAC stage 3 (sprint D2): the armed software-stop registry
        # owns this qty — REGISTER the stop there, same stop-distance
        # math as the broker path. The registry enforces its own
        # never-loosen invariant (ratchet-only; adapters/software_stops).
        log.info(
            "Z9: stop for %s qty=%s routed to the software-stop layer "
            "(broker-side stop does not cover this quantity).",
            ticker, fmt_qty(qty),
        )
        if not math.isfinite(ctx_pct) or ctx_pct <= 0 or ctx_pct >= 1:
            ctx_pct = 0.06
        sw_target = reference_price * (1.0 - ctx_pct)
        if not math.isfinite(sw_target) or sw_target <= 0:
            log.warning("Z9: derived software-stop target=%s non-finite — skipping",
                        sw_target)
            return
        register = getattr(software_stops, "register", None)
        if not callable(register):
            # Armed (is_armed() is True) but no write surface — a stage-3
            # contract violation. Loud: the position is NOT protected.
            log.error(
                "Z9: software-stop layer is ARMED but exposes no "
                "register(); stop for %s qty=%s NOT recorded — position "
                "is NOT stop-protected. Stage-3 registry contract "
                "violation (adapters/software_stops).",
                ticker, fmt_qty(qty),
            )
            return
        try:
            entry = register(ticker, float(qty), float(sw_target),
                             source="z9", today_str=today_str)
        except Exception as exc:
            log.error(
                "Z9: software-stop register(%s, qty=%s, stop=%.2f) FAILED: "
                "%s — position is NOT stop-protected.",
                ticker, fmt_qty(qty), sw_target, exc,
            )
            return
        log.info(
            "Z9: %s software stop registered @ $%.2f × %s shares "
            "(never-loosen: registry ratchets up only)",
            ticker, float(entry.get("stop_price", sw_target)), fmt_qty(qty),
        )
        return
    if route != "broker":
        log.error(
            "Z9: broker-side stop UNAVAILABLE for %s qty=%s (route=%s) — "
            "a fractional quantity cannot be protected by a broker-resident "
            "GTC stop; the software-stop registry is S-FRAC stage 3. NOT "
            "placing a truncated stop. Position is not broker-stop-"
            "protected; fractional entries are fail-closed upstream.",
            ticker, fmt_qty(qty), route,
        )
        return
    if not math.isfinite(ctx_pct) or ctx_pct <= 0 or ctx_pct >= 1:
        ctx_pct = 0.06
    target = reference_price * (1.0 - ctx_pct)
    if not math.isfinite(target) or target <= 0:
        log.warning("Z9: derived target=%s non-finite — skipping", target)
        return

    existing = stop_orders.get(ticker)
    if existing is not None:
        # Never loosen — pick the tighter of current vs proposed.
        target = min(target, float(existing.get("stop_price", target)))
        try:
            broker.cancel_order(existing.get("order_id", ""))
        except Exception as exc:
            log.warning("Z9: cancel existing stop %s for %s failed: %s",
                        existing.get("order_id"), ticker, exc)
        stop_orders.pop(ticker, None)

    try:
        result = broker.place_stop_order(ticker, qty, target)
    except Exception as exc:
        log.warning("Z9: place_stop_order(%s, qty=%s, stop=%.2f) failed: %s",
                    ticker, qty, target, exc)
        return
    stop_orders[ticker] = {
        "order_id":   result.get("order_id"),
        "stop_price": float(target),
        "qty":        float(qty),
        "stamped_at": today_str,
    }
    # S-FRAC stage 0: the old int(qty) display cast truncated a fractional
    # quantity in the log; fmt_qty is byte-identical for whole shares.
    log.info("Z9: %s stop placed @ $%.2f × %s shares (order=%s)",
             ticker, target, fmt_qty(qty), result.get("order_id"))


def cancel_stop(broker: Any, stop_orders: dict, ticker: str,
                reason: str = "", software_stops: Any = None) -> None:
    """Cancel and forget the stop for a ticker. No-op if none exists.
    Mutates ``stop_orders`` in place.

    S-FRAC stage 3: also disarms any software-stop registry entry for
    the ticker (full liquidation / stop_orders GC). A corrupt registry
    refuses the write and logs — never silently mutated."""
    if software_stops is not None:
        dereg = getattr(software_stops, "deregister", None)
        if callable(dereg):
            try:
                dereg(ticker, reason=reason or "Z9 cancel")
            except Exception as exc:
                log.error(
                    "Z9: software-stop deregister for %s failed: %s "
                    "(registry entry may be stale — STATE-GC will retry)",
                    ticker, exc,
                )
    existing = stop_orders.pop(ticker, None)
    if existing is None:
        return
    try:
        broker.cancel_order(existing.get("order_id", ""))
        log.info("Z9: cancelled stop %s for %s (%s)",
                 existing.get("order_id"), ticker, reason or "no reason")
    except Exception as exc:
        log.warning("Z9: cancel stop %s for %s failed: %s",
                    existing.get("order_id"), ticker, exc)
