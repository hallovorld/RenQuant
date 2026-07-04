"""S-FRAC stage 0 — the fractional-capable commit quantity contract.

Design: renquant-orchestrator doc/design/2026-07-02-s-frac-fractional-v2.md
(§2.2 stage-0 deliverable, §2.3 audit tests). This module is the single
authority for how ``RunnerAdapter.commit`` (the ACTIVE live path — see the
v1 post-mortem: capability built on ``ExecutionPipeline`` was a non-active
path) treats order/fill quantities:

1. **Quantity contract** (§2.2.1): broker ``filled_qty`` is authoritative
   and preserved at float precision end-to-end. Whole-share fills snap to
   ``int`` inside :func:`normalize_fill_qty` — the ONE sanctioned
   whole-share branch — so flag-off behavior is byte-identical to the
   legacy ``int(execution["filled_qty"] or shares)`` line this contract
   killed (runner.py, v1 blocker cited by Codex on renquant-pipeline#153).

2. **Stop routing contract** (§2.2.2): Z9 call sites route protection per
   held quantity via :func:`route_stop_protection`. A fractional quantity
   cannot be protected by a broker-resident GTC stop at this broker
   (Alpaca fractional orders are TIF=DAY only — design §4), so it routes
   to the software-stop registry (stage 3). Until stage 3 exists, a
   fractional holding with no software-stop layer is a FAIL-CLOSED
   condition at entry: the buy is never submitted. This is the
   machine-verifiable ordering guard — stage-2 sizing cannot activate
   ahead of stage-3 protection — and it makes the stage-0 outage-window
   loss budget $0 by construction (§2.3).

3. **Capability gate** (§2.2.3, the strategy#36 blocker closed):
   :func:`fractional_capability_gate` is the machine-verifiable preflight
   replacing prose merge-ordering. ``execution.fractional_shares.enabled=
   true`` requires (a) the broker adapter to expose the fractional
   contract (``is_fractionable`` + no-submit classification, from
   renquant-execution#19) and (b) the software-stop layer to report
   itself armed (stage 3). Either missing ⇒ fail-closed before any BUY
   is emitted, with a dedicated audit reason (exits are never blocked).

4. **Active-path liveness** (§2.3): :func:`commit_path_fingerprint` is
   stamped onto the ctx by ``RunnerAdapter.commit`` and recorded in the
   run bundle, so "the live runner exercises the contract-carrying commit
   path" is a recorded fact per run — the direct anti-regression for
   merged-is-not-deployed / deployed-but-dark.

Stage 0 is default-inert: no flag is enabled anywhere; whole-share
behavior is regression-pinned byte-identical
(tests/test_s_frac_stage0_commit_contract.py).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# Contract version tag. Bump when the quantity/stop-routing/capability
# semantics change; the run bundle records it per run.
COMMIT_QTY_CONTRACT = "fractional-v2-stage0"

# A fill quantity within this distance of an integer is a whole-share
# fill (broker float noise like 5.000000001 must not flip the branch;
# real fractional fills are >= 1e-6 away from an integer per Alpaca's
# 6-9dp quantity grid).
QTY_INTEGRAL_EPS = 1e-9


def is_integral_qty(qty: Any) -> bool:
    """True iff *qty* is a finite whole-share quantity (within eps)."""
    try:
        q = float(qty)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(q):
        return False
    return abs(q - round(q)) <= QTY_INTEGRAL_EPS


def normalize_fill_qty(filled_qty: Any, fallback: Any) -> float | int:
    """Broker-authoritative fill quantity, float-preserving (§2.2.1).

    Replaces the legacy truncation ``int(execution["filled_qty"] or
    shares)`` on the RunnerAdapter buy path — the exact line that turned
    a broker fill of 0.435578 into 0 shares in orders_placed, live_state,
    the trade journal, cash accounting, and the Z9 stop quantity.

    Semantics:
      * falsy/zero ``filled_qty`` falls back to *fallback* (unchanged
        legacy behavior — broker_order_execution only reaches this line
        with a confirmed fill);
      * whole-share quantities return ``int`` — the ONE sanctioned
        whole-share branch, keeping flag-off order dicts / journal rows /
        JSON bytes identical to the legacy path;
      * fractional quantities return the broker float VERBATIM (never
        re-derived from notional/price, never rounded).
    """
    q = filled_qty if filled_qty else fallback
    try:
        q = float(q)
    except (TypeError, ValueError):
        q = 0.0
    if not math.isfinite(q):
        q = 0.0
    r = round(q)
    if abs(q - r) <= QTY_INTEGRAL_EPS:
        # whole-share branch: int formats/compares identically to the
        # legacy int() cast for every integral fill.
        return int(r)
    return q


def fmt_qty(qty: Any) -> str:
    """Render a quantity for logs: '5' for whole shares (byte-identical
    to the legacy %d / %.0f formatting), full-precision float otherwise."""
    if is_integral_qty(qty):
        return str(int(round(float(qty))))
    try:
        return repr(float(qty))
    except (TypeError, ValueError):
        return repr(qty)


def software_stops_armed(software_stops: Any) -> bool:
    """Machine probe for the stage-3 software-stop layer (fail-closed).

    Stage 3 (design §3.2) will attach a registry object to the adapter;
    the contract it must satisfy to count as ARMED is an ``is_armed()``
    method returning exactly True. Anything else — absent layer, missing
    method, raising method, truthy-but-not-True return — is NOT armed.
    """
    if software_stops is None:
        return False
    probe = getattr(software_stops, "is_armed", None)
    if not callable(probe):
        return False
    try:
        return probe() is True
    except Exception:  # noqa: BLE001 — a raising stop layer is not armed
        return False


def supports_broker_side_stops_for(broker: Any, symbol: str, qty: Any) -> bool:
    """Qty-aware Z9 stop capability (§2.2.2).

    The v1 renquant-execution#19 round-2 blocker: the per-quantity
    capability signature ``supports_broker_side_stops(symbol, qty)`` was
    built but never consumed by the real Z9 caller (which called the
    no-arg form). This helper is the consumer.

    Fail-closed rules:
      * broker exposes the qty-aware signature → its answer is
        authoritative;
      * legacy no-arg broker → a fractional qty is NOT protectable there
        (the legacy stop path submits whole-share GTC stops; a fractional
        qty would be truncated or rejected) → False; an integral qty
        falls back to the legacy no-arg answer;
      * missing/raising probe → False.
    """
    probe = getattr(broker, "supports_broker_side_stops", None)
    if not callable(probe):
        return False
    try:
        return bool(probe(symbol, qty))
    except TypeError:
        # Legacy no-arg signature.
        if not is_integral_qty(qty):
            return False
        try:
            return bool(probe())
        except Exception:  # noqa: BLE001
            return False
    except Exception:  # noqa: BLE001
        return False


def route_stop_protection(
    broker: Any, symbol: str, qty: Any, software_stops: Any = None,
) -> str:
    """Select the protection layer for a held quantity (§2.2.2).

    Returns one of:
      * ``"broker"``       — broker-resident stop covers this qty;
      * ``"software"``     — stage-3 software-stop registry is armed;
      * ``"unprotectable"``— neither layer can protect this qty. The
        caller must fail closed (entries: do not submit the buy; existing
        positions: log loudly — never place a truncated stop).
    """
    if supports_broker_side_stops_for(broker, symbol, qty):
        return "broker"
    if software_stops_armed(software_stops):
        return "software"
    return "unprotectable"


def fractional_capability_gate(
    config: dict | None, broker: Any, software_stops: Any = None,
) -> dict[str, Any]:
    """Machine-verifiable preflight for ``execution.fractional_shares``
    (§2.2.3 — the strategy#36 "prose-only dependency gate" blocker).

    With the flag OFF (absent/false — all of stage 0-2), the gate is
    trivially ok and inert. With the flag ON, BOTH capabilities must be
    present or the commit path fail-closes BUY emission before any order
    reaches the broker:

      (a) ``broker_fractional_contract`` — the broker adapter exposes the
          renquant-execution#19 contract: callable ``is_fractionable``
          plus a no-submit classifier (``classify_broker_result`` or
          ``is_no_submit_status``);
      (b) ``software_stop_layer`` — the stage-3 layer reports armed via
          :func:`software_stops_armed`.
    """
    exec_cfg = (config or {}).get("execution") or {}
    frac_cfg = exec_cfg.get("fractional_shares") or {}
    enabled = bool(frac_cfg.get("enabled", False))
    missing: list[str] = []
    if enabled:
        has_lookup = callable(getattr(broker, "is_fractionable", None))
        has_no_submit = callable(
            getattr(broker, "classify_broker_result", None)
        ) or callable(getattr(broker, "is_no_submit_status", None))
        if not (has_lookup and has_no_submit):
            missing.append("broker_fractional_contract")
        if not software_stops_armed(software_stops):
            missing.append("software_stop_layer")
    return {
        "contract": COMMIT_QTY_CONTRACT,
        "enabled": enabled,
        "ok": not missing,
        "missing": missing,
    }


def fractional_entry_fail_closed_reason(
    shares: Any,
    gate: dict[str, Any],
    *,
    broker: Any,
    symbol: str,
    software_stops: Any = None,
) -> str | None:
    """Decide whether a BUY intent must fail closed (§2.2.2 + §2.2.3).

    Returns None (proceed) or a dedicated skip_reason string recorded in
    ``ctx.orders_skipped`` — the audit trail the design requires. The
    stage-0 invariant this enforces: **no fractional BUY ever reaches the
    broker while the software-stop layer is absent**, so the stage-0
    outage-window loss budget is $0 by construction (§2.3).
    """
    # Gate-level fail-close: flag ON with a capability missing means the
    # config landed ahead of its dependencies (the #36 failure mode) —
    # no BUY may be emitted at all, integral or not.
    if gate.get("enabled") and not gate.get("ok"):
        return "fractional_capability_gate_failed:" + ",".join(
            gate.get("missing") or [],
        )
    try:
        q = float(shares)
    except (TypeError, ValueError):
        return None  # existing bad-order guards own this case
    if not math.isfinite(q) or q <= 0:
        return None  # existing finite/positive guards own this case
    if is_integral_qty(q):
        return None  # whole-share entry: stage-0 behavior unchanged
    if not gate.get("enabled"):
        # A fractional intent with the flag off is a contract violation
        # upstream (stage-2 sizing leaked past its flag) — never submit.
        return "fractional_intent_flag_off"
    if route_stop_protection(broker, symbol, q, software_stops) == "unprotectable":
        return "fractional_entry_unprotectable_no_stop_layer"
    return None


def commit_path_fingerprint() -> dict[str, Any]:
    """Identify the commit implementation carrying this contract (§2.3).

    ``RunnerAdapter.commit`` stamps this onto the ctx at the top of every
    commit; ``build_run_bundle`` records it in the persisted run bundle.
    A daily-full bundle carrying ``contract == "fractional-v2-stage0"``
    plus the source hash of the executed runner is the active-path
    liveness proof: "the live runner exercises the new path" becomes a
    recorded fact per run, not an assumption.
    """
    import hashlib  # noqa: PLC0415

    here = Path(__file__).resolve()
    runner = here.with_name("runner.py")

    def _sha(p: Path) -> str | None:
        try:
            return hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            return None

    return {
        "contract": COMMIT_QTY_CONTRACT,
        "commit_impl": "adapters.runner.RunnerAdapter.commit",
        "module_file": str(runner),
        "runner_sha256": _sha(runner),
        "commit_contract_sha256": _sha(here),
    }
