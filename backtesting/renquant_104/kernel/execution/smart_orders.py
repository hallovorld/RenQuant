"""Smart order execution — limit orders + size-based routing.

Roadmap #8 / C4 (2026-05-18). Pre-2026-05-09 audit found a deleted
smart_orders.py module that was orphaned (per CLAUDE.md §5.13.2). This
is the rebuild, with explicit prod-wiring claim (NOT-YET; integration
gated on integration tests + commit message). For now the module exists
+ has unit tests; opt-in via `live.alpaca_broker.PlaceOrderPolicy.SMART`
once wiring is added (next session).

Reference:
  - Almgren-Chriss 2000 *J. of Risk* "Optimal Execution of Portfolio
    Transactions" — optimal price impact + risk-aversion tradeoff for
    order slicing. Closed-form solution for normal arrival-price model.
  - Bertsimas-Lo 1998 *J. of Financial Markets* "Optimal Control of
    Execution Costs" — dynamic programming formulation.
  - Cont-Stoikov-Talreja 2010 "A stochastic model for order book
    dynamics" — order-book aware limit-order placement.

Public API:
  - `compute_limit_price(side, mid, spread_bps, aggressiveness)`:
    limit price adjusted from mid by a fraction of the spread. Buys
    bid slightly above mid, sells offer slightly below — capturing
    most of the spread vs naive market order.
  - `slice_into_children(parent_qty, adv_shares, max_pct_adv)`:
    split a large parent order into N child orders each ≤
    `max_pct_adv` of ADV. Returns list of child quantities.
  - `child_arrival_schedule(n_children, interval_seconds)`:
    spread N child orders evenly over a horizon. Returns list of
    arrival timestamps (relative seconds).

Design decisions:
  - We do NOT submit child orders via this module. Callers compose:
    1. Plan children with `slice_into_children`
    2. Set limit price with `compute_limit_price`
    3. Schedule arrivals + monitor fills via broker API
  - This keeps execution policy in this pure-function module; broker
    integration stays in `live.alpaca_broker`. Per CLAUDE.md §1c
    (Task / Job / Pipeline split).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChildOrder:
    """A single child order in a sliced parent."""
    quantity: int
    arrival_offset_seconds: float


def compute_limit_price(
    side: str,
    mid: float,
    spread_bps: float,
    aggressiveness: float = 0.5,
) -> float:
    """Limit price adjusted from mid by a fraction of the spread.

    Args:
        side: "BUY" or "SELL"
        mid: midpoint price
        spread_bps: bid-ask spread in basis points (e.g. 5 = 0.05%)
        aggressiveness: ∈ [0, 1]. 0 = post at mid (passive, may not
            fill); 1 = cross full spread (aggressive, certain fill);
            0.5 = pay half the spread (default; captures half the
            cost vs naive market order while still likely to fill in
            seconds).

    Returns: float limit price, rounded to nearest cent.

    Math (BUY):
        bid = mid * (1 - spread_bps/20_000)   # half spread below mid
        ask = mid * (1 + spread_bps/20_000)
        limit_buy = bid + aggressiveness * (ask - bid)
                  = mid + (aggressiveness - 0.5) * spread_bps/10_000 * mid

    Reference: Cont-Stoikov-Talreja 2010 §3 (one-sided limit order book
    fill probability is monotone in price).
    """
    if mid <= 0:
        raise ValueError(f"mid must be positive, got {mid}")
    if not 0.0 <= aggressiveness <= 1.0:
        raise ValueError(f"aggressiveness must be ∈ [0,1], got {aggressiveness}")
    if spread_bps < 0:
        raise ValueError(f"spread_bps must be ≥0, got {spread_bps}")
    side_u = side.upper()
    if side_u not in ("BUY", "SELL"):
        raise ValueError(f"side must be BUY/SELL, got {side!r}")
    half_spread_pct = spread_bps / 20_000.0  # bps → fraction (one-sided)
    # For BUY: 0=passive (at bid), 1=aggressive (at ask)
    # For SELL: 0=passive (at ask), 1=aggressive (at bid)
    if side_u == "BUY":
        limit = mid + (aggressiveness - 0.5) * 2 * half_spread_pct * mid
    else:
        limit = mid - (aggressiveness - 0.5) * 2 * half_spread_pct * mid
    return round(limit, 2)


def slice_into_children(
    parent_qty: int,
    adv_shares: float,
    max_pct_adv: float = 0.01,
) -> list[int]:
    """Split a parent order into children each ≤ max_pct_adv × ADV.

    Args:
        parent_qty: total shares to execute
        adv_shares: average daily volume (shares) for the symbol
        max_pct_adv: max fraction of ADV per child (default 1%).
            Industry-conventional "stealth" threshold per Almgren-Chriss
            2000 §4 — at ≤1% ADV per execution slice, price impact is
            approximately linear and predictable; above that, regime
            shifts (other agents react, market makers widen spreads).

    Returns: list of integer quantities summing to parent_qty.
        Empty list if parent_qty ≤ 0.

    Algorithm:
        max_child_size = floor(adv_shares * max_pct_adv)
        n = ceil(parent_qty / max_child_size)
        Distribute parent_qty evenly across n children (remainder on
        first child for simplicity).
    """
    import math
    if parent_qty <= 0:
        return []
    if adv_shares <= 0 or max_pct_adv <= 0:
        # Degenerate case — no ADV info; submit as single child (no
        # slicing). Caller will then route this single order as-is.
        return [int(parent_qty)]
    max_child = max(1, int(adv_shares * max_pct_adv))
    if parent_qty <= max_child:
        return [int(parent_qty)]
    n = math.ceil(parent_qty / max_child)
    base = parent_qty // n
    remainder = parent_qty - base * n
    # Put remainder on the FIRST child (slightly larger), rest equal.
    children = [int(base + remainder)] + [int(base)] * (n - 1)
    return children


def child_arrival_schedule(
    n_children: int,
    horizon_seconds: float,
) -> list[float]:
    """Evenly-spaced arrival times for n_children over horizon_seconds.

    Returns list of n_children floats: 0, h/(n-1), 2h/(n-1), ..., h.
    Special cases:
        n_children == 0 → []
        n_children == 1 → [0.0]

    For n ≥ 2, first child arrives immediately (t=0) and last at t=horizon,
    matching the equal-spaced VWAP schedule of Almgren-Chriss 2000 §3
    under uniform-volume assumption.

    For VWAP under empirical intraday volume (typically U-shaped), use
    `child_arrival_schedule_volume_weighted` (not yet implemented).
    """
    if n_children < 0:
        raise ValueError(f"n_children must be ≥0, got {n_children}")
    if horizon_seconds < 0:
        raise ValueError(f"horizon_seconds must be ≥0, got {horizon_seconds}")
    if n_children == 0:
        return []
    if n_children == 1:
        return [0.0]
    step = horizon_seconds / (n_children - 1)
    return [i * step for i in range(n_children)]


def plan_execution(
    side: str,
    parent_qty: int,
    mid: float,
    spread_bps: float,
    adv_shares: float,
    horizon_seconds: float = 1800.0,  # 30 min default
    max_pct_adv: float = 0.01,
    aggressiveness: float = 0.5,
) -> list[ChildOrder]:
    """Compose slicing + scheduling into a full execution plan.

    Returns a list of ChildOrder objects. Caller submits each as a
    Limit order at `compute_limit_price(side, mid, spread_bps,
    aggressiveness)` at the scheduled `arrival_offset_seconds`.

    The limit price is constant across children in this simple v1; a
    future version can refresh mid + spread per child for adaptive
    routing.
    """
    quantities = slice_into_children(parent_qty, adv_shares, max_pct_adv)
    if not quantities:
        return []
    times = child_arrival_schedule(len(quantities), horizon_seconds)
    return [ChildOrder(quantity=q, arrival_offset_seconds=t)
            for q, t in zip(quantities, times)]


__all__ = [
    "ChildOrder",
    "compute_limit_price",
    "slice_into_children",
    "child_arrival_schedule",
    "plan_execution",
]
