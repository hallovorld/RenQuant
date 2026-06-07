"""IGV short-plan state machine (pure core — no I/O, no broker, no clock).

Encodes the operator's discretionary IGV put-spread plan as a deterministic
state machine so it can be unit-tested exhaustively and driven by a cron
(scripts/igv_short_monitor.py). This module NEVER places an order, fetches
data, or reads the clock — it maps (state, market snapshot, config) to a new
state plus a list of *actions* the caller executes (alert, and — only when
explicitly armed — a live order).

Plan (operator-defined):

  WATCH (armed, no position)
    - touch $97.5-99 then reject (hourly close back < 97.5) -> ENTER 98/90 put spread
    - break < $94.8, then bounce $95-96 rejects (hourly close < 95) -> ENTER
    - reclaim $100 -> stand down (do NOT enter this bounce)
    - recover $101.5-102 -> VOID (plan dead)

  IN_POSITION
    - $92-93   -> take profit: close HALF        (once)
    - $88-90   -> take profit: close MOST -> CLOSED
    - >= $100.5 -> stop: cut HALF                 (once)
    - daily CLOSE >= $101.5 -> stop: EXIT ALL -> CLOSED

Reject rule (operator choice): zone *touched* in the recent window AND the
latest CLOSED hourly bar closes back below the zone low.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

State = Literal["WATCH", "IN_POSITION", "VOIDED", "CLOSED"]

# Action kinds the orchestrator knows how to execute.
ENTER = "ENTER"          # open the 98/90 put debit spread
CLOSE_HALF = "CLOSE_HALF"  # scale out 50% of current contracts
CLOSE_MOST = "CLOSE_MOST"  # close the majority (-> CLOSED)
CLOSE_ALL = "CLOSE_ALL"    # full exit (-> CLOSED)
VOID = "VOID"            # abandon the plan pre-entry (-> VOIDED)


@dataclass
class Bar:
    high: float
    low: float
    close: float


@dataclass
class Market:
    """Snapshot the orchestrator assembles before each step()."""
    price: float                 # latest trade/quote price
    hourly_bars: list[Bar]       # recent CLOSED hourly bars, oldest..newest
    daily_close: float | None = None  # most recent completed daily close (for close-based stop)


@dataclass
class PlanConfig:
    # entry zones
    reject_zone: tuple[float, float] = (97.5, 99.0)
    breakdown_level: float = 94.8
    failed_bounce_zone: tuple[float, float] = (95.0, 96.0)
    standby_level: float = 100.0        # reclaim -> do not enter
    void_level: float = 101.5           # recover -> plan dead
    # management
    tp_half_at: float = 93.0            # price <= -> close half
    tp_most_at: float = 90.0            # price <= -> close most
    sl_half_at: float = 100.5           # price >= -> cut half
    sl_exit_close_at: float = 101.5     # daily close >= -> exit all
    reject_lookback_bars: int = 6       # window to look for the zone "touch"


@dataclass
class PlanState:
    state: State = "WATCH"
    broke_below_breakdown: bool = False   # path B armed
    entry_path: str | None = None         # "A" | "B"
    contracts: int = 0                    # open contracts (post-fill, set by orchestrator)
    tp_half_done: bool = False
    sl_half_done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlanState":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


@dataclass
class Action:
    kind: str
    reason: str
    fraction: float | None = None   # for CLOSE_HALF etc., portion of current contracts


def _zone_touched(bars: list[Bar], lo: float, hi: float, lookback: int) -> bool:
    """True if any of the last `lookback` closed bars overlapped [lo, hi]."""
    for b in bars[-lookback:]:
        if b.high >= lo and b.low <= hi:
            return True
    return False


def _rejected(bars: list[Bar], lo: float, hi: float, lookback: int) -> bool:
    """Touched the zone in the window AND the latest closed bar closed back < lo."""
    if not bars:
        return False
    return _zone_touched(bars, lo, hi, lookback) and bars[-1].close < lo


def step(state: PlanState, mkt: Market, cfg: PlanConfig) -> tuple[PlanState, list[Action]]:
    """Advance the plan one tick. Pure: returns a NEW state + actions to execute."""
    s = PlanState.from_dict(state.to_dict())  # copy
    actions: list[Action] = []

    if s.state in ("VOIDED", "CLOSED"):
        return s, actions

    if s.state == "WATCH":
        # Void takes precedence — a recovery kills the whole short thesis.
        if mkt.price >= cfg.void_level:
            s.state = "VOIDED"
            actions.append(Action(VOID, f"price {mkt.price} >= void {cfg.void_level}"))
            return s, actions

        # Arm path B once price has broken below the breakdown level.
        lo_bars = min((b.low for b in mkt.hourly_bars), default=mkt.price)
        if min(mkt.price, lo_bars) < cfg.breakdown_level:
            s.broke_below_breakdown = True

        # Standby: if price has reclaimed $100, do NOT enter on this bounce.
        if mkt.price >= cfg.standby_level:
            return s, actions

        rlo, rhi = cfg.reject_zone
        blo, bhi = cfg.failed_bounce_zone
        if _rejected(mkt.hourly_bars, rlo, rhi, cfg.reject_lookback_bars):
            s.state = "IN_POSITION"
            s.entry_path = "A"
            actions.append(Action(ENTER, f"path A: rejected {rlo}-{rhi} (hourly close {mkt.hourly_bars[-1].close} < {rlo})"))
        elif s.broke_below_breakdown and _rejected(mkt.hourly_bars, blo, bhi, cfg.reject_lookback_bars):
            s.state = "IN_POSITION"
            s.entry_path = "B"
            actions.append(Action(ENTER, f"path B: post-breakdown failed bounce {blo}-{bhi} (hourly close {mkt.hourly_bars[-1].close} < {blo})"))
        return s, actions

    if s.state == "IN_POSITION":
        # Risk first: full close on a confirmed close above the exit level.
        if mkt.daily_close is not None and mkt.daily_close >= cfg.sl_exit_close_at:
            s.state = "CLOSED"
            actions.append(Action(CLOSE_ALL, f"stop: daily close {mkt.daily_close} >= {cfg.sl_exit_close_at}", 1.0))
            return s, actions
        # Stop: cut half once on reclaim of the half-stop level.
        if mkt.price >= cfg.sl_half_at and not s.sl_half_done:
            s.sl_half_done = True
            actions.append(Action(CLOSE_HALF, f"stop: price {mkt.price} >= {cfg.sl_half_at}", 0.5))
            return s, actions
        # Take profit: close most near the deep target.
        if mkt.price <= cfg.tp_most_at:
            s.state = "CLOSED"
            actions.append(Action(CLOSE_MOST, f"target: price {mkt.price} <= {cfg.tp_most_at}", 1.0))
            return s, actions
        # Take profit: close half once near the first target.
        if mkt.price <= cfg.tp_half_at and not s.tp_half_done:
            s.tp_half_done = True
            actions.append(Action(CLOSE_HALF, f"target: price {mkt.price} <= {cfg.tp_half_at}", 0.5))
            return s, actions
        return s, actions

    return s, actions
