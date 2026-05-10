"""Broker commission + regulatory fee computation.

Single source of truth per CLAUDE.md §5.13.5 — every callsite (sim,
LEAN-side cost accounting, runner reconciliation) MUST route through
:func:`compute_buy_fees` / :func:`compute_sell_fees`.

Schedule defaults match Alpaca's published commission schedule
(https://alpaca.markets/learn/commissions) as of Q4 2025:

* SEC Section 31 fee — $27.00 per $1M of principal on **sells only**
  (i.e. ``27.0e-6`` of notional). Quoted as 0.0000278 in some sources
  and revised periodically; the FY2025 rate is $27.00/$1M.
* FINRA TAF (Trading Activity Fee) — $0.000119 per share **sold**,
  capped at $5.95 per execution (cap not modeled here; per-trade
  totals stay well below). Buys are exempt.
* Custom broker commission — zero by default for Alpaca / IBKR's
  zero-commission plans. Set ``custom_bps > 0`` for legacy brokers
  or IBKR Pro Tiered.

Per CLAUDE.md §5.13.11: every arithmetic path here is finite-guarded
so a NaN price or share count cannot poison cash bookkeeping
downstream.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FeeConfig:
    """Immutable fee schedule.

    Attributes match Alpaca Q4 2025 retail equities defaults.
    """

    sec_fee_rate: float = 27.0e-6      # $0.0000027 per dollar on sells (SEC §31)
    taf_per_share: float = 1.19e-4     # $0.000119 per share sold (FINRA TAF)
    custom_bps: float = 0.0            # configurable broker bps (default 0 = Alpaca)


# ── Helpers (each ≤ 50 lines per CLAUDE.md §1c) ───────────────────────────


def _safe_notional(shares: float, price: float) -> float:
    """Return ``shares * price`` when both are finite-positive, else 0.

    Guards against NaN/inf leaking from upstream price feeds (per §5.13.11):
    ``NaN > 0`` evaluates False, so without the explicit isfinite check a
    naked ``shares * price`` would propagate NaN through every downstream
    fee dict.
    """
    if not (math.isfinite(shares) and math.isfinite(price)):
        return 0.0
    if shares <= 0 or price <= 0:
        return 0.0
    return shares * price


def _sec_fee_component(notional: float, cfg: FeeConfig) -> float:
    """SEC Section 31 fee on sell notional (regulatory; floor 0)."""
    if notional <= 0 or not math.isfinite(cfg.sec_fee_rate):
        return 0.0
    fee = notional * max(0.0, cfg.sec_fee_rate)
    return fee if math.isfinite(fee) else 0.0


def _taf_component(shares: float, cfg: FeeConfig) -> float:
    """FINRA TAF on shares sold (regulatory; floor 0)."""
    if not math.isfinite(shares) or shares <= 0:
        return 0.0
    if not math.isfinite(cfg.taf_per_share):
        return 0.0
    fee = shares * max(0.0, cfg.taf_per_share)
    return fee if math.isfinite(fee) else 0.0


def _custom_bps_component(notional: float, cfg: FeeConfig) -> float:
    """Broker bps commission (0 for Alpaca / IBKR-Lite by default)."""
    if notional <= 0 or not math.isfinite(cfg.custom_bps):
        return 0.0
    fee = notional * max(0.0, cfg.custom_bps) * 1.0e-4
    return fee if math.isfinite(fee) else 0.0


# ── Public API ────────────────────────────────────────────────────────────


def compute_sell_fees(shares: float, price: float, cfg: FeeConfig) -> dict:
    """Sell-side fee breakdown.

    Returns a dict with keys ``sec_fee``, ``taf``, ``custom``, ``total``.
    All values are non-negative floats. NaN / inf inputs return zeros
    across the board (defensive — preserves cash invariants).

    Per §5.13.11: if EITHER shares or price is non-finite/non-positive,
    we treat the fill as invalid (no execution → no fees). Without this
    the per-share TAF would survive a NaN price even though no actual
    fill happened, double-counting cash drag on the rejected order.
    """
    if not (math.isfinite(shares) and math.isfinite(price)
            and shares > 0 and price > 0):
        return {"sec_fee": 0.0, "taf": 0.0, "custom": 0.0, "total": 0.0}
    notional = _safe_notional(shares, price)
    sec = _sec_fee_component(notional, cfg)
    taf = _taf_component(shares, cfg)
    custom = _custom_bps_component(notional, cfg)
    total = sec + taf + custom
    return {
        "sec_fee": sec,
        "taf": taf,
        "custom": custom,
        "total": total if math.isfinite(total) else 0.0,
    }


def compute_buy_fees(shares: float, price: float, cfg: FeeConfig) -> dict:
    """Buy-side fee breakdown.

    Returns dict with keys ``sec_fee``, ``taf``, ``custom``, ``total``.
    SEC §31 and FINRA TAF are sell-side regulatory fees and are always
    zero on buys (cf. Alpaca and IBKR fee schedules). Only the optional
    ``custom_bps`` broker commission applies.
    """
    notional = _safe_notional(shares, price)
    custom = _custom_bps_component(notional, cfg)
    return {
        "sec_fee": 0.0,
        "taf": 0.0,
        "custom": custom,
        "total": custom if math.isfinite(custom) else 0.0,
    }
