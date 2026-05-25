"""Portfolio-level helpers shared by LEAN, notebook simulation, and live runner.

Pure functions — stdlib only.  No common/ imports.
"""
from __future__ import annotations

import datetime as _dt
import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def update_drawdown_circuit_breaker(
    portfolio_value: float,
    high_water_mark: float,
    halt_threshold: float,
) -> tuple[float, bool]:
    """Update HWM and determine whether the drawdown circuit breaker fires.

    Returns:
        (new_hwm, should_halt_buys)

    Audit fix PORT-1/PORT-2 (Round 2 deep audit, 2026-04-25): pre-fix,
    `max(NaN_hwm, finite_pv)` returned NaN (CPython max-NaN semantics),
    then the `<=0` guard let NaN slip past, then drawdown = NaN/NaN
    = NaN, then `NaN >= threshold` False → returned (NaN, False)
    silently. Caller persisted NaN HWM, propagating corruption across
    bars. Fail-SAFE: non-finite inputs route to a clean fallback —
    HWM stays at the LAST finite value (or 0), no halt fires.
    """
    if not math.isfinite(portfolio_value):
        # Bad portfolio_value (broker outage, NaN equity) — preserve
        # stored HWM if finite; don't ratchet up; don't halt.
        if math.isfinite(high_water_mark):
            return float(high_water_mark), False
        return 0.0, False
    if not math.isfinite(high_water_mark):
        # Stored HWM corrupted but pv is good — reset HWM to pv so
        # future drawdown calc is meaningful.
        return float(portfolio_value), False
    new_hwm = max(high_water_mark, portfolio_value)
    if halt_threshold <= 0 or new_hwm <= 0:
        return new_hwm, False
    drawdown = (new_hwm - portfolio_value) / new_hwm
    return new_hwm, drawdown >= halt_threshold


def compute_trade_tax(
    gross_pnl: float,
    hold_days: int,
    short_term_rate: float,
    long_term_rate: float,
    long_term_threshold_days: int = 365,
) -> float:
    """Return income tax owed on a realized trade.

    Only positive P&L is taxed.  Long-term rate applies when
    hold_days >= long_term_threshold_days.

    Audit fix PORT-3 (Round 2 deep audit, 2026-04-25): pre-fix, NaN
    gross_pnl slipped past `<= 0` (NaN<=0 False), then `gross_pnl *
    rate = NaN` propagated into tax_drag and rotation cost calcs.
    Now: explicit isfinite guard returns 0 on non-finite (no tax owed
    when we can't compute the gain — fail-safe).
    """
    if not math.isfinite(gross_pnl) or gross_pnl <= 0:
        return 0.0
    rate = long_term_rate if hold_days >= long_term_threshold_days else short_term_rate
    return gross_pnl * rate


def compute_disposed_lot_tax(
    sell_price: float,
    sell_date: _dt.date,
    disposed_lots: Iterable[Any],
    short_term_rate: float,
    long_term_rate: float,
    long_term_threshold_days: int = 365,
) -> dict[str, float]:
    """Tax a sell event using the acquisition date of each disposed lot.

    FIFO/HIFO changes which lot is sold, so the tax age must be computed from
    the same disposed slices used for cost basis. Returning split ST/LT gains
    also lets annual-net reporting stay closer to the actual lot accounting.
    """
    if not math.isfinite(float(sell_price)) or sell_date is None:
        return {
            "tax": 0.0,
            "weighted_hold_days": 0.0,
            "short_term_gross_pnl": 0.0,
            "long_term_gross_pnl": 0.0,
        }
    tax = 0.0
    st_gross = 0.0
    lt_gross = 0.0
    total_shares = 0.0
    weighted_days = 0.0
    for lot in disposed_lots:
        shares = _coerce_finite_float(getattr(lot, "shares", None), default=0.0) or 0.0
        price = _coerce_finite_float(getattr(lot, "price", None), default=0.0) or 0.0
        date = getattr(lot, "date", None)
        if shares <= 0 or price <= 0 or date is None:
            continue
        if isinstance(date, _dt.datetime):
            date = date.date()
        if not isinstance(date, _dt.date):
            continue
        hold_days = max(0, (sell_date - date).days)
        gain = shares * (float(sell_price) - price)
        total_shares += shares
        weighted_days += shares * hold_days
        if hold_days >= int(long_term_threshold_days):
            lt_gross += gain
            rate = long_term_rate
        else:
            st_gross += gain
            rate = short_term_rate
        if math.isfinite(gain) and gain > 0:
            tax += gain * rate
    return {
        "tax": float(tax),
        "weighted_hold_days": (
            float(weighted_days / total_shares) if total_shares > 0 else 0.0
        ),
        "short_term_gross_pnl": float(st_gross),
        "long_term_gross_pnl": float(lt_gross),
    }


def compute_netted_capital_gains_tax(
    short_term_net: float,
    long_term_net: float,
    short_term_rate: float,
    long_term_rate: float,
) -> float:
    """Tax positive capital gains after same-bucket and cross netting.

    This is a reporting helper, not a filing engine. It mirrors the economic
    shape of Schedule D: short-term gains/losses are netted, long-term
    gains/losses are netted, then opposite-sign buckets offset each other.
    Loss carryforwards, wash-sale basis deferrals, NIIT, brackets, and state
    taxes remain outside this simplified simulator scope.
    """
    if not (
        math.isfinite(short_term_net)
        and math.isfinite(long_term_net)
        and math.isfinite(short_term_rate)
        and math.isfinite(long_term_rate)
    ):
        return 0.0
    if short_term_net >= 0 and long_term_net >= 0:
        return short_term_net * short_term_rate + long_term_net * long_term_rate
    if short_term_net <= 0 and long_term_net <= 0:
        return 0.0
    if short_term_net > 0 and long_term_net < 0:
        return max(0.0, short_term_net + long_term_net) * short_term_rate
    if long_term_net > 0 and short_term_net < 0:
        return max(0.0, long_term_net + short_term_net) * long_term_rate
    return 0.0


def compute_annual_net_capital_gains_tax(
    realized_events: Iterable[dict[str, Any]],
    short_term_rate: float,
    long_term_rate: float,
    long_term_threshold_days: int = 365,
    *,
    date_key: str = "date",
    pnl_key: str = "gross_pnl",
    hold_days_key: str = "hold_days",
) -> dict[str, Any]:
    """Estimate annual tax after calendar-year short/long netting.

    ``compute_trade_tax`` is deliberately event-level and conservative for
    cash-stress simulations. This helper gives reports the complementary
    annual-net estimate so losing trades offset winning trades within the
    same calendar year instead of making tax drag look mechanically larger
    than the strategy's net realized gain.
    """
    buckets: dict[int, dict[str, float]] = defaultdict(
        lambda: {"short_term_net": 0.0, "long_term_net": 0.0},
    )
    for event in realized_events:
        year = _coerce_year(event.get(date_key) or event.get("exit_date"))
        if year is None:
            continue
        decision_inputs = event.get("decision_inputs")
        if not isinstance(decision_inputs, dict):
            decision_inputs = {}
        has_lot_split = (
            "short_term_gross_pnl" in event
            or "long_term_gross_pnl" in event
            or "short_term_gross_pnl" in decision_inputs
            or "long_term_gross_pnl" in decision_inputs
        )
        if has_lot_split:
            buckets[year]["short_term_net"] += _coerce_finite_float(
                event.get("short_term_gross_pnl")
                if "short_term_gross_pnl" in event
                else decision_inputs.get("short_term_gross_pnl"),
                default=0.0,
            ) or 0.0
            buckets[year]["long_term_net"] += _coerce_finite_float(
                event.get("long_term_gross_pnl")
                if "long_term_gross_pnl" in event
                else decision_inputs.get("long_term_gross_pnl"),
                default=0.0,
            ) or 0.0
            continue
        gross_pnl = _coerce_finite_float(event.get(pnl_key))
        if gross_pnl is None:
            continue
        hold_days = _coerce_finite_float(event.get(hold_days_key), default=0.0)
        if hold_days is None:
            hold_days = 0.0
        bucket = (
            "long_term_net"
            if int(hold_days) >= int(long_term_threshold_days)
            else "short_term_net"
        )
        buckets[year][bucket] += gross_pnl

    rows: list[dict[str, float | int]] = []
    total_tax = 0.0
    for year in sorted(buckets):
        st_net = buckets[year]["short_term_net"]
        lt_net = buckets[year]["long_term_net"]
        tax = compute_netted_capital_gains_tax(
            st_net, lt_net, short_term_rate, long_term_rate,
        )
        total_tax += tax
        rows.append({
            "year": int(year),
            "short_term_net": float(st_net),
            "long_term_net": float(lt_net),
            "estimated_tax": float(tax),
        })
    return {"total_estimated_tax": float(total_tax), "years": rows}


def _coerce_finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _coerce_year(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, _dt.datetime | _dt.date):
        return int(value.year)
    year = getattr(value, "year", None)
    if isinstance(year, int):
        return year
    text = str(value)
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return None
