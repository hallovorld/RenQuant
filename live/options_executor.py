"""Minimal Alpaca multi-leg options executor for the IGV put-spread plan.

`alpaca_broker.py` is equity-only; this adds the narrow options capability the
IGV plan needs: resolve the 98/90 put pair at the nearest weekly expiry, open a
defined-risk put DEBIT spread (BUY 98 put / SELL 90 put) as a single limit
multi-leg order, and scale/close it.

Safety invariants (enforced here, not just documented):
  * LIMIT orders only — never market (options spreads slip badly on market).
  * Hard contract ceiling (MAX_CONTRACTS) regardless of caller request.
  * Debit sanity bound 0 < debit < strike width.
  * Deterministic client_order_id (idempotent — a re-run never double-submits).
  * paper flag is explicit; the caller (orchestrator) decides paper vs live and
    is responsible for the arming gate.

This module does no scheduling, state, or alerting.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

log = logging.getLogger("live.options_executor")

MAX_CONTRACTS = int(os.environ.get("IGV_MAX_CONTRACTS", "5"))  # absolute ceiling


@dataclass
class SpreadLegs:
    long_put_occ: str   # bought (98)
    short_put_occ: str  # sold (90)
    expiry: date
    long_strike: float
    short_strike: float


def _client(paper: bool):
    from alpaca.trading.client import TradingClient  # noqa: PLC0415
    key = os.environ.get("ALPACA_API_KEY", "")
    sec = os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not sec:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
    return TradingClient(api_key=key, secret_key=sec, paper=paper)


def resolve_put_spread(
    underlying: str,
    long_strike: float,
    short_strike: float,
    *,
    dte_min: int = 0,
    dte_max: int = 7,
    paper: bool = True,
    client=None,
) -> SpreadLegs:
    """Find the nearest expiry in [dte_min, dte_max] where BOTH puts are listed.

    Returns the resolved OCC symbols straight from Alpaca (no hand-built OCC),
    so strike/expiry formatting can't drift. Raises if no expiry has both.
    """
    from alpaca.trading.requests import GetOptionContractsRequest  # noqa: PLC0415
    from alpaca.trading.enums import ContractType  # noqa: PLC0415

    client = client or _client(paper)
    today = datetime.now(timezone.utc).date()
    req = GetOptionContractsRequest(
        underlying_symbols=[underlying],
        type=ContractType.PUT,
        expiration_date_gte=today + timedelta(days=dte_min),
        expiration_date_lte=today + timedelta(days=dte_max),
        strike_price_gte=str(min(long_strike, short_strike)),
        strike_price_lte=str(max(long_strike, short_strike)),
        limit=10000,
    )
    contracts = list(getattr(client.get_option_contracts(req), "option_contracts", []) or [])
    by_exp: dict[date, dict[float, str]] = {}
    for c in contracts:
        exp = c.expiration_date if isinstance(c.expiration_date, date) else date.fromisoformat(str(c.expiration_date))
        by_exp.setdefault(exp, {})[float(c.strike_price)] = c.symbol
    for exp in sorted(by_exp):
        strikes = by_exp[exp]
        if long_strike in strikes and short_strike in strikes:
            return SpreadLegs(
                long_put_occ=strikes[long_strike],
                short_put_occ=strikes[short_strike],
                expiry=exp,
                long_strike=long_strike,
                short_strike=short_strike,
            )
    raise RuntimeError(
        f"no expiry in {dte_min}-{dte_max} DTE lists both {long_strike}/{short_strike} puts on {underlying}"
    )


def _check_caps(contracts: int, limit_price: float, width: float) -> None:
    if contracts < 1:
        raise ValueError(f"contracts must be >= 1, got {contracts}")
    if contracts > MAX_CONTRACTS:
        raise ValueError(f"contracts {contracts} exceeds hard cap MAX_CONTRACTS={MAX_CONTRACTS}")
    if not (0 < limit_price < width):
        raise ValueError(f"debit {limit_price} outside sane bound (0, width={width})")


def _submit_mleg(client, *, long_occ, short_occ, contracts, limit_price,
                 opening: bool, client_order_id: str):
    """Build + submit the two-leg limit order. opening=True: BUY long / SELL short."""
    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest  # noqa: PLC0415
    from alpaca.trading.enums import OrderSide, OrderClass, TimeInForce, PositionIntent  # noqa: PLC0415

    if opening:
        long_side, short_side = OrderSide.BUY, OrderSide.SELL
        long_intent, short_intent = PositionIntent.BUY_TO_OPEN, PositionIntent.SELL_TO_OPEN
    else:  # closing: reverse both legs
        long_side, short_side = OrderSide.SELL, OrderSide.BUY
        long_intent, short_intent = PositionIntent.SELL_TO_CLOSE, PositionIntent.BUY_TO_CLOSE

    legs = [
        OptionLegRequest(symbol=long_occ, side=long_side, ratio_qty=1, position_intent=long_intent),
        OptionLegRequest(symbol=short_occ, side=short_side, ratio_qty=1, position_intent=short_intent),
    ]
    order = LimitOrderRequest(
        qty=contracts,
        limit_price=round(limit_price, 2),
        order_class=OrderClass.MLEG,
        time_in_force=TimeInForce.DAY,
        legs=legs,
        client_order_id=client_order_id,
    )
    return client.submit_order(order)


def open_put_spread(legs: SpreadLegs, contracts: int, limit_debit: float, *,
                    plan_id: str, paper: bool = True, client=None):
    """Open the 98/90 put debit spread. Idempotent via client_order_id."""
    width = abs(legs.long_strike - legs.short_strike)
    _check_caps(contracts, limit_debit, width)
    client = client or _client(paper)
    coid = f"igv-open-{plan_id}-{legs.expiry.isoformat()}"
    log.warning("OPEN %s put spread %s/%s exp=%s qty=%d debit<=%.2f paper=%s",
                "IGV", legs.long_strike, legs.short_strike, legs.expiry, contracts, limit_debit, paper)
    return _submit_mleg(client, long_occ=legs.long_put_occ, short_occ=legs.short_put_occ,
                        contracts=contracts, limit_price=limit_debit, opening=True,
                        client_order_id=coid)


def close_put_spread(legs: SpreadLegs, contracts: int, limit_credit: float, *,
                     plan_id: str, tag: str, paper: bool = True, client=None):
    """Close `contracts` of the spread (reverse legs). `tag` keeps the
    client_order_id unique per scale-out (e.g. 'tp_half', 'sl_half', 'exit')."""
    width = abs(legs.long_strike - legs.short_strike)
    _check_caps(contracts, limit_credit, width)
    client = client or _client(paper)
    coid = f"igv-close-{tag}-{plan_id}-{legs.expiry.isoformat()}"
    log.warning("CLOSE(%s) %d contracts credit>=%.2f paper=%s", tag, contracts, limit_credit, paper)
    return _submit_mleg(client, long_occ=legs.long_put_occ, short_occ=legs.short_put_occ,
                        contracts=contracts, limit_price=limit_credit, opening=False,
                        client_order_id=coid)
