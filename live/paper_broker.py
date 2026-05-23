"""Paper (simulated) broker for testing the live runner without IBKR."""

from __future__ import annotations

import logging

from .broker import BaseBroker

log = logging.getLogger(__name__)


class PaperBroker(BaseBroker):
    broker_name = "paper"  # state-file isolation tag (see kernel.state_paths)

    """Simulates order fills locally.  No real money is involved.

    Round-2 audit (#R2-7..10): now tracks cash, average cost basis, and
    last-fill price for sensible mark-to-market. Pre-fix, place_order
    was a pure position-counter that ignored cash + price entirely, so
    the strategy in `--broker paper` mode saw infinite cash regardless
    of trades.
    """

    def __init__(self, initial_cash: float = 100_000):
        self._cash:    float = float(initial_cash)
        self._initial_cash: float = float(initial_cash)
        self._positions:  dict[str, float] = {}
        self._avg_cost:   dict[str, float] = {}
        self._last_price: dict[str, float] = {}
        self._order_counter = 0
        self._connected = False
        # Z9: broker-side stop simulation. {order_id: {symbol, qty, stop_price}}.
        # Triggers happen via _check_stops() which the runner / tests can
        # call after set_price to simulate broker-side fills.
        self._stop_orders: dict[str, dict] = {}

    def connect(self) -> None:
        self._connected = True
        log.info("PaperBroker connected (cash=$%.2f)", self._cash)

    def disconnect(self) -> None:
        self._connected = False
        log.info("PaperBroker disconnected")

    def get_position(self, symbol: str) -> float:
        return self._positions.get(symbol, 0.0)

    def get_account_value(self) -> float:
        # Mark-to-market: cash + Σ qty × last_price
        total = self._cash
        for sym, qty in self._positions.items():
            if qty <= 0:
                continue
            total += qty * self._last_price.get(sym, self._avg_cost.get(sym, 0.0))
        return total

    def get_cash(self) -> float:
        return self._cash

    def get_avg_cost(self, symbol: str) -> float:
        return self._avg_cost.get(symbol, 0.0)

    def get_all_positions(self) -> list[dict]:
        rows: list[dict] = []
        for sym, qty in self._positions.items():
            if qty <= 0:
                continue
            cost  = self._avg_cost.get(sym, 0.0)
            price = self._last_price.get(sym, cost)
            rows.append({
                "symbol":          sym,
                "qty":             qty,
                "avg_entry_price": cost,
                "market_value":    qty * price,
                "unrealized_pl":   qty * (price - cost),
            })
        return rows

    def set_price(self, symbol: str, price: float) -> None:
        """Test/runner hook: stamp the last-known price for mark-to-market.

        The runner doesn't currently feed prices in (it pulls from parquet),
        so callers can call this between place_order calls to keep
        get_account_value in sync with reality during paper-mode dry-runs.
        """
        if price > 0:
            self._last_price[symbol] = float(price)

    def place_order(
        self, symbol: str, action: str, quantity: float, price: "float | None" = None,
    ) -> dict:
        # 2026-05-09 audit fix (PB-NaN-1): pre-fix, NaN/inf quantity slipped
        # past `qty <= 0` (NaN comparisons return False) → invest=NaN →
        # cash=NaN → all subsequent get_account_value calls return NaN.
        # Same NaN-propagation pattern as SAB-3 / DC-1 fixed in adapters.
        # Now: explicit isfinite guard rejects bad orders cleanly.
        import math as _math_pb  # noqa: PLC0415
        if not _math_pb.isfinite(quantity) or quantity <= 0:
            log.warning(
                "PaperBroker.place_order(%s, %s, %s): rejecting non-finite "
                "or non-positive quantity — order NOT placed",
                action, symbol, quantity,
            )
            self._order_counter += 1
            oid = f"PAPER-REJECTED-{self._order_counter:04d}"
            return {"order_id": oid, "status": "rejected",
                    "action": action.upper(), "symbol": symbol,
                    "quantity": 0, "price": None,
                    "reject_reason": f"non-finite or non-positive quantity ({quantity})"}
        self._order_counter += 1
        oid = f"PAPER-{self._order_counter:04d}"
        action_u = action.upper()
        # Use supplied price; fallback to last known; final fallback to avg
        # cost (for closing trades). If we have no price reference, the
        # cash impact is undefined — but that's a configuration issue we
        # surface rather than silently mis-compute.
        if price is None:
            price = self._last_price.get(symbol)
        if price is None:
            log.warning(
                "PaperBroker.place_order(%s, %s, %s): no price — cash NOT updated",
                action, symbol, quantity,
            )
            invest = 0.0
        else:
            price = float(price)
            # 2026-05-09 audit fix (PB-NaN-2): same isfinite guard on price.
            # Pre-fix, fetch_intraday_bars rare race could return NaN price
            # → invest=NaN → cash=NaN → state corrupted.
            if not _math_pb.isfinite(price) or price <= 0:
                log.warning(
                    "PaperBroker.place_order(%s, %s, %s): non-finite price "
                    "(%s) — treating as no price; cash NOT updated",
                    action, symbol, quantity, price,
                )
                price = None
                invest = 0.0
            else:
                invest = float(quantity) * price
                self._last_price[symbol] = price

        if action_u == "BUY":
            if invest > self._cash + 1e-6 and price is not None:
                log.warning(
                    "PaperBroker: insufficient cash for %s (need $%.2f, have $%.2f) — "
                    "order rejected",
                    symbol, invest, self._cash,
                )
                return {"order_id": oid, "status": "rejected",
                        "action": action_u, "symbol": symbol,
                        "quantity": 0, "price": price,
                        "reject_reason": "insufficient cash"}
            old_qty   = self._positions.get(symbol, 0.0)
            old_cost  = self._avg_cost.get(symbol, 0.0)
            new_qty   = old_qty + quantity
            if new_qty > 0 and price is not None:
                self._avg_cost[symbol] = (
                    old_cost * old_qty + price * quantity
                ) / new_qty
            self._positions[symbol] = new_qty
            self._cash -= invest
        elif action_u == "SELL":
            held = self._positions.get(symbol, 0.0)
            if quantity > held + 1e-9:
                log.warning(
                    "PaperBroker: SELL %s qty=%s exceeds held=%s — clipping",
                    symbol, quantity, held,
                )
                quantity = held
                if price is not None:
                    invest = quantity * price
            new_qty = held - quantity
            self._positions[symbol] = max(0.0, new_qty)
            if self._positions[symbol] == 0:
                self._avg_cost.pop(symbol, None)
            self._cash += invest

        log.info("Order %s: %s %s %.0f shares @ $%s",
                 oid, action_u, symbol, quantity,
                 f"{price:.2f}" if price is not None else "?")
        return {"order_id": oid, "status": "filled", "action": action_u,
                "symbol": symbol, "quantity": quantity, "price": price}

    # ── Broker-side stop simulation (Z9) ─────────────────────────────────

    def supports_broker_side_stops(self) -> bool:
        return True

    def place_stop_order(
        self, symbol: str, quantity: float, stop_price: float,
    ) -> dict:
        if quantity <= 0:
            raise ValueError(f"place_stop_order: quantity must be positive (got {quantity})")
        if stop_price <= 0:
            raise ValueError(f"place_stop_order: stop_price must be positive (got {stop_price})")
        held = self._positions.get(symbol, 0.0)
        if quantity > held + 1e-9:
            raise ValueError(
                f"place_stop_order: qty={quantity} exceeds held={held} for {symbol}"
            )
        self._order_counter += 1
        oid = f"PAPER-STP-{self._order_counter:04d}"
        self._stop_orders[oid] = {
            "symbol":     symbol,
            "quantity":   float(quantity),
            "stop_price": float(stop_price),
        }
        log.info("Stop order %s: SELL %s %.0f @ stop=$%.2f (queued)",
                 oid, symbol, quantity, stop_price)
        return {"order_id": oid, "status": "accepted", "symbol": symbol,
                "quantity": float(quantity), "stop_price": float(stop_price)}

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._stop_orders:
            self._stop_orders.pop(order_id, None)
            log.info("Cancelled stop order %s", order_id)
            return True
        log.warning("cancel_order(%s): unknown order id", order_id)
        return False

    def _check_stops(self) -> list[dict]:
        """Trigger any stop whose stop_price is at-or-above the symbol's
        current last_price. Mimics broker-side fills.

        Returns a list of dicts describing executed fills. Tests + the
        runner can call this after set_price() to simulate the broker
        firing the stop in real time. Each fill reduces the position and
        increases cash like a SELL market order.
        """
        triggered: list[dict] = []
        for oid in list(self._stop_orders.keys()):
            spec = self._stop_orders[oid]
            sym = spec["symbol"]
            stop_p = spec["stop_price"]
            qty = spec["quantity"]
            last = self._last_price.get(sym)
            if last is None:
                continue
            # Sell-stop fires when last_price <= stop_price
            if last <= stop_p:
                # Execute as market sell at last_price (a real broker
                # might fill below the stop on a gap; we keep it simple)
                fill_price = float(last)
                held = self._positions.get(sym, 0.0)
                exec_qty = min(qty, held)
                if exec_qty <= 0:
                    self._stop_orders.pop(oid, None)
                    continue
                self._positions[sym] = held - exec_qty
                if self._positions[sym] == 0:
                    self._avg_cost.pop(sym, None)
                self._cash += exec_qty * fill_price
                self._stop_orders.pop(oid, None)
                triggered.append({
                    "order_id":   oid,
                    "symbol":     sym,
                    "quantity":   exec_qty,
                    "fill_price": fill_price,
                    "stop_price": stop_p,
                })
                log.info("Stop %s TRIGGERED: SOLD %s %.0f @ %.2f (stop=%.2f)",
                         oid, sym, exec_qty, fill_price, stop_p)
        return triggered
