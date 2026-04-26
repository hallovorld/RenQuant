"""Alpaca broker for live and paper trading.

Requires ``alpaca-py`` (``pip install alpaca-py``) and API credentials
set via environment variables or passed at construction:

    ALPACA_API_KEY
    ALPACA_SECRET_KEY

By default connects to the paper trading endpoint.  Pass ``paper=False``
for live trading.
"""

from __future__ import annotations

import logging
import os

from .broker import BaseBroker

log = logging.getLogger(__name__)


class AlpacaBroker(BaseBroker):
    """Execute orders via Alpaca Markets API."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
    ):
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self._paper = paper
        self._trading_client = None
        self._order_counter = 0

    def connect(self) -> None:
        try:
            from alpaca.trading.client import TradingClient
        except ImportError:
            raise ImportError(
                "alpaca-py is required for Alpaca broker. "
                "Install it with: pip install alpaca-py"
            )

        if not self._api_key or not self._secret_key:
            raise ValueError(
                "Alpaca API credentials not found. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables, or pass them to "
                "AlpacaBroker(api_key=..., secret_key=...)."
            )

        self._trading_client = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=self._paper,
        )

        account = self._trading_client.get_account()
        mode = "paper" if self._paper else "LIVE"
        log.info(
            "Alpaca connected (%s) — equity=$%s, cash=$%s, status=%s",
            mode, account.equity, account.cash, account.status,
        )

        if account.status != "ACTIVE":
            log.warning("Account status is %s — trading may be restricted", account.status)

    def disconnect(self) -> None:
        self._trading_client = None
        log.info("Alpaca disconnected")

    def get_position(self, symbol: str) -> float:
        from alpaca.common.exceptions import APIError

        try:
            position = self._trading_client.get_open_position(symbol)
            return float(position.qty)
        except APIError:
            return 0.0

    def get_account_value(self) -> float:
        account = self._trading_client.get_account()
        return float(account.equity)

    def get_cash(self) -> float:
        """Return available cash (non-margin)."""
        account = self._trading_client.get_account()
        return float(account.cash)

    def place_order(self, symbol: str, action: str, quantity: float) -> dict:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        # Audit fix ALPACA-ACCT-STATUS (Round 2 deep audit, 2026-04-25):
        # pre-fix, account status was checked at connect() only — and even
        # then logged-not-blocked. Alpaca can disable an account mid-day
        # for PDT violations, settlement issues, margin calls, regulatory
        # holds, etc. The live runner would keep submitting orders that
        # all fail at the API layer, with no clear "account is restricted"
        # signal until the operator looked at logs. Worse: paper accounts
        # rarely test this path. Now: re-check at every place_order; if
        # status is not ACTIVE, raise so the adapter's existing
        # try/except records it as a broker failure (EXITS-FAIL on the
        # sell side; orders_skipped on the buy side).
        try:
            account = self._trading_client.get_account()
            status = str(getattr(account, "status", ""))
        except Exception as exc:
            raise RuntimeError(
                f"alpaca pre-trade account check failed: {exc}"
            ) from exc
        if status not in ("ACTIVE", "AccountStatus.ACTIVE"):
            raise RuntimeError(
                f"alpaca account status is '{status}' (not ACTIVE) — refusing to place "
                f"{action} {symbol} x{quantity}. Operator action required."
            )

        side = OrderSide.BUY if action.upper() == "BUY" else OrderSide.SELL

        request = MarketOrderRequest(
            symbol=symbol,
            qty=int(quantity),
            side=side,
            time_in_force=TimeInForce.DAY,
        )

        order = self._trading_client.submit_order(request)
        self._order_counter += 1

        log.info(
            "Order %s: %s %s %d shares — status=%s",
            order.id, action, symbol, int(quantity), order.status,
        )

        return {
            "order_id": str(order.id),
            "status": str(order.status),
            "action": action,
            "symbol": symbol,
            "quantity": int(quantity),
        }

    def get_avg_cost(self, symbol: str) -> float:
        from alpaca.common.exceptions import APIError
        try:
            position = self._trading_client.get_open_position(symbol)
            return float(position.avg_entry_price)
        except APIError:
            return 0.0

    def get_all_positions(self) -> list[dict]:
        """Return all open positions as a list of dicts.

        Audit fix BROKER-AVAILABLE-QTY (2026-04-26 round-3): include
        `qty_available` so SellTask can avoid asking for shares locked
        in pending orders. Pre-fix, e2e round 3 saw PLTR sell rejected
        with `available=0, existing_qty=5, held_for_orders=5`. Now we
        expose qty_available; SellTask uses min(req, qty_available).
        Falls back to qty when Alpaca SDK doesn't return qty_available
        (older versions).
        """
        positions = self._trading_client.get_all_positions()
        out = []
        for p in positions:
            qty = float(p.qty)
            # qty_available = qty - held_for_orders (str on alpaca-py).
            # Audit fix PLTR-AVAILABLE-QTY (2026-04-26 round-4):
            # `getattr(...) or qty` collapses 0-available BACK to qty
            # because 0.0 is falsy. Pre-fix bug masked the "all locked"
            # case. Now: explicit None check.
            _qa_raw = getattr(p, "qty_available", None)
            qty_avail = qty if _qa_raw is None else float(_qa_raw)
            out.append({
                "symbol": p.symbol,
                "qty": qty,
                "qty_available": qty_avail,
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "avg_entry_price": float(p.avg_entry_price),
            })
        return out

    def get_filled_orders(self, after: str | None = None) -> list[dict]:
        """Return filled orders, optionally filtered to those after a date string 'YYYY-MM-DD'.

        Each entry: {"symbol", "action" (BUY/SELL), "qty", "filled_at" (ISO string), "avg_price"}

        Audit fix DBT-1 (2026-04-25 followups): pre-fix, single page of
        100 orders. Long-tenure positions held across many trades had
        their original BUY paginated past the 100-order cap → ENTRY-DATE-
        FROM-FILLS would silently fall back to the 31-day sentinel,
        artificially compressing tenure. Now: paginate via `until`
        cursor walking backward in time until a page returns < page_size
        orders OR until we exceed `max_pages` (safety cap to prevent
        unbounded iteration on accounts with thousands of orders).
        """
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        from datetime import datetime, timezone

        page_size = 500
        max_pages = 10  # 5000-order cap — covers ≥1y of weekly trading
        all_orders: list = []
        until_cursor: "datetime | None" = None
        after_dt: "datetime | None" = (
            datetime.fromisoformat(after).replace(tzinfo=timezone.utc)
            if after else None
        )
        for _page in range(max_pages):
            params = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED, limit=page_size, direction="desc",
            )
            if after_dt is not None:
                params.after = after_dt
            if until_cursor is not None:
                params.until = until_cursor
            page = self._trading_client.get_orders(filter=params)
            if not page:
                break
            all_orders.extend(page)
            if len(page) < page_size:
                break
            # Cursor for next page: walk backward in time. Use the OLDEST
            # order's submitted_at minus 1µs to avoid re-fetching it.
            try:
                oldest = min(
                    (o.submitted_at for o in page if o.submitted_at is not None),
                    default=None,
                )
            except Exception:
                oldest = None
            if oldest is None:
                break
            until_cursor = oldest
        orders = all_orders
        result = []
        # Audit fix ALPACA-STATUS (Round 2 deep audit, 2026-04-25):
        # pre-fix, brittle string comparison `str(o.status) not in
        # ("OrderStatus.FILLED", "filled")` only matched two specific
        # representations. Across alpaca-py versions:
        #   * <= 0.27 returned "filled"
        #   * >= 0.28 returned "OrderStatus.FILLED"
        #   * 0.30+ may add PARTIALLY_FILLED reps; PartiallyFilled etc.
        # Worse, the filter excluded PARTIALLY_FILLED — partial fills
        # were silently dropped from the "filled orders" list, so the
        # pipeline thought no buy/sell happened when in fact some shares
        # did execute. Reconciliation between live_state.json and the
        # actual broker position would then drift over time.
        # Now: case-insensitive substring match on `filled` covers
        # both FILLED and PARTIALLY_FILLED across versions, and same
        # for side comparison (BUY/SELL).
        def _is_filled(s) -> bool:
            return "filled" in str(s).lower()
        def _is_buy(s) -> bool:
            return "buy" in str(s).lower()
        for o in orders:
            if not _is_filled(o.status):
                continue
            filled_at = o.filled_at.isoformat() if o.filled_at else None
            result.append({
                "symbol":    o.symbol,
                "action":    "BUY" if _is_buy(o.side) else "SELL",
                "qty":       float(o.filled_qty or o.qty or 0),
                "filled_at": filled_at,
                "avg_price": float(o.filled_avg_price or 0),
                # Surface partial-fill state so callers can branch.
                "partial":   "partial" in str(o.status).lower(),
            })
        return result

    def get_open_orders(self) -> set[str]:
        """Return symbols that have a pending or open order right now."""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        params = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = self._trading_client.get_orders(filter=params)
        return {o.symbol for o in orders}

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        clock = self._trading_client.get_clock()
        return clock.is_open
