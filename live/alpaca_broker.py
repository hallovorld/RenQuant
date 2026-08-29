"""Alpaca broker for live and paper trading.

Requires ``alpaca-py`` (``pip install alpaca-py``) and API credentials
set via environment variables or passed at construction:

    ALPACA_API_KEY
    ALPACA_SECRET_KEY

By default connects to the paper trading endpoint.  Pass ``paper=False``
for live trading.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

from .broker import BaseBroker

log = logging.getLogger(__name__)

# P-BROKER-CONNECT bounded account read (twin of renquant-execution#41; this
# pair is a DELIBERATE diverged_pin, so the behaviour is ported, not the file).
#
# The alpaca-py SDK exposes NO timeout knob -- `RESTClient.__init__` has no
# `timeout` parameter and `_one_request` never passes one -- so every account
# read here inherits requests' default `timeout=None` and can hang on the OS
# TCP timeout. That is the 2026-08-11 07:00 abort that cost a ~12 min intraday
# cycle. `live/runner.py` imports THIS module (`from .alpaca_broker import
# AlpacaBroker`), so execution#41's fix does not reach the order path; this is
# that fix on the stack that actually trades.
#
# Values: a healthy Alpaca `GET /v2/account` returns in well under a second, so
# 5s connect / 10s read is ample slack for a transient blip without tolerating
# an open-ended hang. This bounds NO-PROGRESS stalls -- requests' timeout is an
# inactivity timer, not a wall-clock cap on the whole request.
_BROKER_CONNECT_TIMEOUT_SECONDS = 5.0
_BROKER_READ_TIMEOUT_SECONDS = 10.0


class _FractionableLookupError(RuntimeError):
    """Asset fractionability could not be determined (lookup failed or the
    trading client is not connected). Raised — never cached — so a transient
    error is never remembered as an authoritative "not fractionable"."""


class AlpacaBroker(BaseBroker):
    """Execute orders via Alpaca Markets API.

    Multi-account support (2026-05-15): credentials are loaded from
    environment variables keyed by ``env_prefix`` (default ``ALPACA``).
    This lets distinct broker instances target distinct Alpaca accounts:

      env_prefix="ALPACA"          → ALPACA_API_KEY / ALPACA_SECRET_KEY (live or paper sandbox)
      env_prefix="ALPACA_SHORTS"   → ALPACA_SHORTS_API_KEY / ALPACA_SHORTS_SECRET_KEY (paper, shorts-only testing)

    The ``label`` parameter customizes ``broker_name`` so each account
    gets its own state-file namespace (see kernel.state_paths). Without
    label collisions, live_state.alpaca.json and live_state.alpaca-shorts.json
    track positions independently.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
        env_prefix: str = "ALPACA",
        label: str | None = None,
    ):
        self._env_prefix = env_prefix
        self._api_key = api_key or os.environ.get(f"{env_prefix}_API_KEY", "")
        self._secret_key = secret_key or os.environ.get(f"{env_prefix}_SECRET_KEY", "")
        self._paper = paper
        self._label = label
        self._trading_client = None
        self._data_client = None
        self._order_counter = 0
        # symbol (upper) -> confirmed fractionable verdict. Only CONFIRMED
        # lookups are stored (see _lookup_fractionable).
        self._fractionable_cache: dict[str, bool] = {}

    @property
    def broker_name(self) -> str:  # state-file isolation tag (see kernel.state_paths)
        if self._label is not None:
            return self._label
        return "alpaca-paper" if self._paper else "alpaca"

    @contextlib.contextmanager
    def _bounded_account_timeout(self) -> Any:
        """Temporarily WRAP the trading client's HTTP session so the preflight
        account reads carry a bounded ``(connect, read)`` timeout, restoring the
        session EXACTLY as it was on exit.

        Wrap, not replace: swapping in a fresh session would silently drop the
        SDK's seeded ``proxies`` / ``verify`` / ``cert`` / ``cookies`` /
        ``hooks`` / ``params`` / ``auth`` and mounted adapters. Overriding just
        the SAME object's ``request`` preserves every piece of transport state
        by construction. **Order submission never runs inside this context**, so
        its socket semantics are byte-for-byte unchanged.

        Raises rather than degrading silently: if the client has no usable
        session, an unbounded fallback would defeat the fast-fail contract this
        exists to provide, so it fails loud and closed inside the caller's
        bounded P-BROKER-CONNECT retry.
        """
        session = getattr(self._trading_client, "_session", None)
        original_request = getattr(session, "request", None)
        if session is None or not callable(original_request):
            raise RuntimeError(
                "AlpacaBroker cannot arm a bounded account-read timeout: the "
                "trading client's HTTP session is missing or has no callable "
                f"'request' (session={session!r}, type={type(session).__name__}). "
                "Refusing to run the account read unbounded."
            )
        timeout = (_BROKER_CONNECT_TIMEOUT_SECONDS, _BROKER_READ_TIMEOUT_SECONDS)
        had_own = "request" in vars(session)

        def _bounded_request(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("timeout", timeout)  # respect a caller's own
            return original_request(*args, **kwargs)

        session.request = _bounded_request
        try:
            yield
        finally:
            if had_own:
                session.request = original_request
            else:
                del session.request

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

        # Preflight read 1 of 2 — bounded so a stalled socket fails fast.
        with self._bounded_account_timeout():
            account = self._trading_client.get_account()
        mode = "paper" if self._paper else "LIVE"
        nmbp = getattr(account, "non_marginable_buying_power", None)
        log.info(
            "Alpaca connected (%s) — account=%s, equity=$%s, settled_cash=$%s, "
            "non_marginable_buying_power=$%s, status=%s",
            mode, account.account_number, account.equity, account.cash, nmbp,
            account.status,
        )

        # P0-10 (audit 2026-05-20) — defensive guard against silent
        # paper-key-vs-live-broker mismatch. If RENQUANT_EXPECTED_LIVE_ACCOUNT
        # is set in .env, assert connected account matches expected. Per
        # 2026-05-17 e2e mandate "我他妈说了一万遍了 LIVE account": never
        # silently flip to paper. If env is missing in LIVE mode, fail closed
        # so a daily e2e cannot proceed without positive account identity.
        if not self._paper:
            expected = os.environ.get("RENQUANT_EXPECTED_LIVE_ACCOUNT")
            actual = str(account.account_number)
            if not expected:
                raise RuntimeError(
                    "RENQUANT_EXPECTED_LIVE_ACCOUNT is required for LIVE Alpaca "
                    f"connections; connected account_number={actual}. Pin this "
                    "in .env before running real-money daily e2e."
                )
            if actual != expected:
                raise RuntimeError(
                    f"ALPACA LIVE-ACCOUNT MISMATCH: connected to "
                    f"account_number={actual} but RENQUANT_EXPECTED_LIVE_ACCOUNT={expected}. "
                    f"This guard prevents silent paper/live key swap. If new account is "
                    f"intentional, update .env and restart."
                )
            log.info("LIVE account guard PASSED (account_number=%s matches expected)", actual)

        if account.status != "ACTIVE":
            log.warning("Account status is %s — trading may be restricted", account.status)

    def disconnect(self) -> None:
        self._trading_client = None
        self._data_client = None
        log.info("Alpaca disconnected")

    def get_last_price(self, symbol: str) -> float:
        """Latest trade price via alpaca-py market data (IEX feed — free
        tier cannot query current-day SIP, same constraint as
        kernel/data.py:fetch_intraday_bars). Used by the G2 breaker for
        pre-trade notional accounting; raises on any failure so callers
        decide how to degrade."""
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest
        from alpaca.data.enums import DataFeed

        if getattr(self, "_data_client", None) is None:
            self._data_client = StockHistoricalDataClient(
                api_key=self._api_key, secret_key=self._secret_key,
            )
        req = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
        trades = self._data_client.get_stock_latest_trade(req)
        price = float(trades[symbol].price)
        if price <= 0:
            raise ValueError(f"get_last_price({symbol}): non-positive price {price}")
        return price

    def get_position(self, symbol: str) -> float:
        from alpaca.common.exceptions import APIError

        try:
            position = self._trading_client.get_open_position(symbol)
            return float(position.qty)
        except APIError:
            return 0.0

    def get_account_value(self) -> float:
        # Preflight read 2 of 2. Bounded HERE rather than in a shared helper so
        # every other account reader on the order path keeps its existing,
        # unbounded behaviour.
        with self._bounded_account_timeout():
            account = self._trading_client.get_account()
        return float(account.equity)

    def get_cash(self) -> float:
        """Return available cash for new orders.

        P0-9 (BUG D, audit 2026-05-20) — fixed 2026-05-20.
        Pre-fix: returned `account.cash` (SETTLED ONLY) — excludes T+N
        pending sell proceeds that Alpaca's margin account treats as
        immediately spendable buying power. Result: live under-stated
        cash post-sell vs sim path (which includes pending settlement
        via `sim.py:_t2_queue.pending_total()`).

        Post-fix: returns `account.non_marginable_buying_power` which is
        Alpaca's "cash + unsettled sell proceeds, no 2x/4x margin" field.
        Matches sim's T+N-aware accounting. For non-margin (cash) accounts this equals
        `cash`. For margin accounts it's `cash + pending`.

        We deliberately do NOT use `account.buying_power` (the 2× / 4× margin
        amount) — that would over-state available cash and break the
        non-margin policy.
        """
        account = self._trading_client.get_account()
        # Field availability check (older alpaca-py may lack it)
        nmbp = getattr(account, 'non_marginable_buying_power', None)
        if nmbp is not None:
            try:
                return float(nmbp)
            except (TypeError, ValueError):
                pass
        # Fallback to settled cash (legacy behavior; logged as warning)
        log.warning("alpaca account.non_marginable_buying_power unavailable; "
                    "falling back to settled cash (pending settlement NOT counted)")
        return float(account.cash)

    def place_order(self, symbol: str, action: str, quantity: float) -> dict:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        # G2 agent breaker (2026-06-12, eng plan §III.4 disaster guards):
        # hard daily caps below ALL pipeline logic + manual TRADING_OFF file.
        # BreakerTripped propagates like any broker failure (recorded as
        # EXITS-FAIL / orders_skipped by the adapter) and must not be retried.
        from live.agent_breaker import AgentBreaker  # noqa: PLC0415
        if not hasattr(self, "_g2_breaker"):
            self._g2_breaker = AgentBreaker()
        notional = None
        try:
            notional = abs(float(quantity)) * self.get_last_price(symbol)
        except Exception as exc:
            # Price unavailable: order slot is still consumed; notional cap
            # cannot account this order. Loud so degraded accounting is
            # visible in the run log (no silent-continue, eng plan §IV.1).
            log.warning("G2: last price unavailable for %s (%s) — "
                        "count-only accounting for this order", symbol, exc)
        self._g2_breaker.admit(symbol=symbol, notional=notional)

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
            "filled_qty": float(getattr(order, "filled_qty", 0) or 0),
            "filled_avg_price": float(getattr(order, "filled_avg_price", 0) or 0),
            "submitted_at": str(getattr(order, "submitted_at", "") or ""),
            "filled_at": str(getattr(order, "filled_at", "") or ""),
        }

    # ── Broker-side stop orders (Z9, 2026-04-28) ────────────────────────────
    # Invariant: stops live broker-side. NVTS post-mortem: 30-min cron
    # cadence let −12% drop happen between polled stop checks. Broker-side
    # stops trigger in ms.

    def supports_broker_side_stops(
        self, symbol: str | None = None, qty: float | None = None,
    ) -> bool:
        # S-FRAC stage 0 (§2.2.2): qty-aware capability. This adapter's
        # stop path submits WHOLE-SHARE GTC stops (place_stop_order casts
        # qty to int; Alpaca fractional orders are TIF=DAY only — no GTC),
        # so a fractional quantity is NOT protectable here. Fail closed:
        # the Z9 router will route it to the software-stop layer (stage 3)
        # or refuse the entry. No-arg / integral-qty answers are unchanged.
        if qty is None:
            return True
        import math  # noqa: PLC0415

        try:
            q = float(qty)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(q) or q <= 0:
            return False
        return abs(q - round(q)) <= 1e-9

    def place_stop_order(
        self, symbol: str, quantity: float, stop_price: float,
    ) -> dict:
        """Place a GTC sell-stop at stop_price."""
        from alpaca.trading.requests import StopOrderRequest  # noqa: PLC0415
        from alpaca.trading.enums import OrderSide, TimeInForce  # noqa: PLC0415

        if quantity <= 0:
            raise ValueError(f"place_stop_order: quantity must be positive (got {quantity})")
        if stop_price <= 0:
            raise ValueError(f"place_stop_order: stop_price must be positive (got {stop_price})")

        # Account-status check (mirrors place_order — see ALPACA-ACCT-STATUS).
        try:
            account = self._trading_client.get_account()
            status = str(getattr(account, "status", ""))
        except Exception as exc:
            raise RuntimeError(
                f"alpaca pre-stop account check failed: {exc}"
            ) from exc
        if status not in ("ACTIVE", "AccountStatus.ACTIVE"):
            raise RuntimeError(
                f"alpaca account status is '{status}' (not ACTIVE) — refusing to place "
                f"stop {symbol} x{quantity} @ ${stop_price:.2f}."
            )

        # GTC so the stop survives across days (matches the invariant —
        # we want the stop active until either it triggers or we cancel).
        request = StopOrderRequest(
            symbol=symbol,
            qty=int(quantity),
            side=OrderSide.SELL,
            stop_price=round(float(stop_price), 2),
            time_in_force=TimeInForce.GTC,
        )
        order = self._trading_client.submit_order(request)
        log.info(
            "Stop order %s: SELL %s %d @ stop=$%.2f — status=%s",
            order.id, symbol, int(quantity), stop_price, order.status,
        )
        return {
            "order_id":  str(order.id),
            "status":    str(order.status),
            "symbol":    symbol,
            "quantity":  int(quantity),
            "stop_price": float(stop_price),
        }

    # ── Fractional capability contract (S-FRAC gate leg (a)) ─────────────────
    # Ported semantics of renquant-execution ``AlpacaBroker._lookup_fractionable``
    # / ``is_fractionable`` (src/renquant_execution/alpaca_broker.py, pin
    # 91c7bf88). ``live/runner.py`` imports THIS module, so the execution
    # repo's implementation does not reach the order path (deliberate
    # diverged_pin — same situation as the bounded-timeout port above).
    #
    # These methods are capability PROBES + a cached asset lookup. Nothing in
    # this module's order-submission path calls them; ``place_order`` /
    # ``place_stop_order`` are byte-identical to the pre-port module. The
    # commit-path gate only reads them when
    # ``execution.fractional_shares.enabled`` is true.

    def _lookup_fractionable(self, symbol: str) -> bool:
        """Return whether ``symbol`` is fractionable, caching only CONFIRMED
        lookups. Raises ``_FractionableLookupError`` when the client is not
        connected or the asset lookup fails, so a transient error is never
        cached as an authoritative verdict."""
        key = str(symbol).upper()
        # Lazy init: tests (and any caller) may construct via
        # ``AlpacaBroker.__new__`` without running ``__init__``.
        cache = self.__dict__.setdefault("_fractionable_cache", {})
        cached = cache.get(key)
        if cached is not None:
            return cached
        client = getattr(self, "_trading_client", None)
        if client is None:
            raise _FractionableLookupError(
                f"trading client not connected; cannot look up asset {symbol!r}"
            )
        try:
            asset = client.get_asset(symbol)
        except Exception as exc:  # noqa: BLE001 — surface as a fail-closed signal
            raise _FractionableLookupError(repr(exc)) from exc
        fractionable = bool(getattr(asset, "fractionable", False))
        cache[key] = fractionable
        return fractionable

    def is_fractionable(self, symbol: str) -> bool:
        """Whether ``symbol`` supports fractional Alpaca orders (cached).

        Returns ``False`` on lookup failure (safe default) but, unlike a
        confirmed lookup, does NOT cache that failure — so a later call
        retries rather than treating a transient error as a permanent
        verdict. Callers that must distinguish "confirmed non-fractionable"
        from "lookup failed" use ``_lookup_fractionable`` directly.
        """
        try:
            return self._lookup_fractionable(symbol)
        except _FractionableLookupError as exc:
            log.warning(
                "is_fractionable(%s): lookup failed, answering False "
                "(not cached; will retry): %s", symbol, exc,
            )
            return False

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns False on already-filled / unknown id."""
        from alpaca.common.exceptions import APIError  # noqa: PLC0415
        try:
            self._trading_client.cancel_order_by_id(order_id)
            log.info("Cancelled order %s", order_id)
            return True
        except APIError as exc:
            log.warning("cancel_order(%s) failed: %s", order_id, exc)
            return False

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
            # Codex #76: surface ``order_id`` so STATE-EXT-SELL attribution
            # can correlate a fill against tracked Z9 stop order_ids. Without
            # this field, every Z9 stop fill mis-classifies as
            # "external_or_manual" in the daily warning log.
            result.append({
                "order_id":  str(getattr(o, "id", "") or ""),
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
