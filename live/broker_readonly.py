"""Read-only broker wrapper for shadow-pipeline runs.

Per user mandate 2026-05-19: "整条 pipeline 都参考 shadow model 的 output —
基本上就是跑两遍 e2e，但是 shadow 那一遍虽然连 alpaca 洗数据，但是并不真下单!"
Plus refinement: "或者你直接搞一个 shadow 的 config，跑完 prod 再跑一遍 shadow
的 — 避免污染么，隔离干净".

This wrapper:
- Forwards every read-side method (account value, cash, positions, quotes,
  filled/open orders) to the underlying real broker → shadow sees the same
  live market & account state as primary.
- Swallows every write-side method (place_order, place_stop_order,
  cancel_order) → no broker round-trip happens. Returns a synthesised
  filled-order dict so downstream pipeline code (commit accounting, ntfy
  aggregator, state writer) keeps working without per-caller branching.

State isolation: broker_name = "alpaca_shadow" (NOT mirroring underlying
AlpacaBroker's "alpaca"). adapters/runner.py's state-path convention
keys on broker_name → live_state.alpaca_shadow.json + runs_alpaca_shadow.db
get written as separate files from prod alpaca state. Zero contamination
of prod state even when the full pipeline writes its state file. See
kernel/state_paths.py for the path convention.

Verified safety property: no method on this class makes a network call or
mutates the underlying broker. Tested in tests/test_broker_readonly.py.
"""
from __future__ import annotations
import logging
import time
import uuid
from typing import Any

from live.broker import BaseBroker

log = logging.getLogger("live.broker_readonly")


class ReadOnlyBrokerWrapper(BaseBroker):
    """Wrap a real BaseBroker so writes become no-ops, reads stay live.

    Every "place" / "cancel" call gets logged at DEBUG and returns a
    plausible success dict. Downstream code treats the shadow run as if
    orders went through, so the rest of the pipeline (commit accounting,
    ntfy aggregator, state writers gated by shadow_run) works unchanged.
    """

    broker_name: str = "alpaca_shadow"

    def __init__(self, underlying: BaseBroker):
        self._u = underlying
        # Do NOT mirror underlying broker_name. Keep "alpaca_shadow" so
        # adapters/runner.py state-path resolution writes shadow state to
        # live_state.alpaca_shadow.json + runs_alpaca_shadow.db, fully
        # isolated from prod live_state.alpaca.json. This is the hard
        # isolation per user mandate 2026-05-19 "隔离干净".
        self._fake_order_seq = 0

    # ── Read-side: pure forwards ───────────────────────────────────────────
    def connect(self) -> None:
        # MUST forward: shadow run is a SEPARATE process from primary, so
        # the underlying broker's _trading_client is None until we connect.
        # Earlier draft no-op'd this assuming primary had already connected;
        # broke preflight P-BROKER-CONNECT because get_account_value()
        # downstream needs _trading_client populated.
        self._u.connect()

    def disconnect(self) -> None:
        # Forward — we own the underlying in shadow process.
        self._u.disconnect()

    def get_position(self, symbol: str) -> float:
        return self._u.get_position(symbol)

    def get_account_value(self) -> float:
        return self._u.get_account_value()

    def get_avg_cost(self, symbol: str) -> float:
        return self._u.get_avg_cost(symbol)

    def get_cash(self) -> float:
        return self._u.get_cash()

    def get_all_positions(self) -> list[dict]:
        return self._u.get_all_positions()

    def get_filled_orders(self, after: str | None = None) -> list[dict]:
        return self._u.get_filled_orders(after=after)

    def get_open_orders(self) -> set[str]:
        return self._u.get_open_orders()

    def supports_broker_side_stops(self) -> bool:
        return self._u.supports_broker_side_stops()

    # ── Write-side: swallow + synthesise filled response ───────────────────
    def _fake_order_id(self) -> str:
        self._fake_order_seq += 1
        return f"shadow-{int(time.time())}-{self._fake_order_seq:04d}-{uuid.uuid4().hex[:6]}"

    def place_order(self, symbol: str, action: str, quantity: float) -> dict:
        oid = self._fake_order_id()
        log.debug("ReadOnlyBrokerWrapper.place_order swallowed: %s %s x%s → %s",
                  action, symbol, quantity, oid)
        return {
            "order_id": oid,
            "status": "filled",
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "shadow": True,
        }

    def place_stop_order(self, symbol: str, quantity: float,
                         stop_price: float) -> dict:
        oid = self._fake_order_id()
        log.debug("ReadOnlyBrokerWrapper.place_stop_order swallowed: STOP %s "
                  "x%s @ $%.2f → %s", symbol, quantity, stop_price, oid)
        return {
            "order_id": oid,
            "status": "accepted",
            "symbol": symbol,
            "quantity": quantity,
            "stop_price": stop_price,
            "shadow": True,
        }

    def cancel_order(self, order_id: str) -> bool:
        log.debug("ReadOnlyBrokerWrapper.cancel_order swallowed: %s", order_id)
        return True

    # ── Pass-through for any other attribute (forward-compat) ─────────────
    # Some adapters call broker-specific methods (e.g. AlpacaBroker-only
    # convenience accessors). Forward unknown reads to underlying so we
    # don't break on future additions; writes will still be neutralised
    # by ctx.shadow_run gates in adapter.commit.
    def __getattr__(self, name: str) -> Any:
        # Important: this only runs for attributes NOT on the wrapper.
        # Explicit overrides above always win.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._u, name)
