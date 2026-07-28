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
keys on broker_name → live_state.alpaca_shadow.json + runs.alpaca_shadow.db
get written as separate files from prod alpaca state. Zero contamination
of prod state even when the full pipeline writes its state file. See
kernel/state_paths.py for the path convention.

Shadow-lane tag (2026-07-27, shadow_blend rail — operator directive): the
broker tag is now parameterizable so MULTIPLE readonly shadow lanes can
coexist with disjoint state. Threading mechanism: the ``RENQUANT_READONLY_TAG``
environment variable (read at wrapper construction), chosen over a
``--broker readonly-alpaca:TAG`` CLI syntax because the orchestrator
live-bridge validates ``--broker`` against a fixed set (ALPACA_BROKERS in
renquant_orchestrator/live_bridge.py) and env vars thread through the
bridge subprocess boundary with zero orchestrator changes. Default
(env unset/empty) = "alpaca_shadow" → byte-identical legacy behavior.
Tag "alpaca_shadow_blend" → live_state.alpaca_shadow_blend.json +
runs.alpaca_shadow_blend.db. Every tag MUST start with "alpaca_shadow"
(preserves prod-state isolation AND live/runner.py's startswith-based
readonly-label + shadow-preflight checks) and MUST be [A-Za-z0-9_]+ (no
path traversal). Invalid tags raise ValueError — fail closed rather than
silently falling back to the legacy tag and contaminating its state.
NOTE: new tags must also be added to ALLOWED_BROKERS in
backtesting/renquant_104/kernel/state_paths.py (single-source allowlist).

Verified safety property: no method on this class makes a network call or
mutates the underlying broker. Tested in tests/test_broker_readonly_tag.py
(tag routing) + tests/test_runner_trade_ntfy.py (ntfy title contract).
"""
from __future__ import annotations
import logging
import os
import re
import time
import uuid
from typing import Any

from live.broker import BaseBroker

log = logging.getLogger("live.broker_readonly")

#: Legacy default tag — MUST stay "alpaca_shadow" (byte-identical legacy lane).
DEFAULT_READONLY_TAG = "alpaca_shadow"

#: Env var that selects the shadow-lane tag (see module docstring).
READONLY_TAG_ENV = "RENQUANT_READONLY_TAG"

_TAG_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def validate_readonly_tag(tag: str) -> str:
    """Validate a shadow-lane broker tag; return it unchanged.

    Raises ValueError (fail closed) unless the tag is [A-Za-z0-9_]+ AND
    starts with "alpaca_shadow". The prefix rule is load-bearing:
    - state-path isolation from prod "alpaca" state files;
    - live/runner.py keys the [READONLY] ntfy label and the shadow
      preflight-strictness branch off broker_name.startswith("alpaca_shadow").
    """
    if not _TAG_PATTERN.fullmatch(tag or ""):
        raise ValueError(
            f"Invalid readonly shadow tag {tag!r}: must match [A-Za-z0-9_]+ "
            f"(filename-safe, no path separators)."
        )
    if not tag.startswith(DEFAULT_READONLY_TAG):
        raise ValueError(
            f"Invalid readonly shadow tag {tag!r}: must start with "
            f"{DEFAULT_READONLY_TAG!r} to preserve prod-state isolation and "
            f"the runner's readonly-label/preflight checks."
        )
    return tag


def resolve_readonly_tag() -> str:
    """Resolve the shadow-lane tag from RENQUANT_READONLY_TAG.

    Unset/empty env → DEFAULT_READONLY_TAG ("alpaca_shadow", legacy lane).
    A set-but-invalid env raises ValueError — the run aborts instead of
    silently writing into the legacy lane's state files.
    """
    raw = os.environ.get(READONLY_TAG_ENV, "").strip()
    if not raw:
        return DEFAULT_READONLY_TAG
    return validate_readonly_tag(raw)


class ReadOnlyBrokerWrapper(BaseBroker):
    """Wrap a real BaseBroker so writes become no-ops, reads stay live.

    Every "place" / "cancel" call gets logged at DEBUG and returns a
    plausible success dict. Downstream code treats the shadow run as if
    orders went through, so the rest of the pipeline (commit accounting,
    ntfy aggregator, state writers gated by shadow_run) works unchanged.
    """

    broker_name: str = DEFAULT_READONLY_TAG

    def __init__(self, underlying: BaseBroker, tag: str | None = None):
        self._u = underlying
        # Do NOT mirror underlying broker_name. Keep an "alpaca_shadow*"
        # tag so adapters/runner.py state-path resolution writes shadow
        # state to live_state.<tag>.json + runs.<tag>.db, fully isolated
        # from prod live_state.alpaca.json. This is the hard isolation per
        # user mandate 2026-05-19 "隔离干净".
        #
        # 2026-07-27 shadow_blend rail: tag now parameterized. Explicit
        # ctor arg wins; otherwise RENQUANT_READONLY_TAG; otherwise the
        # legacy "alpaca_shadow" (byte-identical legacy lane). Both paths
        # validate (fail closed) — see validate_readonly_tag.
        if tag is not None:
            self.broker_name = validate_readonly_tag(tag)
        else:
            self.broker_name = resolve_readonly_tag()
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

    def supports_broker_side_stops(
        self, symbol: str | None = None, qty: float | None = None,
    ) -> bool:
        # S-FRAC stage 0 (§2.2.2): pass the qty-aware probe through so a
        # readonly/shadow run answers exactly like the underlying broker
        # (fail-closed parity for fractional quantities).
        if symbol is None and qty is None:
            return self._u.supports_broker_side_stops()
        try:
            return self._u.supports_broker_side_stops(symbol, qty)
        except TypeError:
            # Underlying broker predates the qty-aware signature: a
            # fractional qty is unprotectable there — fail closed.
            try:
                q = float(qty)
            except (TypeError, ValueError):
                return False
            if abs(q - round(q)) > 1e-9:
                return False
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
