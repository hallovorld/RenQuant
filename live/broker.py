"""Broker abstraction for live trading."""

from abc import ABC, abstractmethod
from typing import Any

# ── No-submit status vocabulary (S-FRAC leg (a) of the capability gate) ──────
#
# The umbrella commit path's ``fractional_capability_gate``
# (backtesting/renquant_104/adapters/commit_contract.py) probes the broker
# object for a no-submit classifier (``classify_broker_result`` or
# ``is_no_submit_status``) before it will ever emit a fractional BUY. The
# vocabulary is OWNED by renquant-execution (``renquant_execution.broker``,
# ``NO_SUBMIT_STATUSES`` + ``is_no_submit_status``): a "no-submit" result is
# an order the adapter deliberately never sent (non-fractionable asset,
# failed asset lookup, precision/notional floor, ...) — NOT a broker
# rejection and NOT a pending order.
#
# Import the owner's copy when the pinned renquant-execution checkout is on
# PYTHONPATH (the live run has it — ``adapters/runner_execmath.py`` already
# imports ``renquant_execution.order_math`` the same guarded way). When it is
# absent (a bare umbrella venv, an older pin), FALL BACK to the local
# frozenset below, which is a verbatim copy of the owner vocabulary at
# renquant-execution 91c7bf88 (subrepos.lock.json pin). The umbrella must
# never crash on import because a sibling checkout is missing, and the
# fallback is pinned equal to the owner by
# tests/test_live_broker_fractional_contract.py so any drift trips a test
# rather than silently diverging.
_FALLBACK_NO_SUBMIT_STATUSES: frozenset[str] = frozenset({
    "rejected_non_fractionable",
    "rejected_fractionable_lookup_failed",
    "rejected_precision_exceeds_9dp",
    "rejected_below_min_notional",
    "rejected_invalid_fractional_order",
    "rejected_invalid_crypto_order",
    "rejected_crypto_no_short",
    "rejected_below_min_order_size",
    "rejected_crypto_spec_lookup_failed",
    # Legacy floor-to-zero status, kept recognized for back-compat audit replay.
    "skipped_non_fractionable_dust",
})

try:
    from renquant_execution.broker import (  # type: ignore[import-not-found]
        NO_SUBMIT_STATUSES as NO_SUBMIT_STATUSES,
    )
    NO_SUBMIT_VOCABULARY_SOURCE = "renquant_execution"
except Exception:  # noqa: BLE001 — absent / older pin / broken sibling: fall back
    NO_SUBMIT_STATUSES = _FALLBACK_NO_SUBMIT_STATUSES
    NO_SUBMIT_VOCABULARY_SOURCE = "local_fallback"


def is_no_submit_status(status: Any) -> bool:
    """Whether ``status`` denotes a result that never reached the broker.

    Same normalisation as the owner (``renquant_execution.broker
    .is_no_submit_status``): ``None``/empty → False; case- and
    whitespace-insensitive membership in :data:`NO_SUBMIT_STATUSES`.
    """
    return str(status or "").strip().lower() in NO_SUBMIT_STATUSES


class BaseBroker(ABC):
    """Interface for order execution backends."""

    # Broker tag for state-file isolation. Each subclass overrides via
    # class attribute or @property. Used by adapters/runner.py to compute
    # broker-specific paths for live_state.json and runs.db so a paper
    # smoke can never contaminate alpaca live state. See
    # kernel/state_paths.py for the path convention.
    broker_name: str = "unknown"

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_position(self, symbol: str) -> float:
        """Return current share count for *symbol* (0 if flat)."""
        ...

    @abstractmethod
    def get_account_value(self) -> float:
        """Return total account liquidation value."""
        ...

    def get_avg_cost(self, symbol: str) -> float:
        """Return average cost basis per share for *symbol* (0 if not held). Override for accuracy."""
        return 0.0

    def get_cash(self) -> float:
        """Return available cash.  Defaults to account value; override for accuracy."""
        return self.get_account_value()

    def get_all_positions(self) -> list[dict]:
        """Return all open positions as a list of dicts with keys:
        symbol, qty, avg_entry_price, market_value, unrealized_pl.
        Override for batch-efficient broker implementations.
        """
        return []

    def get_filled_orders(self, after: str | None = None) -> list[dict]:
        """Return filled orders since *after* ('YYYY-MM-DD').  Returns [] if not supported."""
        return []

    def get_open_orders(self) -> set[str]:
        """Return the set of symbols that have a pending/open order right now.
        Used to avoid placing duplicate orders when a catch-up run fires after a
        prior run already submitted orders that haven't filled yet.
        Returns an empty set if not supported by this broker.
        """
        return set()

    @abstractmethod
    def place_order(self, symbol: str, action: str, quantity: float) -> dict:
        """Place a market order.

        Args:
            symbol:   Ticker.
            action:   ``"BUY"`` or ``"SELL"``.
            quantity: Number of shares (positive).

        Returns:
            Order confirmation dict with at least ``{"order_id", "status"}``.
        """
        ...

    # ── Broker-side stop orders (Z9, 2026-04-28) ─────────────────────────────
    # Invariant: stops live broker-side, not in our polling loop.
    # NVTS post-mortem: 30-min cron lag let −12% drop happen between
    # ticks. Broker-side stops trigger in ms regardless of our poll
    # cadence. Default impls raise NotImplementedError so brokers that
    # don't support stops fail loudly rather than silently no-op.

    def supports_broker_side_stops(
        self, symbol: str | None = None, qty: float | None = None,
    ) -> bool:
        """Whether this broker supports broker-side stop orders.

        Brokers that do (Alpaca, IBKR, Paper-with-simulator) override and
        return True. Brokers that don't return False so the runner falls
        back to polled stop_loss without needing per-broker code paths.

        S-FRAC stage 0 (design 2026-07-02 §2.2.2): the signature is
        quantity-aware. Called with no arguments it answers the broker-
        level capability (legacy Z9 enable check). Called with
        ``(symbol, qty)`` it must answer for THAT quantity — brokers
        whose stop path is whole-share-only (GTC stops reject/truncate
        fractional qty) must return False for a non-integral qty so the
        Z9 router fails closed instead of placing a truncated stop.
        """
        return False

    def place_stop_order(
        self, symbol: str, quantity: float, stop_price: float,
    ) -> dict:
        """Place a sell-stop order at *stop_price* for *quantity* shares of *symbol*.

        Stops are GTC (good-til-canceled). Caller is responsible for
        cancelling the stop when the position is sold or replaced.

        Returns dict with at least ``{"order_id", "status"}``.

        Default raises NotImplementedError — supporters override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support broker-side stop orders. "
            "Check supports_broker_side_stops() before calling."
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a previously-placed order by id.

        Returns True if cancellation accepted (not necessarily filled-cancel
        round-trip), False if the order was unknown or already filled.

        Default raises NotImplementedError — supporters override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement cancel_order."
        )

    # ── Fractional capability contract (S-FRAC gate leg (a)) ─────────────────
    # ``fractional_capability_gate`` (adapters/commit_contract.py) requires,
    # with ``execution.fractional_shares.enabled`` ON, a callable
    # ``is_fractionable`` AND a callable no-submit classifier on the broker
    # object. The classifier is pure vocabulary, so it lives here on the base
    # (mirrors renquant_execution.broker.BaseBroker.is_no_submit_status).
    #
    # ``is_fractionable`` is DELIBERATELY NOT defaulted on the base: the gate
    # treats its presence as "this broker can answer per-asset
    # fractionability", so a base default (any callable) would make every
    # broker — paper, read-only wrapper, fakes — pass the structural probe
    # and turn the gate fail-open. Only brokers with a real asset lookup
    # define it (live/alpaca_broker.py).

    @staticmethod
    def is_no_submit_status(status: Any) -> bool:
        """Instance-callable no-submit classifier (capability-gate probe).

        Delegates to the module-level vocabulary so every adapter answers
        the same way.
        """
        return is_no_submit_status(status)
