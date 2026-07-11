"""Broker abstraction for live trading."""

from abc import ABC, abstractmethod
from typing import Any

# Statuses that denote a result that never reached the broker (S-FRAC stage 0,
# adapters/commit_contract.py::fractional_capability_gate's no-submit-
# classifier probe). Mirrors renquant-execution's NO_SUBMIT_STATUSES
# vocabulary exactly (renquant_execution.broker) so a future migration of the
# live order path onto that adapter is a behavioral no-op for this check.
# Statuses this broker's own place_order/place_notional_order do not
# currently produce (the fractional/crypto-specific ones) are included as a
# forward-compatible superset, matching renquant-execution's own convention
# of keeping the vocabulary broker-generic rather than broker-specific.
NON_FRACTIONABLE_STATUS = "rejected_non_fractionable"
FRACTIONABLE_LOOKUP_FAILED_STATUS = "rejected_fractionable_lookup_failed"
PRECISION_EXCEEDS_9DP_STATUS = "rejected_precision_exceeds_9dp"
BELOW_MIN_NOTIONAL_STATUS = "rejected_below_min_notional"
INVALID_FRACTIONAL_ORDER_STATUS = "rejected_invalid_fractional_order"
INVALID_CRYPTO_ORDER_STATUS = "rejected_invalid_crypto_order"
CRYPTO_NO_SHORT_STATUS = "rejected_crypto_no_short"
BELOW_MIN_ORDER_SIZE_STATUS = "rejected_below_min_order_size"
CRYPTO_SPEC_LOOKUP_FAILED_STATUS = "rejected_crypto_spec_lookup_failed"
NO_SUBMIT_STATUSES = frozenset({
    NON_FRACTIONABLE_STATUS,
    FRACTIONABLE_LOOKUP_FAILED_STATUS,
    PRECISION_EXCEEDS_9DP_STATUS,
    BELOW_MIN_NOTIONAL_STATUS,
    INVALID_FRACTIONAL_ORDER_STATUS,
    INVALID_CRYPTO_ORDER_STATUS,
    CRYPTO_NO_SHORT_STATUS,
    BELOW_MIN_ORDER_SIZE_STATUS,
    CRYPTO_SPEC_LOOKUP_FAILED_STATUS,
    # Legacy floor-to-zero status, kept recognized for back-compat audit replay.
    "skipped_non_fractionable_dust",
})


def is_no_submit_status(status: Any) -> bool:
    """Whether ``status`` denotes a result that never reached the broker."""
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

    # ── Fractional-shares capability contract (S-FRAC stage 0) ──────────────
    # adapters/commit_contract.py::fractional_capability_gate probes every
    # broker for a callable is_fractionable AND a callable no-submit
    # classifier (classify_broker_result or is_no_submit_status) before
    # allowing execution.fractional_shares.enabled=True to admit a BUY.
    # is_no_submit_status is shared here (not per-subclass) so every backend
    # answers with the same vocabulary; is_fractionable has no safe generic
    # default (whether a symbol trades fractionally is broker-specific) and
    # is deliberately NOT declared here — a subclass that does not override
    # it correctly fails the gate's callable-attribute check, which is the
    # intended fail-closed behavior for a broker with no fractional support.

    @staticmethod
    def is_no_submit_status(status: Any) -> bool:
        """Instance-callable no-submit classifier (S-FRAC stage-0 gate probe).

        Delegates to the module-level vocabulary so every subclass answers
        the same way; mirrors renquant-execution's identical BaseBroker
        method (renquant_execution.broker.BaseBroker.is_no_submit_status).
        """
        return is_no_submit_status(status)
