"""Broker abstraction for live trading."""

from abc import ABC, abstractmethod
from decimal import ROUND_DOWN, Decimal
import math
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


# ── Whole-share snap / fail-closed fractional discipline (S-FRAC step 2) ─────
#
# The live order path used to build every request with ``qty=int(quantity)``:
# a non-integral intent was SILENTLY truncated (0.435578 → 0 shares, 7.5 → 7).
# This block ports the SEMANTICS of the owner's discipline
# (renquant-execution pin 91c7bf88, ``broker.py`` ``is_whole_share`` /
# ``validate_fractional_order`` and ``alpaca_broker.py::place_order``):
#
#   * an eps-integral quantity (within ``QTY_INTEGRAL_EPS`` of an integer)
#     is a WHOLE-SHARE order and is snapped to that integer — the ONE
#     sanctioned whole-share branch (same rule as the commit contract's
#     ``normalize_fill_qty`` and ``supports_broker_side_stops``);
#   * anything else is a FRACTIONAL intent and is NEVER truncated: it is
#     either submitted exactly (on the broker's 9dp grid, rounded DOWN so
#     the submitted qty never exceeds the intent) or REFUSED with
#     :class:`FractionalOrderRefused` — an explicit no-submit outcome;
#   * BEFORE that split, every quantity must be finite and strictly
#     positive (:func:`validate_order_quantity`), and a whole-share snap
#     must be > 0 — otherwise :class:`InvalidOrderQuantity`, raised before
#     any I/O. ``is_whole_share`` itself stays the owner's pure eps-integral
#     predicate (0 IS integral); the zero/negative refusal is the caller's.
#
# The constants are replicated verbatim (the repos cannot import each other;
# ``tests/test_live_broker_no_silent_truncation.py`` pins them against the
# owner when the sibling checkout is present). Alpaca facts pinned by the
# S-FRAC v2 design inventory (§4, verified 2026-07-02): fractional qty is
# accepted only for ``fractionable=True`` assets, on MARKET orders with
# time-in-force DAY (no GTC on any fractional order), up to 9 decimal
# places, minimum notional $1.
QTY_INTEGRAL_EPS = 1e-9
MAX_ORDER_DECIMAL_PLACES = 9
MIN_FRACTIONAL_NOTIONAL_USD = 1.0
FRACTIONAL_ORDER_TYPE = "market"
FRACTIONAL_TIME_IN_FORCE = "day"

# No-submit statuses a refusal carries (members of NO_SUBMIT_STATUSES).
NON_FRACTIONABLE_STATUS = "rejected_non_fractionable"
FRACTIONABLE_LOOKUP_FAILED_STATUS = "rejected_fractionable_lookup_failed"
BELOW_MIN_NOTIONAL_STATUS = "rejected_below_min_notional"
INVALID_FRACTIONAL_ORDER_STATUS = "rejected_invalid_fractional_order"


def is_whole_share(quantity: Any) -> bool:
    """True iff ``quantity`` is a finite, eps-integral (whole-share) amount.

    Non-numeric / non-finite input answers False (it is not a whole-share
    quantity), leaving the fractional preflight to refuse it explicitly.
    """
    try:
        q = float(quantity)
    except (TypeError, ValueError):
        return False
    return math.isfinite(q) and abs(q - round(q)) <= QTY_INTEGRAL_EPS


def snap_qty_to_broker_grid(quantity: float) -> float:
    """Round ``quantity`` DOWN (toward zero) onto the broker's 9dp grid.

    Never rounds up past the intent: a submitted fractional qty is at most
    what the pipeline asked for. ``repr`` (shortest round-trip) feeds the
    Decimal so 0.435578 stays 0.435578 rather than its binary expansion.
    """
    q = float(quantity)
    if not math.isfinite(q):
        raise ValueError(f"cannot snap non-finite quantity {quantity!r}")
    grid = Decimal(1).scaleb(-MAX_ORDER_DECIMAL_PLACES)
    return float(Decimal(repr(q)).quantize(grid, rounding=ROUND_DOWN))


class InvalidOrderQuantity(ValueError):
    """The requested order quantity is not a finite, strictly positive number
    — or a whole-share intent snaps to zero (e.g. ``5e-10`` is eps-integral to
    ZERO). Refused BEFORE ANY I/O: no price read, no asset lookup, no account
    read, no breaker slot, no submission. Applies to every quantity, integral
    or not, and runs ahead of the whole-share / fractional split.

    Subclasses ``ValueError`` so the runner's existing ``except Exception``
    order handlers absorb it exactly like :class:`FractionalOrderRefused`
    (BUY → ``ctx.orders_skipped`` ``broker_error:InvalidOrderQuantity``;
    SELL → ``ctx.exits_failed``; z9_stops → warning, no stop).
    """

    def __init__(self, symbol: str, quantity: Any, reason: str) -> None:
        self.symbol = symbol
        self.quantity = quantity
        self.reason = reason
        super().__init__(
            f"invalid order quantity (no submit) for {symbol} "
            f"qty={quantity!r}: {reason}"
        )


def validate_order_quantity(symbol: str, quantity: Any) -> float:
    """Return ``float(quantity)`` iff it is finite and strictly positive.

    Raises :class:`InvalidOrderQuantity` otherwise (non-numeric, NaN, ±inf,
    zero, negative) — and ALSO for a positive value that is eps-integral to
    ZERO (``5e-10``: ``is_whole_share`` is True and the whole-share snap
    would be ``qty=0``). Pure — no I/O — so callers run it before their
    first read.
    """
    try:
        q = float(quantity)
    except (TypeError, ValueError):
        raise InvalidOrderQuantity(
            symbol, quantity, f"quantity is not numeric: {quantity!r}",
        ) from None
    if not math.isfinite(q):
        raise InvalidOrderQuantity(
            symbol, quantity, f"quantity must be finite, got {quantity!r}",
        )
    if q <= 0.0:
        raise InvalidOrderQuantity(
            symbol, quantity, f"quantity must be strictly positive, got {quantity!r}",
        )
    if is_whole_share(q) and round(q) <= 0:
        raise InvalidOrderQuantity(
            symbol, quantity,
            f"whole-share quantity {quantity!r} snaps to zero — qty=0 is never submitted",
        )
    return q


class FractionalOrderRefused(ValueError):
    """A NON-integral order intent was refused before the account read, the
    breaker admit and the order submission (a price read and the asset
    lookup — metadata I/O — may already have happened).

    Raised by the live broker's order paths instead of truncating. Carries
    the ``symbol``, the requested ``quantity``, a human ``reason`` and a
    ``status`` from the no-submit vocabulary (so
    ``is_no_submit_status(exc.status)`` is True). It subclasses
    ``ValueError`` so the runner's existing ``except Exception`` order
    handlers (adapters/runner.py BUY → ``ctx.orders_skipped`` with
    ``skip_reason="broker_error:FractionalOrderRefused"``; SELL →
    ``ctx.exits_failed``; z9_stops → warning + no stop recorded) absorb it
    as a no-submit outcome and the run continues.

    Deliberately an exception, NOT a no-submit result dict: the runner's
    ``broker_order_execution`` classifies any status outside its terminal
    reject set as PENDING, so a returned ``rejected_*`` dict would be
    recorded as an open order that never existed.
    """

    def __init__(
        self,
        symbol: str,
        quantity: Any,
        reason: str,
        *,
        status: str = INVALID_FRACTIONAL_ORDER_STATUS,
    ) -> None:
        self.symbol = symbol
        self.quantity = quantity
        self.reason = reason
        self.status = status
        super().__init__(
            f"fractional order refused (no submit) for {symbol} "
            f"qty={quantity!r}: {reason} [status={status}]"
        )



# ── Buy-sizing buying-power mode (RenQuant fix/size-on-settled-cash, 2026-08-30)
#
# What number the runner may SIZE NEW BUYS against. Vocabulary shared with the
# sim path (``adapters/sim_order_helpers.py`` / ``adapters/lean_account.py``
# read the same ``execution.buying_power_mode`` key with the same
# ``settled_cash`` / ``non_marginable_buying_power`` canonical names and the
# same aliases), so ONE config key drives sim and live alike.
#
#   settled_cash                 Alpaca ``account.cash`` — settled funds only.
#                                DEFAULT when the key is absent. Never spends
#                                T+1 pending sell proceeds, never margin.
#   non_marginable_buying_power  Alpaca ``account.non_marginable_buying_power``
#                                — cash + unsettled sell proceeds, no 2x/4x
#                                margin. On a MARGIN account this lets a buy
#                                clear against proceeds that have not settled,
#                                which the broker fronts as a margin debit
#                                until settlement (the 08-27/08-28 HPE / WELL
#                                buys against settled cash of $33 / −$1,140).
#   buying_power                 Alpaca ``account.buying_power`` — the 2x/4x
#                                margin figure. Explicit opt-in only; the sim
#                                has no margin model (its normalisers reject
#                                the value), so setting it breaks sim/live
#                                parity by construction. Logged as a WARNING
#                                on every read.
#
# Sizing cash is FLOORED AT ZERO: a negative settled balance (the account is
# on margin) sizes to $0 and the run records ``no_settled_cash`` — no buys.
# Exits are never sized here and are never blocked by this.
BUYING_POWER_MODE_SETTLED = "settled_cash"
BUYING_POWER_MODE_NMBP = "non_marginable_buying_power"
BUYING_POWER_MODE_MARGIN = "buying_power"
DEFAULT_BUYING_POWER_MODE = BUYING_POWER_MODE_SETTLED
BUYING_POWER_MODES: tuple[str, ...] = (
    BUYING_POWER_MODE_SETTLED,
    BUYING_POWER_MODE_NMBP,
    BUYING_POWER_MODE_MARGIN,
)
BUYING_POWER_MODE_ALIASES: dict[str, str] = {
    BUYING_POWER_MODE_SETTLED: BUYING_POWER_MODE_SETTLED,
    "settled": BUYING_POWER_MODE_SETTLED,
    "cash": BUYING_POWER_MODE_SETTLED,
    BUYING_POWER_MODE_NMBP: BUYING_POWER_MODE_NMBP,
    "cash_plus_unsettled": BUYING_POWER_MODE_NMBP,
    "unsettled": BUYING_POWER_MODE_NMBP,
    BUYING_POWER_MODE_MARGIN: BUYING_POWER_MODE_MARGIN,
    "margin": BUYING_POWER_MODE_MARGIN,
}
#: ``sizing_reason`` when the mode's balance is <= 0 (or unreadable) and the
#: buy budget is therefore $0 for the bar.
NO_SETTLED_CASH_REASON = "no_settled_cash"
NO_BUYING_POWER_REASON = "no_buying_power"
UNREADABLE_CASH_REASON = "cash_unreadable"


def normalize_buying_power_mode(raw: Any) -> str:
    """Canonical ``execution.buying_power_mode``; absent/empty → settled cash.

    Raises ``ValueError`` on an unrecognised value so a typo in a deployed
    config FAIL-CLOSES (the runner's cash read wraps this and sizes to $0)
    instead of silently picking a mode.
    """
    if raw is None:
        return DEFAULT_BUYING_POWER_MODE
    mode = str(raw).strip().lower()
    if not mode:
        return DEFAULT_BUYING_POWER_MODE
    if mode not in BUYING_POWER_MODE_ALIASES:
        raise ValueError(
            "execution.buying_power_mode must be one of "
            f"{sorted(BUYING_POWER_MODE_ALIASES)}; got {raw!r}"
        )
    return BUYING_POWER_MODE_ALIASES[mode]


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def resolve_sizing_cash(
    mode: str,
    *,
    settled_cash: Any,
    non_marginable_buying_power: Any = None,
    buying_power: Any = None,
) -> dict[str, Any]:
    """Pure: pick the balance ``mode`` names, floor it at zero, name the reason.

    Returns the buying-power snapshot dict the runner logs and records::

        mode, settled_cash, non_marginable_buying_power, buying_power,
        sizing_cash, sizing_source, sizing_reason

    ``sizing_source`` says WHICH field produced ``sizing_cash``. When the
    requested field is unavailable the function degrades CONSERVATIVELY —
    to settled cash (never to a larger figure) — and says so in the source.
    ``sizing_reason`` is ``None`` when there is a positive budget, otherwise
    ``no_settled_cash`` (settled mode), ``no_buying_power`` (the other
    modes) or ``cash_unreadable`` (nothing parseable at all).
    """
    mode = normalize_buying_power_mode(mode)
    settled = _finite_or_none(settled_cash)
    nmbp = _finite_or_none(non_marginable_buying_power)
    margin_bp = _finite_or_none(buying_power)

    if mode == BUYING_POWER_MODE_SETTLED:
        chosen, source = settled, "account.cash"
    elif mode == BUYING_POWER_MODE_NMBP:
        if nmbp is not None:
            chosen, source = nmbp, "account.non_marginable_buying_power"
        else:
            chosen, source = settled, "account.cash (non_marginable_buying_power unavailable)"
    else:  # BUYING_POWER_MODE_MARGIN
        if margin_bp is not None:
            chosen, source = margin_bp, "account.buying_power"
        elif nmbp is not None:
            chosen, source = nmbp, "account.non_marginable_buying_power (buying_power unavailable)"
        else:
            chosen, source = settled, "account.cash (buying_power unavailable)"

    if chosen is None:
        sizing, reason = 0.0, UNREADABLE_CASH_REASON
    elif chosen <= 0.0:
        sizing = 0.0
        reason = (NO_SETTLED_CASH_REASON if mode == BUYING_POWER_MODE_SETTLED
                  else NO_BUYING_POWER_REASON)
    else:
        sizing, reason = chosen, None

    return {
        "mode": mode,
        "settled_cash": settled,
        "non_marginable_buying_power": nmbp,
        "buying_power": margin_bp,
        "sizing_cash": float(sizing),
        "sizing_source": source,
        "sizing_reason": reason,
    }


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

    def get_buying_power_snapshot(self, mode: str | None = None) -> dict[str, Any]:
        """Balances the runner may size NEW BUYS against, per ``mode``.

        See :func:`resolve_sizing_cash` for the dict contract. The base
        implementation knows only ``get_cash()`` and treats it as SETTLED
        cash — the conservative reading for a broker that exposes no
        unsettled / margin figures (PaperBroker, IBKR). Brokers that do
        expose them (AlpacaBroker) override this to read the account once
        and fill in every field. ``mode`` absent → settled cash.
        """
        return resolve_sizing_cash(
            normalize_buying_power_mode(mode),
            settled_cash=self.get_cash(),
        )

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
