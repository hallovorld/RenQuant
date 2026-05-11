"""Immutable execution-layer message types.

Two value types travel between :class:`kernel.pipeline.pp_execution.ExecutionPipeline`
and concrete :class:`kernel.execution.backend.ExecutionBackend` impls:

* :class:`OrderIntent` — what the pipeline *asks* the backend to do.
* :class:`Fill` — what the backend *reports back* after execution.

Both are frozen ``@dataclass`` instances with field-level finite guards
(CLAUDE.md §5.13.11). Construction is the only place we trust user-
provided floats; once instantiated, callers may rely on every numeric
field being finite and within its declared sign domain.

Per §5.13.5 the **only** place exit-type strings are tracked is on
:class:`OrderIntent.exit_type` — adapters must not maintain parallel
exit counters keyed off ad-hoc strings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class OrderIntent:
    """A pipeline-level execution request.

    Fields:
        ticker: equity symbol (case-sensitive, e.g. ``"AAPL"``).
        side: :class:`OrderSide.BUY` or :class:`OrderSide.SELL`.
        shares: positive int for partial sells / explicit buy quantities.
            ``None`` is reserved for full-liquidate SELLs and means
            "close the entire current position" — the backend resolves
            it to a concrete share count at fill time.
        target_pct: target portfolio weight after fill. For BUY this
            sizes the order; for SELL it's informational (always 0
            on full liquidate) and may be 0 on partial trims when the
            pipeline didn't compute a residual target.
        today: bar timestamp on which the order is placed (NYSE
            calendar day for daily strategies, UTC bar for intraday).
        reason: human-readable string for logs / postmortems.
            Persisted as-is into ``Fill.reason`` mirroring.
        exit_type: categorical tag from
            ``{"stop_loss", "trailing_stop", "single_day_loss",
              "max_hold", "model_sell", "rotation", "qp_sell",
              "qp_close"}`` for SELL intents; ``None`` for BUY.
    """

    ticker: str
    side: OrderSide
    shares: Optional[int]
    target_pct: float
    today: pd.Timestamp
    reason: str
    exit_type: Optional[str]

    def __post_init__(self) -> None:
        # target_pct must be finite (§5.13.11) — even SELL paths that
        # don't use it depend on downstream `>` comparisons not seeing
        # NaN (NaN > 0 silently False).
        if not math.isfinite(self.target_pct):
            raise ValueError(
                f"OrderIntent.target_pct must be finite, got {self.target_pct!r}"
            )

        if self.side == OrderSide.BUY:
            # Buys MUST be explicit and positive — None or zero is a
            # pipeline bug (e.g. SizeAndEmitTask emitting on rejected order).
            if self.shares is None or self.shares <= 0:
                raise ValueError(
                    f"BUY OrderIntent.shares must be a positive int, "
                    f"got {self.shares!r}"
                )
            if self.target_pct <= 0:
                raise ValueError(
                    f"BUY OrderIntent.target_pct must be > 0, "
                    f"got {self.target_pct!r}"
                )
            if self.exit_type is not None:
                raise ValueError(
                    f"BUY OrderIntent.exit_type must be None, "
                    f"got {self.exit_type!r}"
                )
        elif self.side == OrderSide.SELL:
            # SELL: shares=None → full liquidate; positive int → partial.
            # Zero / negative is a bug.
            if self.shares is not None and self.shares <= 0:
                raise ValueError(
                    f"partial SELL OrderIntent.shares must be > 0 or None, "
                    f"got {self.shares!r}"
                )
        else:  # pragma: no cover — enum exhaustiveness
            raise ValueError(f"unknown OrderSide {self.side!r}")

        if not self.ticker:
            raise ValueError("OrderIntent.ticker must be non-empty")
        if not self.reason:
            raise ValueError("OrderIntent.reason must be non-empty")

    @property
    def is_full_liquidate(self) -> bool:
        return self.side == OrderSide.SELL and self.shares is None


@dataclass(frozen=True)
class Fill:
    """A confirmed execution report from an :class:`ExecutionBackend`.

    All numeric fields are finite, ``shares > 0``, ``price > 0``, and
    ``fees >= 0`` by construction. Cash accounting downstream MAY trust
    these invariants without re-guarding.
    """

    ticker: str
    side: OrderSide
    shares: int
    price: float
    fees: float
    today: pd.Timestamp

    def __post_init__(self) -> None:
        if not isinstance(self.shares, int) or self.shares <= 0:
            raise ValueError(
                f"Fill.shares must be a positive int, got {self.shares!r}"
            )
        if not math.isfinite(self.price) or self.price <= 0:
            raise ValueError(
                f"Fill.price must be finite and positive, got {self.price!r}"
            )
        if not math.isfinite(self.fees) or self.fees < 0:
            raise ValueError(
                f"Fill.fees must be finite and non-negative, got {self.fees!r}"
            )
        if not self.ticker:
            raise ValueError("Fill.ticker must be non-empty")

    @property
    def notional(self) -> float:
        """Gross trade value (positive for both sides)."""
        return self.shares * self.price


__all__ = ["OrderSide", "OrderIntent", "Fill"]
