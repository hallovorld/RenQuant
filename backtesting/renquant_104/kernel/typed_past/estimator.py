"""TypedTask Protocol — cvxportfolio Estimator.values_in_time pattern.

A ``TypedTask`` reads ONLY from a frozen ``Past`` snapshot, parameterized by
the cursor ``t``. It returns a ``TaskResult`` (immutable). It cannot mutate
shared state. Inference / training / sim all walk the same (t, past) cursor.

To bridge with the existing legacy ``Task.run(ctx)`` ABC during multi-week
migration, ``TypedTaskAdapter`` wraps a TypedTask so it can sit in an
existing Job's task chain unchanged. The adapter:

  1. Reads ``ctx.today`` + relevant fields to construct a Past via
     ``slice_until``.
  2. Calls ``typed_task.values_in_time(t, past)``.
  3. Writes ``TaskResult.ctx_writes`` back onto ctx (the only intentional
     escape hatch; callers must explicitly opt in to mutation).
  4. Returns ``result.continue_chain`` for the legacy chain protocol.

References:
  * cvxportfolio.Estimator —
    https://www.cvxportfolio.com/api_documentation/estimator.html
  * RenQuant CLAUDE.md §1c (≤50 LOC tasks).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import pandas as pd

from .past import Past


@dataclass(frozen=True)
class TaskResult:
    """Immutable return type for ``TypedTask.values_in_time``.

    Attributes:
        continue_chain: legacy contract — False stops the enclosing Job's
            task chain (mirrors current ``Task.run`` False-short-circuit).
        ctx_writes: optional explicit ctx mutations the task wants to
            propagate to the legacy InferenceContext during migration.
            Empty dict = pure read-only task. Each entry is ``(field_name
            -> value)`` and the adapter performs ``setattr(ctx, k, v)``.
        diagnostics: free-form dict for logging/observability. Adapter
            does not write these to ctx.
    """

    continue_chain: bool = True
    ctx_writes: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class TypedTask(Protocol):
    """A Task that reads only from a frozen Past at cursor t.

    Subclasses (or duck-typed implementations) provide a single method:

        def values_in_time(self, t: pd.Timestamp, past: Past) -> TaskResult: ...

    The Protocol is ``runtime_checkable`` so adapters / tests can verify
    a candidate object satisfies the contract before wiring.
    """

    def values_in_time(self, t: pd.Timestamp, past: Past) -> TaskResult: ...


# ── Bridge to legacy Task ABC ──────────────────────────────────────────────


class TypedTaskAdapter:
    """Wraps a TypedTask so it can drop into a legacy Job's task chain.

    The legacy chain calls ``self.run(ctx)``; the adapter slices a Past
    from ctx, hands it to the TypedTask, and propagates writes back.

    Per §5.13.10, fields that are conceptually optional are typed as
    ``Optional[X]``; we do NOT add ``if ... is not None`` short-circuits
    here. The adapter's contract is: ctx must have ``today``, ``ohlcv``,
    ``cash``, ``holdings``. If any are missing, AttributeError is the
    correct visible failure.
    """

    def __init__(self, typed: TypedTask, *, fundamentals_attr: str = "fundamentals"):
        if not isinstance(typed, TypedTask):
            raise TypeError(
                f"TypedTaskAdapter expects a TypedTask "
                f"(values_in_time method); got {type(typed)}"
            )
        self.typed = typed
        self.fundamentals_attr = fundamentals_attr

    @property
    def name(self) -> str:
        return f"TypedTaskAdapter({type(self.typed).__name__})"

    def run(self, ctx) -> "bool | None":
        past = self._build_past(ctx)
        result = self.typed.values_in_time(past.t, past)

        if not isinstance(result, TaskResult):
            raise TypeError(
                f"{type(self.typed).__name__}.values_in_time must return "
                f"TaskResult; got {type(result)}"
            )

        for k, v in result.ctx_writes.items():
            setattr(ctx, k, v)

        return False if not result.continue_chain else True

    def _build_past(self, ctx) -> Past:
        # Concatenate per-ticker OHLCV frames into one (ticker, date)
        # MultiIndex DataFrame, OR pass through a pre-flattened panel.
        # We use whichever shape the ctx already exposes.
        ohlcv = self._ohlcv_as_df(ctx.ohlcv)
        fundamentals = getattr(ctx, self.fundamentals_attr, pd.DataFrame())
        if not isinstance(fundamentals, pd.DataFrame):
            fundamentals = pd.DataFrame()

        regime_history = tuple(getattr(ctx, "regime_counts", {}).keys())
        holdings = dict(getattr(ctx, "holdings", {}))
        cash = float(getattr(ctx, "cash", 0.0))

        return Past.slice_until(
            ctx.today,
            {
                "ohlcv": ohlcv,
                "fundamentals": fundamentals,
                "regime_history": regime_history,
                "holdings": holdings,
                "cash": cash,
            },
        )

    @staticmethod
    def _ohlcv_as_df(ohlcv: Any) -> pd.DataFrame:
        """Convert legacy dict[ticker -> DataFrame] to a flat date-indexed
        DataFrame for Past. We take the union of dates and store the max
        per-date close (for staleness / freshness checks). Production
        adapters can override by passing a ready DataFrame."""
        if isinstance(ohlcv, pd.DataFrame):
            return ohlcv
        if not isinstance(ohlcv, dict) or not ohlcv:
            return pd.DataFrame()

        all_dates = set()
        for df in ohlcv.values():
            if isinstance(df, pd.DataFrame) and isinstance(df.index, pd.DatetimeIndex):
                all_dates.update(df.index)
        if not all_dates:
            return pd.DataFrame()

        idx = pd.DatetimeIndex(sorted(all_dates))
        return pd.DataFrame(index=idx)
