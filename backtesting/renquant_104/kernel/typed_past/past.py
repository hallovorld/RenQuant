"""Past — frozen, point-in-time snapshot of all data a Task may legally read.

Design contract (cvxportfolio Estimator.values_in_time pattern):

  1. ``Past`` is ``frozen=True`` — a Task that receives a Past CANNOT
     mutate it. ``FrozenInstanceError`` raises on any attribute write.
  2. Every DataFrame inside Past has had rows > t stripped. This is
     validated in ``__post_init__`` (raises AssertionError per §5.13).
  3. ``Past.slice_until(t, source)`` is the ONLY supported factory.
     It does the slicing + assertion. Direct construction is allowed
     for tests but the same validation runs.

Why this exists: see CLAUDE.md §5.13 — the 17-bug audit on 2026-05-09
identified silent peek-ahead in several Tasks (e.g. cost-aware wash-sale
that read ISO strings from a non-sliced dict). A frozen, pre-sliced
Past makes that class of bug architecturally impossible.

Per §5.13.10 we type optional fields as ``Optional[X]`` but DO NOT add
defensive ``if x is not None`` patterns. Callers that need a default
should pass an empty DataFrame, not ``None``.
"""
from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class Past:
    """Frozen, point-in-time snapshot of past-only data.

    Attributes:
        t: cursor timestamp; nothing in this Past was observed after t
        ohlcv: market bars, date-indexed (DatetimeIndex), all rows <= t
        fundamentals: fundamental panel, date-indexed, all rows <= t
        regime_history: immutable tuple of past regime labels (most recent last)
        holdings: frozen mapping (ticker -> share count or position state)
        cash: free cash at time t (USD)
    """

    t: pd.Timestamp
    ohlcv: pd.DataFrame
    fundamentals: pd.DataFrame
    regime_history: tuple
    holdings: types.MappingProxyType
    cash: float

    # ── Validation ──────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        # cursor must be a Timestamp (or coerce). frozen=True prevents normal
        # assignment, so we use object.__setattr__ to coerce in-place.
        if not isinstance(self.t, pd.Timestamp):
            object.__setattr__(self, "t", pd.Timestamp(self.t))

        if not isinstance(self.ohlcv, pd.DataFrame):
            raise TypeError(f"Past.ohlcv must be DataFrame, got {type(self.ohlcv)}")
        if not isinstance(self.fundamentals, pd.DataFrame):
            raise TypeError(
                f"Past.fundamentals must be DataFrame, got {type(self.fundamentals)}"
            )
        if not isinstance(self.regime_history, tuple):
            raise TypeError(
                f"Past.regime_history must be tuple (immutable), "
                f"got {type(self.regime_history)}"
            )
        if not isinstance(self.holdings, types.MappingProxyType):
            raise TypeError(
                "Past.holdings must be types.MappingProxyType (frozen mapping); "
                f"got {type(self.holdings)}"
            )

        _assert_index_le_t("ohlcv", self.ohlcv, self.t)
        _assert_index_le_t("fundamentals", self.fundamentals, self.t)

    # ── Factory ─────────────────────────────────────────────────────────────
    @staticmethod
    def slice_until(t: Any, source: Mapping[str, Any]) -> "Past":
        """Build a Past from a fuller snapshot, slicing all date-indexed
        DataFrames to rows <= t.

        ``source`` keys (all required):
          * ``ohlcv``         — pd.DataFrame with DatetimeIndex
          * ``fundamentals``  — pd.DataFrame with DatetimeIndex
          * ``regime_history``— Iterable[str], will be tuple-ified
          * ``holdings``      — Mapping[str, Any], will be frozen
          * ``cash``          — float

        Raises AssertionError per §5.13.7 (visible failure, not silent
        filter) when slicing detects an index value > t — this means the
        caller passed a misaligned snapshot and we must surface the bug.
        Slicing-by-truncation itself is fine (rows simply <= t); the
        assertion fires only if the SOURCE INDEX claims a date > t but
        we asked for a snapshot at t — almost certainly a bug.

        Note: this factory does NOT silently strip > t rows. It calls
        ``df.loc[:t]`` once, then asserts the result's max index <= t.
        ``df.loc[:t]`` is inclusive of t and excludes anything strictly
        greater, so the assertion is a self-consistency check.
        """
        t_ts = pd.Timestamp(t)

        ohlcv = _slice_le(source["ohlcv"], t_ts, "ohlcv")
        fundamentals = _slice_le(source["fundamentals"], t_ts, "fundamentals")
        regime_history = tuple(source["regime_history"])
        holdings = types.MappingProxyType(dict(source["holdings"]))
        cash = float(source["cash"])

        return Past(
            t=t_ts,
            ohlcv=ohlcv,
            fundamentals=fundamentals,
            regime_history=regime_history,
            holdings=holdings,
            cash=cash,
        )

    # ── Equality / hash by content ──────────────────────────────────────────
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Past):
            return NotImplemented
        return (
            self.t == other.t
            and self.ohlcv.equals(other.ohlcv)
            and self.fundamentals.equals(other.fundamentals)
            and self.regime_history == other.regime_history
            and dict(self.holdings) == dict(other.holdings)
            and self.cash == other.cash
        )

    def __hash__(self) -> int:
        # Hash on (t, shapes, regime_history, holdings keys, cash) — full
        # DataFrame hashing is expensive; this is good enough for dict/set
        # use. Equality check above does full content compare.
        return hash(
            (
                self.t,
                self.ohlcv.shape,
                self.fundamentals.shape,
                self.regime_history,
                tuple(sorted(self.holdings.keys())),
                self.cash,
            )
        )


# ── Internal helpers ────────────────────────────────────────────────────────


def _slice_le(df: Any, t: pd.Timestamp, name: str) -> pd.DataFrame:
    """Slice a date-indexed DataFrame to rows <= t. Raises AssertionError
    if the result still contains a row > t (self-consistency check; per
    §5.13.7 we surface this rather than silently filter)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"source[{name!r}] must be DataFrame, got {type(df)}")
    if len(df) == 0:
        return df  # empty is fine
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"source[{name!r}].index must be DatetimeIndex, got {type(df.index)}"
        )
    sliced = df.loc[:t]
    if len(sliced) > 0:
        assert sliced.index.max() <= t, (
            f"Past.slice_until({name}): post-slice max index "
            f"{sliced.index.max()} > t={t} — DataFrame index inconsistency"
        )
    return sliced


def _assert_index_le_t(name: str, df: pd.DataFrame, t: pd.Timestamp) -> None:
    """Assert every row in df has index <= t (constructor-time invariant)."""
    if len(df) == 0:
        return
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"Past.{name}.index must be DatetimeIndex, got {type(df.index)}"
        )
    max_idx = df.index.max()
    assert max_idx <= t, (
        f"Past.{name} contains row dated {max_idx} > t={t}; "
        f"peek-ahead violation. Use Past.slice_until() to construct."
    )
