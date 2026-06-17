"""Typed live-state model (LiveStateV2) — umbrella kernel MIRROR.

s1-wire-live-state-v2 (roadmap). The orchestrator package
(`renquant_orchestrator.live_state_v2`) owns the canonical typed model and
its one-place v1->v2 `parse` / lossless `to_v1_dict` inverse. The umbrella
cannot import that package across the subrepo boundary (only
`renquant_common` is on the umbrella path; see
`adapters/state_store.py`), so a MINIMAL, behaviour-equivalent copy lives
here instead — exactly as the cross-repo policy prescribes.

Purpose
-------
Let the runner read/write live state through a typed model BEHIND A FLAG, so
adding a per-holding field (e.g. ``protection_breaches``) becomes a one-line
schema change instead of touching every read/write site. When the flag is
OFF the runner keeps using plain dicts and the on-disk JSON is unchanged.

LOSSLESS contract
-----------------
``LiveStateV2.parse(v1_dict).to_v1_dict()`` is the byte-for-byte inverse of
the v1 flat dict:

  * top-level key ORDER is preserved (matters: the on-disk JSON is emitted
    by ``json.dumps(state, indent=2)`` which is order-sensitive),
  * every value is passed through unchanged, including keys this schema does
    not yet model (``stop_orders``, ``recent_sell_orders``, ``monitor_state``,
    ``regime_state``, ``entry_signals`` …) — they ride along verbatim,
  * the per-holding "columns" (``entry_dates``, ``sell_streaks``,
    ``protection_breaches``, ``position_hwm``) are ALSO surfaced as a typed
    per-ticker ``HoldingV2`` view for ergonomic, one-line-extensible access,
    WITHOUT changing how they serialise.

So the flag-ON write path is ``parse(...).to_v1_dict()`` -> identical bytes;
the typed view is a read/edit convenience layered on top of the lossless
container, never a lossy re-projection.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Per-holding "column" maps: ticker -> value. Each is a flat v1 dict that the
# runner writes today. To add a new per-holding field, append ONE name here
# and ONE attribute on HoldingV2 below — nothing else changes.
_HOLDING_COLUMNS: tuple[str, ...] = (
    "entry_dates",          # ticker -> entry date (ISO str)
    "sell_streaks",         # ticker -> consecutive sell-signal day count
    "protection_breaches",  # ticker -> consecutive model-protection breaches
    "position_hwm",         # ticker -> per-position high-water mark
)


class HoldingV2(BaseModel):
    """Typed per-ticker view assembled from the v1 column maps.

    Adding a per-holding field = one attribute here + one name in
    ``_HOLDING_COLUMNS``. The lossless container does the rest.
    """

    model_config = ConfigDict(extra="forbid")

    entry_date: str | None = None
    sell_streak: int = 0
    protection_breaches: int = 0
    position_hwm: float | None = None


class LiveStateV2(BaseModel):
    """Lossless typed container over the v1 flat live-state dict.

    ``raw`` is the source of truth for serialisation — it preserves the
    exact top-level key order and every value verbatim. ``holdings`` is a
    derived, typed convenience view over the per-holding column maps.
    """

    model_config = ConfigDict(extra="forbid")

    # The verbatim v1 dict (ordered). NEVER reordered or coerced — this is
    # what guarantees the byte-for-byte round-trip.
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def parse(cls, v1: dict[str, Any] | None) -> "LiveStateV2":
        """v1 flat dict -> typed v2. Captures the dict verbatim (ordered)."""
        # dict() preserves insertion order (Python 3.7+); a shallow copy is
        # enough — we never mutate nested values, only pass them through.
        return cls(raw=dict(v1 or {}))

    @property
    def holdings(self) -> dict[str, HoldingV2]:
        """Typed per-ticker view assembled from the v1 column maps.

        Read-only projection: tickers are the union of all column maps so a
        holding that appears in any map is surfaced. Editing live state still
        goes through ``to_v1_dict`` / the column maps to stay lossless.
        """
        cols = {c: (self.raw.get(c) or {}) for c in _HOLDING_COLUMNS}
        tickers: list[str] = []
        seen: set[str] = set()
        for c in _HOLDING_COLUMNS:
            for t in cols[c]:
                if t not in seen:
                    seen.add(t)
                    tickers.append(t)
        out: dict[str, HoldingV2] = {}
        for t in tickers:
            out[t] = HoldingV2(
                entry_date=cols["entry_dates"].get(t),
                sell_streak=int(cols["sell_streaks"].get(t, 0) or 0),
                protection_breaches=int(cols["protection_breaches"].get(t, 0) or 0),
                position_hwm=cols["position_hwm"].get(t),
            )
        return out

    def to_v1_dict(self) -> dict[str, Any]:
        """Typed v2 -> v1 flat dict. The LOSSLESS inverse of ``parse``.

        Returns the verbatim v1 dict in its original key order. ``json.dumps``
        of this is byte-identical to ``json.dumps`` of the dict ``parse`` was
        given.
        """
        return dict(self.raw)
