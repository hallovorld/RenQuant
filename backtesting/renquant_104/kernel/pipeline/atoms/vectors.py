"""Vector-building atoms — build per-asset numpy arrays from ctx
collections like ctx.holdings / ctx.candidates.

These were the most repeated patterns in the QP / rotation / sell-aggregation
monoliths. Centralizing them ensures NaN handling is uniform.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..pipeline import Task
from .ctx_ops import _get_path, _set_path


class BuildVectorFromMappingTask(Task):
    """Build an n-vector indexed by a tickers-list, pulling `attr` from
    each entry of a ctx mapping (`source` field).

    Designed for both ctx.holdings (dict[ticker → HoldingState]) and
    a candidates-by-ticker dict. Caller passes `tickers_field` as the
    canonical order. Missing tickers → `default`. Non-finite/None → `default`.
    """

    def __init__(
        self,
        tickers_field: str,
        source_field: str,
        attr: str,
        target: str,
        default: float = 0.0,
        fallback_attr: str | None = None,
    ):
        self.tickers_field = tickers_field
        self.source_field = source_field
        self.attr = attr
        self.target = target
        self.default = default
        self.fallback_attr = fallback_attr

    @property
    def name(self) -> str:
        return f"BuildVector({self.attr}→{self.target})"

    def run(self, ctx) -> bool | None:
        tickers = _get_path(ctx, self.tickers_field)
        source = _get_path(ctx, self.source_field) or {}
        if tickers is None:
            return False
        n = len(tickers)
        out = np.full(n, float(self.default))
        for i, t in enumerate(tickers):
            obj = source.get(t) if isinstance(source, dict) else None
            if obj is None:
                continue
            v = getattr(obj, self.attr, None)
            if v is None and self.fallback_attr is not None:
                v = getattr(obj, self.fallback_attr, None)
            try:
                f = float(v) if v is not None else self.default
                if math.isfinite(f):
                    out[i] = f
            except (TypeError, ValueError):
                pass
        _set_path(ctx, self.target, out)


class BuildMaskFromConditionTask(Task):
    """Build an n-bool mask. Caller passes a predicate fn(ctx, ticker) -> bool.

    Used for wash_sale_mask, post-stop blackout, defensive_set membership, etc.
    """

    def __init__(
        self,
        tickers_field: str,
        target: str,
        predicate,                       # Callable[[ctx, str], bool]
    ):
        self.tickers_field = tickers_field
        self.target = target
        self.predicate = predicate

    @property
    def name(self) -> str:
        return f"BuildMask({self.target})"

    def run(self, ctx) -> bool | None:
        tickers = _get_path(ctx, self.tickers_field) or []
        mask = np.zeros(len(tickers), dtype=bool)
        for i, t in enumerate(tickers):
            try:
                mask[i] = bool(self.predicate(ctx, t))
            except Exception:
                mask[i] = False
        _set_path(ctx, self.target, mask)


class StableTickerOrderTask(Task):
    """Build canonical tickers list = `held_field` + (cands_field − held_field).

    Used by every vector-building Job to fix the order in which subsequent
    BuildVector tasks index tickers.
    """

    def __init__(
        self,
        held_field: str,
        cands_field: str,
        target: str,
    ):
        self.held_field = held_field
        self.cands_field = cands_field
        self.target = target

    @property
    def name(self) -> str:
        return f"StableTickerOrder→{self.target}"

    def run(self, ctx) -> bool | None:
        held = _get_path(ctx, self.held_field) or {}
        cands = _get_path(ctx, self.cands_field) or []
        held_tickers = list(held.keys()) if isinstance(held, dict) else list(held)
        cand_tickers = []
        for c in cands:
            t = getattr(c, "ticker", None) if not isinstance(c, str) else c
            if t and t not in held_tickers and t not in cand_tickers:
                cand_tickers.append(t)
        _set_path(ctx, self.target, held_tickers + cand_tickers)


__all__ = [
    "BuildVectorFromMappingTask",
    "BuildMaskFromConditionTask",
    "StableTickerOrderTask",
]
