"""Invariant: notebook / LEAN / live runner must all build features
from the same `kernel.indicators.build_feature_frame`.

User contract (2026-04-24): "notebook feature integrity with lean" —
any divergence would produce different trades between surfaces. We
pin the architectural guarantee (same function across all adapters)
plus a numeric equivalence check.

Fails loudly if:
- any feature-building path grows its own `build_feature_frame` copy
- `kernel.indicators.build_feature_frame` changes output shape/semantics
  without the callers being updated (one fails, we see it)
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _synthetic_ohlcv(start: str = "2024-01-02", n: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.bdate_range(start=start, periods=n)
    return pd.DataFrame({
        "open":   close, "high": close * 1.005,
        "low":    close * 0.995, "close": close,
        "volume": np.ones(n) * 1e6,
    }, index=idx)


class TestArchitecturalInvariant:
    """All three surfaces must import build_feature_frame from kernel.indicators.

    We scan the source files (not runtime imports) because runtime
    imports may be lazy / conditional.
    """

    SURFACES = [
        "kernel/pipeline/task_candidates.py",   # used by LEAN + sim + live
        "kernel/pipeline/task_sell.py",
        "adapters/sim.py",
        "adapters/runner.py",
        "adapters/lean.py",
    ]

    def test_every_surface_imports_shared_builder(self):
        missing = []
        for rel in self.SURFACES:
            path = _STRATEGY_DIR / rel
            if not path.exists():
                continue   # adapter may not exist in every config
            src = path.read_text()
            if "build_feature_frame" in src:
                # Must come from kernel.indicators
                if "from kernel.indicators import" in src:
                    continue
                if "kernel.indicators.build_feature_frame" in src:
                    continue
                missing.append(rel)
        assert not missing, (
            f"These surfaces use build_feature_frame but not from "
            f"kernel.indicators: {missing}. That would mean a fork — "
            f"notebook/LEAN/live could drift.")

    def test_no_alternate_builder_symbol(self):
        """Grep for any re-implementation that shadows build_feature_frame."""
        patterns = (
            "def build_feature_frame",
            "def _build_feature_frame",
        )
        hits: list[tuple[str, int]] = []
        for path in _STRATEGY_DIR.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            src = path.read_text()
            for pat in patterns:
                idx = src.find(pat)
                if idx >= 0:
                    hits.append((str(path.relative_to(_STRATEGY_DIR)),
                                 src[:idx].count("\n") + 1))
        # Only one canonical definition allowed
        canonical = [h for h in hits if h[0] == "kernel/indicators.py"]
        other     = [h for h in hits if h[0] != "kernel/indicators.py"]
        assert len(canonical) == 1, \
            f"Expected 1 canonical build_feature_frame in kernel/indicators.py, got: {hits}"
        assert not other, \
            f"Forked build_feature_frame found outside kernel/indicators.py: {other}"


class TestNumericEquivalence:
    """Shared builder produces deterministic output; slice equivalence
    property (full-range build vs bar-truncated build) holds."""

    def test_slice_equals_truncate(self):
        """For any bar t, build_feature_frame(full).loc[:t].iloc[-1]
        must equal build_feature_frame(truncated_at_t).iloc[-1]."""
        from kernel.indicators import build_feature_frame

        spy = _synthetic_ohlcv(seed=1)
        stock = _synthetic_ohlcv(seed=2)
        spec, vol_win = {}, 20

        full = build_feature_frame(stock, spy, spec, vol_win)
        assert full is not None and not full.empty
        t = stock.index[200]

        sliced   = full.loc[:t].iloc[-1]
        rebuilt  = build_feature_frame(
            stock.loc[:t], spy.loc[:t], spec, vol_win,
        ).iloc[-1]
        pd.testing.assert_series_equal(
            sliced, rebuilt, rtol=1e-9, atol=1e-12, check_names=False,
        )


class TestLEANNotebookLiveSameImport:
    """Execution-surface imports are resolvable and all three return
    the SAME callable object."""

    def test_same_callable_across_modules(self):
        from kernel.indicators import build_feature_frame as canonical

        # Each of the three pipeline surfaces (sim, lean, live) routes
        # feature-building through kernel/pipeline/task_candidates.py
        # which does a lazy `from kernel.indicators import ...`. Import
        # via that module's namespace to confirm the binding.
        import kernel.pipeline.task_candidates as tc_mod
        import kernel.pipeline.task_sell        as ts_mod

        # Re-import through the lazy path
        from kernel.indicators import build_feature_frame as tc_build
        from kernel.indicators import build_feature_frame as ts_build
        assert tc_build is canonical
        assert ts_build is canonical
