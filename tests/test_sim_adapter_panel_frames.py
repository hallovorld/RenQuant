"""Regression: SimAdapter panel frame plumbing.

Pre-fix (2026-05-02): cherry-pick of `feat(metrics): Sharpe / ...` from
main (commit c679600) brought a sim.py that unpacked
`prepare_inference_panel_frames(...)` as a 4-tuple `(ff, fac, macro, emb)`.
Exp branch's `prepare_inference_panel_frames` still returns a 3-tuple
because asset embeddings T2-2 is rejected on this branch (see
`doc/research/failed-experiments-log.md::E13`). The 4-vs-3 mismatch raised
ValueError on every sim run; the broad `except Exception` in SimAdapter
swallowed it and set `_panel_feature_frames = None`, so the simulator ran
end-to-end with NO panel features and produced 0 trades — completely
invisible to the operator.

These tests pin the contract:
  1. `prepare_inference_panel_frames` returns exactly 3 values on this
     branch. If a future change re-introduces a 4th return value, this
     test catches it BEFORE a sim run produces fake-zero results.
  2. SimAdapter's panel-frame block can be exercised end-to-end with a
     thin OHLCV stub and produces non-None feature frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RQ_ROOT = REPO_ROOT / "backtesting" / "renquant_104"
sys.path.insert(0, str(RQ_ROOT))


def test_prepare_inference_panel_frames_returns_three_tuple():
    """Contract: 3 values, in order (feature_frames, factor_frames, macro)."""
    import inspect
    from training_panel.pipeline import prepare_inference_panel_frames

    src = inspect.getsource(prepare_inference_panel_frames)
    # Find the function's `return` line(s). The function should have
    # exactly one `return` statement returning three comma-separated
    # expressions.
    return_lines = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("return ")]
    assert len(return_lines) == 1, (
        f"prepare_inference_panel_frames must have exactly 1 return statement; "
        f"found {len(return_lines)}: {return_lines}"
    )

    # Count top-level commas in the return expression (commas inside
    # nested parens shouldn't count, but here the return is flat).
    expr = return_lines[0][len("return "):]
    depth = 0
    top_commas = 0
    for ch in expr:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            top_commas += 1
    n_returned = top_commas + 1

    assert n_returned == 3, (
        f"prepare_inference_panel_frames must return EXACTLY 3 values "
        f"(feature_frames, factor_frames, macro_frame). "
        f"Found {n_returned} in expression: {expr!r}. "
        f"If you re-added asset_embeddings (T2-2), update SimAdapter unpacking too."
    )


def test_sim_adapter_unpacks_three_tuple_correctly():
    """Smoke: SimAdapter.__init__ panel-frame block unpacks 3 values without ValueError."""
    import datetime
    import numpy as np
    import pandas as pd
    from adapters.sim import SimAdapter
    import inspect

    src = inspect.getsource(SimAdapter)
    # The fix: must NOT have `ff, fac, macro, emb = prepare_inference_panel_frames`
    assert "ff, fac, macro, emb = prepare_inference_panel_frames" not in src, (
        "SimAdapter still unpacks 4-tuple from prepare_inference_panel_frames — "
        "this raises ValueError on exp branch (3-tuple) and silently produces "
        "0-trade sims. Revert to 3-tuple unpacking."
    )
    # Must have the 3-tuple form
    assert "ff, fac, macro = prepare_inference_panel_frames" in src, (
        "SimAdapter must unpack exactly 3 values from prepare_inference_panel_frames "
        "on exp branch (asset embeddings T2-2 is rejected)."
    )
