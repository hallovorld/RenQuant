"""Regression: all panel-using adapters must unpack the same arity from
prepare_inference_panel_frames as the function actually returns.

Bug class: silent unpacking-arity mismatch between pipeline.py's
`prepare_inference_panel_frames` return tuple and the per-adapter
caller. When pipeline grows from 3-tuple to 4-tuple (T2-2 phase C-2,
commit 56c9fc6), each adapter's call site must update too.

Pre-fix incident (2026-05-02): on main, pipeline.py returned 4-tuple
(ff, fac, macro, emb) but lean.py still unpacked 3-tuple. The broad
`except Exception` swallowed the ValueError silently; LEAN backtests
ran end-to-end with `_panel_cache_ff = None` → panel scoring disabled
→ no diagnostic output. Caught only because the analogous bug bit
the sim adapter on exp branch (3-tuple pipeline + 4-tuple sim.py
unpacking).

Invariant pinned by these tests:
   arity(prepare_inference_panel_frames return) ==
   arity(adapters/sim.py unpack) ==
   arity(adapters/runner.py unpack) ==
   arity(adapters/lean.py unpack)

If any one drifts, this test fires immediately — pre-merge, not at
runtime.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RQ_ROOT = REPO_ROOT / "backtesting" / "renquant_104"
sys.path.insert(0, str(RQ_ROOT))


def _count_top_level_commas(expr: str) -> int:
    depth = 0
    n = 0
    for ch in expr:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            n += 1
    return n


def _pipeline_return_arity() -> int:
    from training_panel.pipeline import prepare_inference_panel_frames
    src = inspect.getsource(prepare_inference_panel_frames)
    return_lines = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("return ")]
    assert len(return_lines) == 1, (
        f"prepare_inference_panel_frames must have exactly 1 return statement; "
        f"found {len(return_lines)}: {return_lines}"
    )
    expr = return_lines[0][len("return "):]
    return _count_top_level_commas(expr) + 1


def _adapter_unpack_arity(adapter_module_name: str) -> int:
    """Find the line `<vars> = prepare_inference_panel_frames(...)` and
    count the variables on the LHS."""
    src_path = RQ_ROOT / "adapters" / f"{adapter_module_name}.py"
    src = src_path.read_text()
    # Match e.g. `ff, fac, macro = prepare_inference_panel_frames(`
    # or `ff, fac, macro, emb = prepare_inference_panel_frames(`
    pattern = re.compile(
        r"^(\s*)([\w,\s]+?)\s*=\s*prepare_inference_panel_frames\b",
        re.MULTILINE,
    )
    matches = pattern.findall(src)
    assert len(matches) >= 1, f"No `... = prepare_inference_panel_frames(...)` line in {src_path}"
    # If multiple call sites, all must agree
    arities = []
    for _indent, lhs in matches:
        arity = _count_top_level_commas(lhs.strip()) + 1
        arities.append(arity)
    assert len(set(arities)) == 1, (
        f"Multiple call sites in {adapter_module_name} have different unpack arities: {arities}"
    )
    return arities[0]


def test_pipeline_return_arity_pinned():
    """Pipeline currently returns 4-tuple (T2-2 added asset_embeddings). If this
    changes, also update every adapter at the same time."""
    assert _pipeline_return_arity() == 4, (
        "prepare_inference_panel_frames return arity changed. Update every "
        "adapter (sim.py, runner.py, lean.py) to match before changing this test."
    )


def test_sim_adapter_unpack_matches_pipeline():
    pipe = _pipeline_return_arity()
    sim = _adapter_unpack_arity("sim")
    assert sim == pipe, (
        f"adapters/sim.py unpacks {sim} values; pipeline returns {pipe}. "
        f"This silent-mismatch produces 0-trade sims (caught 2026-05-02 on exp branch)."
    )


def test_runner_adapter_unpack_matches_pipeline():
    pipe = _pipeline_return_arity()
    runner = _adapter_unpack_arity("runner")
    assert runner == pipe, (
        f"adapters/runner.py unpacks {runner} values; pipeline returns {pipe}. "
        f"Same silent-failure pattern as sim adapter."
    )


def test_lean_adapter_unpack_matches_pipeline():
    pipe = _pipeline_return_arity()
    lean = _adapter_unpack_arity("lean")
    assert lean == pipe, (
        f"adapters/lean.py unpacks {lean} values; pipeline returns {pipe}. "
        f"LEAN backtests crash with no diagnostic — broad except swallows "
        f"the ValueError and disables panel scoring silently."
    )
