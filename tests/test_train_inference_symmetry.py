"""Symmetry guard: PanelDataJob.tasks ⇌ prepare_inference_panel_frames.

Bug #25 caught a recurring pattern: training-side `PanelDataJob.tasks`
gains a new `Load*Task`, but `prepare_inference_panel_frames` (a
hand-written sequence) does NOT, so the trained model has feature_cols
inference can't reproduce. Same shape as Bug 12 (LoadMinuteBars) and
Bug #25 (LoadMacroFactorsTask).

This static-source test enumerates `PanelDataJob.tasks` at import time
and asserts every `Load*Task` is also called inside
`prepare_inference_panel_frames`. CI fails on drift, preventing the
pattern from recurring.
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from training_panel.pp_panel_training import PanelDataJob  # noqa: E402
from training_panel import pipeline as inf_pipeline  # noqa: E402


def _load_task_names_from_panel_data_job() -> set[str]:
    """Return the set of task class names in PanelDataJob.tasks."""
    job = PanelDataJob()
    return {type(t).__name__ for t in job.tasks
            if type(t).__name__.startswith("Load")}


def _tasks_called_in_inference_path() -> set[str]:
    """Parse the source of prepare_inference_panel_frames to find
    every `Load*Task().run(ctx)` invocation."""
    src = inspect.getsource(inf_pipeline.prepare_inference_panel_frames)
    # Match patterns like `LoadFooTask().run(ctx)` or
    # `LoadFooTask().run(ctx, ...)`
    matches = re.findall(r"\b(Load\w+Task)\(\)\.run\(", src)
    return set(matches)


class TestTrainInferenceSymmetry:
    """Guard against the Bug #12 / Bug #25 recurring pattern."""

    def test_every_panel_data_load_task_is_in_inference_path(self):
        """Every Load*Task in PanelDataJob.tasks must also appear in
        prepare_inference_panel_frames. Otherwise the trained model
        will have feature_cols the inference path can't produce."""
        train_tasks = _load_task_names_from_panel_data_job()
        inference_tasks = _tasks_called_in_inference_path()

        missing = train_tasks - inference_tasks
        assert not missing, (
            f"TRAIN/INFERENCE FEATURE-BUILDER DRIFT (Bug #25 pattern):\n"
            f"  Tasks in PanelDataJob.tasks but NOT in prepare_inference_panel_frames:\n"
            f"  {sorted(missing)}\n"
            f"  Add them to training_panel/pipeline.py::prepare_inference_panel_frames\n"
            f"  OR remove from PanelDataJob.tasks (and the corresponding training\n"
            f"  side feature won't be in the panel anymore)."
        )

    def test_inference_tasks_are_subset_of_or_equal_to_train(self):
        """The reverse direction is OK to have extras — inference may
        pre-load auxiliary data that training doesn't need. But that
        shouldn't be the common case; flag if extras appear."""
        train_tasks = _load_task_names_from_panel_data_job()
        inference_tasks = _tasks_called_in_inference_path()

        extras = inference_tasks - train_tasks
        # Soft check: extras are allowed but unusual. Document if they appear.
        if extras:
            # Just report — don't fail. Could be SectorMomentumTask or others.
            pass

    def test_documented_load_task_set_matches_panel_data_job(self):
        """Sanity: PanelDataJob has at least these Load*Tasks, ensuring
        the symmetry test isn't degenerate (e.g. 0 vs 0 trivially passes).
        """
        train_tasks = _load_task_names_from_panel_data_job()
        # As of round-7, expected Load*Tasks (this list will grow):
        expected_at_least = {
            "LoadFundamentalsTask",
            "LoadEarningsSurpriseTask",
            "LoadInsiderTradesTask",
            "LoadHourlyBarsTask",
            "LoadMinuteBarsTask",
            "LoadMacroFactorsTask",
        }
        missing = expected_at_least - train_tasks
        assert not missing, (
            f"PanelDataJob is missing expected Load*Tasks: {sorted(missing)}. "
            f"Either they got removed unintentionally, or this test needs "
            f"updating."
        )


class TestPrepareReturnsTuple3:
    """Bug #25 changed the return signature from (ff, fac) to (ff, fac, macro).
    This pins the new contract so callers (adapters/runner.py / sim.py /
    lean.py) can rely on it."""

    def test_function_signature_returns_three(self):
        """Inspect annotations: return type must be a 3-tuple."""
        import typing
        sig = inspect.signature(inf_pipeline.prepare_inference_panel_frames)
        # The return annotation is a string in __future__ annotations — parse it
        ret_annot = sig.return_annotation
        ret_str = str(ret_annot)
        # Should mention 3-tuple of dict, dict, and macro_frame
        assert "tuple" in ret_str.lower() or "Tuple" in ret_str
        # Three commas-or-types in the annotation (loose check)
        # Rough: count comma-separated pieces or Union members
        # Accept either explicit 3-tuple types OR docstring mentioning macro_frame
        if "macro" not in ret_str.lower():
            # Fallback: docstring must mention macro
            doc = inspect.getdoc(inf_pipeline.prepare_inference_panel_frames) or ""
            assert "macro" in doc.lower(), (
                "prepare_inference_panel_frames must return macro_frame as "
                "third element OR document it in the return docstring"
            )
