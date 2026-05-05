"""Smoke tests for the sanity / training script entry points.

After 2026-05-03 incident — `scripts/run_sanity_checks.py` had imported
``training_panel.purged_cv.PurgedCVSplitter`` which never existed (canonical
class is ``PurgedKFold``). The script silently failed inside
``run_ablation_followups.sh`` because that wrapper used ``|| true``.

This module just imports the script + verifies the symbols it relies on
actually exist. It runs in <1s and catches the entire class of "rename
rot" bugs.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))


class TestSanityCheckImports(unittest.TestCase):
    """Verify run_sanity_checks.py's downstream symbols still exist."""

    def test_purged_kfold_importable(self) -> None:
        from training_panel.purged_cv import PurgedKFold
        self.assertTrue(callable(PurgedKFold))
        # Must have a .split method that takes a DataFrame
        inst = PurgedKFold(n_splits=3, embargo_days=1, lookahead_days=1)
        self.assertTrue(hasattr(inst, "split"))

    def test_panel_ltr_model_importable(self) -> None:
        from training_panel.ltr_model import PanelLTRModel
        self.assertTrue(callable(PanelLTRModel))
        # Must support train + predict — the signature sanity script uses
        self.assertTrue(hasattr(PanelLTRModel, "train"))
        self.assertTrue(hasattr(PanelLTRModel, "predict"))

    def test_panel_training_pieces_importable(self) -> None:
        from training_panel.pp_panel_training import (
            PanelTrainingContext, PanelTrainingPipeline,
            PanelDataJob, PanelFeatureJob, PanelAssemblyJob,
        )
        for cls in (PanelTrainingContext, PanelTrainingPipeline,
                    PanelDataJob, PanelFeatureJob, PanelAssemblyJob):
            self.assertTrue(callable(cls), f"{cls!r} not callable")

    def test_run_sanity_checks_module_imports_cleanly(self) -> None:
        """Loading the script as a module must not raise.

        argparse parsing happens inside main(); top-level imports are the
        rename-rot trap.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_sanity_checks", REPO / "scripts" / "run_sanity_checks.py",
        )
        self.assertIsNotNone(spec, "spec_from_file_location returned None")
        mod = importlib.util.module_from_spec(spec)
        # Loading executes top-of-file imports + any module-level code, but
        # main() only runs under __main__. This is the smoke we want.
        spec.loader.exec_module(mod)
        self.assertTrue(callable(getattr(mod, "main", None)),
                        "run_sanity_checks.py must expose main()")


if __name__ == "__main__":
    unittest.main()
