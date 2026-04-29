"""Tests for the 4 P0 CV bugs discovered 2026-04-28.

  BUG-CV-1: linspace fold boundary drift (data leakage on n_dates roll)
  BUG-CV-2: best_iter < 20 silently saved (production was best_iter=4)
  BUG-CV-3: early-stop eval set misaligned with CPCV last fold
  BUG-G7  : acceptance gate hardcoded panel-ltr.json — ignores
            panel_ltr.artifact_path config

These fixes are mandatory before any retrain. See CLAUDE.md "P0 BUGS"
section.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


# ── BUG-CV-1: integer-division fold edges, no drift ────────────────────────

class TestBugCV1FoldStability:
    """Same calendar dates must land in the same fold regardless of small
    n_dates changes (rolling daily window adds 1-2 dates)."""

    def _split(self, n_dates: int, n_splits: int = 6):
        from training_panel.purged_cv import PurgedKFold  # noqa: PLC0415
        # Build a panel with `n_dates` unique dates, 5 tickers each
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
        panel = pd.DataFrame({
            "date":   np.repeat(dates, 5),
            "ticker": np.tile(list("ABCDE"), n_dates),
        })
        cv = PurgedKFold(n_splits=n_splits, embargo_days=2, lookahead_days=10)
        folds = []
        for tr, te in cv.split(panel, date_col="date"):
            test_dates = sorted(set(panel.iloc[te]["date"]))
            folds.append(test_dates)
        return folds

    def test_fold_edges_use_integer_division(self):
        """The fold boundaries must come from integer division, not linspace.
        With n_dates=100, n_splits=5 → fold size 20, edges [0,20,40,60,80,100]."""
        folds = self._split(n_dates=100, n_splits=5)
        # Each fold has n_dates // n_splits = 20 unique dates (last absorbs remainder)
        for i, f in enumerate(folds):
            if i < 4:
                assert len(f) == 20, f"fold {i} expected 20 dates, got {len(f)}"

    def test_fold_assignment_stable_across_n_dates_roll(self):
        """The most-recent date in fold k of run-1 (n_dates=750) must also
        land in fold k of run-2 (n_dates=753) — no drift from a daily roll."""
        f750 = self._split(n_dates=750, n_splits=6)
        f753 = self._split(n_dates=753, n_splits=6)
        # The last fold's first date in both runs should be the same calendar
        # date (the rollover added 3 dates at the END, which fall in the LAST
        # fold; earlier folds are unchanged).
        # Actually — earlier fold edges shouldn't shift. Check fold 0..n-2.
        for i in range(len(f750) - 1):
            # Each non-final fold has exactly fold_size dates; the START of
            # fold i is i × fold_size. fold_size differs (750/6=125 vs 753/6=125
            # in integer division — same!). So edges are identical for the
            # first n-1 folds.
            assert f750[i][0] == f753[i][0], (
                f"fold {i} start drifted: {f750[i][0]} vs {f753[i][0]}"
            )

    def test_no_remainder_lost(self):
        """When n_dates % n_splits != 0, the last fold absorbs the remainder
        so total coverage is exact."""
        folds = self._split(n_dates=753, n_splits=6)
        total = sum(len(f) for f in folds)
        assert total == 753, f"total coverage {total} ≠ 753 (lost remainder)"


# ── BUG-CV-2: best_iter < 20 must hard-fail before saving ──────────────────

class TestBugCV2BestIterGuard:
    """The production model had best_iter=4 silently saved.
    With eta=0.02, that's 0.08 total shrinkage — model is untrained.
    Guard must refuse to save when best_iter < min_best_iter (default 20).
    """

    def test_audit_tag_present(self):
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        assert "BUG-CV-2 hard guard" in src
        assert "min_best_iter" in src

    def test_guard_raises_when_best_iter_below_threshold(self):
        """Construct a minimal FinalFitTask scenario with best_iter=4."""
        # Simulating just the guard logic (the full task would need a real
        # XGBoost training loop). We test that the source contains the raise.
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        # The raise statement must mention the early-stop semantics
        assert "FinalFit early_stopping fired at round" in src
        assert "Artifact NOT saved" in src
        assert 'cfg.get("min_best_iter", 20)' in src

    def test_guard_skipped_for_transformer_backend(self):
        """Transformer has different best_iter semantics; guard must skip."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        # The guard runs only when backend in (xgboost, lightgbm)
        idx = src.find("BUG-CV-2 hard guard")
        block = src[idx:idx + 2000]
        assert 'backend in ("xgboost", "lightgbm")' in block

    def test_guard_configurable(self):
        """Operator can override min_best_iter for diagnostic runs."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        idx = src.find("BUG-CV-2 hard guard")
        block = src[idx:idx + 2000]
        assert "panel_ltr.min_best_iter" in block, (
            "error message must tell operator how to override"
        )


# ── BUG-CV-3: early-stop eval must align with CPCV last fold ───────────────

class TestBugCV3EvalAlignment:
    def test_audit_tag_present(self):
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        assert "BUG-CV-3 fix" in src

    def test_eval_size_matches_cpcv_fold_size(self):
        """n_eval must come from cv_n_splits (1/n_splits of dates),
        not the hardcoded 20%."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        idx = src.find("BUG-CV-3 fix")
        block = src[idx:idx + 1000]
        assert 'cv_splits_for_eval = int(cfg.get("cv_n_splits"' in block
        assert "n_total // max(2, cv_splits_for_eval)" in block

    def test_old_hardcoded_20pct_removed(self):
        """The pre-fix `int(round(n_total * 0.20))` must be gone."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        # The old expression was `n_eval  = max(2, int(round(n_total * 0.20)))`
        # The fix line uses cv_splits_for_eval. Source must not contain the
        # old hardcoded fraction in the FinalFitTask area.
        idx = src.find("BUG-CV-3 fix")
        block = src[idx:idx + 800]
        assert "n_total * 0.20" not in block, (
            "the hardcoded 20% must be replaced — block still contains it"
        )


# ── BUG-G7: acceptance gate must respect panel_ltr.artifact_path ───────────

class TestBugG7ArtifactPath:
    def test_audit_tag_present(self):
        src = (REPO_ROOT / "scripts/train_104.py").read_text()
        assert "BUG-G7 fix" in src

    def test_reads_artifact_path_from_config(self):
        src = (REPO_ROOT / "scripts/train_104.py").read_text()
        assert "panel_ltr.artifact_path" in src or "artifact_rel" in src
        # Specific implementation: panel_cfg.get("artifact_path", "artifacts/panel-ltr.json")
        assert 'panel_cfg.get("artifact_path", "artifacts/panel-ltr.json")' in src

    def test_old_hardcoded_path_removed(self):
        """The pre-fix `strategy_dir / "artifacts" / "panel-ltr.json"` should
        no longer be the active_path source — it's now the default fallback
        inside panel_cfg.get(...)."""
        src = (REPO_ROOT / "scripts/train_104.py").read_text()
        # The hardcoded line `active_path = strategy_dir / "artifacts" / "panel-ltr.json"`
        # must NOT appear; only the configurable path setup.
        # Defensive: this is a string contract.
        assert 'active_path = strategy_dir / "artifacts" / "panel-ltr.json"' not in src
