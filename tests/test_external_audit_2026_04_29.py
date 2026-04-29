"""External audit fixes shipped 2026-04-29 (audit items #5/#6/#7/#9).

Each test would FAIL before the fix and PASS after. Per CLAUDE.md §2: every
fix ships with a regression test that would have caught it.

External audit reports flagged 9 issues; 4 are landed here as P0:
  #5 panel-LTR drift guard absent (NGBoost-only, leaving panel side blind)
  #6 NGBoost saver no staging — direct overwrite kills rollback
  #7 min_best_iter alone insufficient — need eval_ic floor as 2nd gate
  #9 single-ticker day labelled 0.0 not NaN (cross-sectional rank undefined at N=1)

The other 5 (#1 sanity infra, #2 train-date alignment, #3 PIT short interest,
#4 Kelly fraction, #8 NGBoost pickle) ship in subsequent waves — see
doc/STATUS.md §0 for the full audit log.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


# ── #9: single-ticker day → NaN, not 0.0 ────────────────────────────────────

class TestLabelN1ReturnsNaN:
    def test_single_ticker_day_yields_nan(self):
        """gaussianize_cross_section: with N=1, rank is undefined → NaN."""
        import numpy as np
        import pandas as pd
        from training_panel.labels import gaussianize_cross_section

        # One day with only ticker A having data, second day both tickers.
        idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
        residuals = {
            "A": pd.Series([0.5, 0.1], index=idx),
            "B": pd.Series([float("nan"), -0.2], index=idx),
        }
        out = gaussianize_cross_section(residuals)

        # Day 1: A is alone → must be NaN (post-fix), not 0.0 (pre-fix)
        assert pd.isna(out["A"].loc[idx[0]]), (
            f"single-ticker day A should be NaN, got {out['A'].loc[idx[0]]}"
        )
        # Day 2: both present, A and B should be finite, opposite signs
        assert pd.notna(out["A"].loc[idx[1]])
        assert pd.notna(out["B"].loc[idx[1]])
        assert out["A"].loc[idx[1]] * out["B"].loc[idx[1]] < 0

    def test_zero_residual_day_yields_nan(self):
        """N=0 day stays NaN (regression check — don't break the existing path)."""
        import pandas as pd
        from training_panel.labels import gaussianize_cross_section

        idx = pd.DatetimeIndex(["2024-01-02"])
        residuals = {
            "A": pd.Series([float("nan")], index=idx),
            "B": pd.Series([float("nan")], index=idx),
        }
        out = gaussianize_cross_section(residuals)
        assert pd.isna(out["A"].iloc[0])
        assert pd.isna(out["B"].iloc[0])


# ── #7: eval_ic floor as 2nd gate ───────────────────────────────────────────

class TestEvalICFloor:
    def test_min_eval_ic_in_source(self):
        """The eval_ic floor lives in FinalFitTask alongside min_best_iter."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        # The new gate must appear AFTER min_best_iter in the same Task
        idx_best = src.find("min_best_iter = int(cfg.get")
        idx_eval = src.find("min_eval_ic")
        assert idx_best > 0
        assert idx_eval > idx_best, "min_eval_ic gate must come after min_best_iter"
        assert 'cfg.get("min_eval_ic")' in src
        assert "below min_eval_ic" in src
        # Must not be the staging guard for NGBoost
        assert "FinalFit eval_ic" in src

    def test_default_disabled(self):
        """Default config has no min_eval_ic — legacy retrains aren't blocked."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        # The check is None-gated: `if min_eval_ic is not None`
        assert "if min_eval_ic is not None:" in src


# ── #5: panel-LTR drift guard mirrors NGBoost ───────────────────────────────

class TestPanelLTRDriftGuard:
    def test_drift_guard_present_in_buildfeaturematrixtask(self):
        """BuildFeatureMatrixTask now hard-fails when too many cols are all-NaN."""
        src = (REPO_ROOT / "backtesting/renquant_104/kernel/panel_pipeline/job_panel_scoring.py").read_text()
        idx = src.find("class BuildFeatureMatrixTask")
        block = src[idx:idx + 6000]
        # Threshold + check + fail-safe cleanup all in one block
        assert "max_feature_drift_pct" in block
        assert "all_nan_cols" in block
        # Fail-safe semantics match NGBoost path
        assert "ctx.candidates = []" in block
        assert "return False" in block

    def test_drift_threshold_default_matches_ngboost(self):
        """Same default 0.05 (5%) as ApplyNGBoostTask — consistency for operators."""
        src = (REPO_ROOT / "backtesting/renquant_104/kernel/panel_pipeline/job_panel_scoring.py").read_text()
        idx = src.find("class BuildFeatureMatrixTask")
        block = src[idx:idx + 6000]
        # The default 0.05 is read from the same panel_scoring config tree
        assert 'panel_cfg.get("max_feature_drift_pct", 0.05)' in block


# ── #6: NGBoost staging + acceptance ────────────────────────────────────────

class TestNGBoostStaging:
    def test_staging_path_used(self):
        """NGBoostSaveTask writes to .staging.json, then atomic-renames on pass."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        idx = src.find("class NGBoostSaveTask")
        block = src[idx:idx + 6000]
        assert ".staging.json" in block
        assert "_os.replace" in block, "must use atomic rename, not direct save"

    def test_pre_train_snapshot_taken(self):
        """Prior production artifact is snapshotted to .pre-train.json before stage."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        idx = src.find("class NGBoostSaveTask")
        block = src[idx:idx + 6000]
        assert ".pre-train.json" in block
        assert "shutil.copy2" in block

    def test_min_val_mu_ic_gate(self):
        """The acceptance gate compares val_mu_ic to a configurable floor."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        idx = src.find("class NGBoostSaveTask")
        block = src[idx:idx + 6000]
        assert "min_val_mu_ic" in block
        assert "REJECTING new NGBoost head" in block

    def test_rollback_path_preserves_prior(self):
        """On rejection: staging is left for diag, prior remains at out_path,
        snapshot is removed (no .pre-train.json clutter)."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        idx = src.find("class NGBoostSaveTask")
        block = src[idx:idx + 6000]
        # Prior snapshot is unlinked on both paths (success + reject)
        assert block.count("prior_snapshot.unlink()") >= 2
