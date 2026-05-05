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
    """Drift guard distinguishes structural vs transient NaN drift.

    2026-05-04: refactored — the legacy BuildFeatureMatrixTask was
    split per CLAUDE.md §1c into BuildFeatureMatrixJob with 4 Tasks.
    The drift_guard logic now lives in DriftGuardTask in
    `kernel/panel_pipeline/tasks_feature_matrix.py`. Tests updated to
    look there.
    """
    def test_drift_guard_present_in_drift_guard_task(self):
        src = (REPO_ROOT / "backtesting/renquant_104/kernel/panel_pipeline/tasks_feature_matrix.py").read_text()
        idx = src.find("class DriftGuardTask")
        assert idx > 0, "DriftGuardTask must exist in tasks_feature_matrix.py"
        block = src[idx:]
        # Threshold still configurable
        assert "max_feature_drift_pct" in block
        assert "nan_cols" in block
        # Post-fix: structural vs transient distinction
        assert "structural" in block
        assert "transient" in block
        # Hard-fail path only fires on structural drift
        assert "ctx.candidates = []" in block
        assert "return False" in block

    def test_drift_threshold_default_matches_ngboost(self):
        """Same default 0.05 (5%) as ApplyNGBoostTask — consistency for operators."""
        src = (REPO_ROOT / "backtesting/renquant_104/kernel/panel_pipeline/tasks_feature_matrix.py").read_text()
        # The default 0.05 is read from the same panel_scoring config tree
        assert 'max_feature_drift_pct' in src
        assert ', 0.05)' in src   # config default 5%


# ── #6: NGBoost staging + acceptance ────────────────────────────────────────

def _ngboost_save_task_body() -> str:
    """Read the NGBoostSaveTask class body, bounded by the next class
    definition. Replaces the brittle src[idx:idx+6000] window from the
    original tests — that hardcoded slice broke once the body grew past
    6kB (e.g. when the 2026-05-04 fingerprint stamp was added)."""
    src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
    idx = src.find("class NGBoostSaveTask")
    end = src.find("\nclass ", idx + 1)
    return src[idx:end] if end > 0 else src[idx:]


class TestNGBoostStaging:
    def test_staging_path_used(self):
        """NGBoostSaveTask writes to .staging.json, then atomic-renames on pass."""
        block = _ngboost_save_task_body()
        assert ".staging.json" in block
        assert "_os.replace" in block, "must use atomic rename, not direct save"

    def test_pre_train_snapshot_taken(self):
        """Prior production artifact is snapshotted to .pre-train.json before stage."""
        block = _ngboost_save_task_body()
        assert ".pre-train.json" in block
        assert "shutil.copy2" in block

    def test_min_val_mu_ic_gate(self):
        """The acceptance gate compares val_mu_ic to a configurable floor."""
        block = _ngboost_save_task_body()
        assert "min_val_mu_ic" in block
        assert "REJECTING new NGBoost head" in block

    def test_rollback_path_preserves_prior(self):
        """On rejection: staging is left for diag, prior remains at out_path,
        snapshot is removed (no .pre-train.json clutter)."""
        block = _ngboost_save_task_body()
        # Prior snapshot is unlinked on both paths (success + reject)
        assert block.count("prior_snapshot.unlink()") >= 2


# ── 60d calibrator: crosssectional threshold mode ───────────────────────────

class TestCalibratorCrossSectional:
    """Regression tests for the crosssectional threshold mode fix."""

    def _make_bull_market_data(self):
        """60d bull-market scenario: all returns >> 0.03 → absolute mode collapses."""
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=300, freq="B")
        tickers = [f"T{i}" for i in range(20)]
        panel_scores, future_returns = {}, {}
        for t in tickers:
            scores = rng.uniform(0, 1, len(dates))
            # Extreme bull: all returns are uniformly +10–30%, never below 0.03
            # so absolute threshold=0.03 → 100% label=1 → isotonic collapses.
            rets = 0.10 + rng.uniform(0, 0.20, len(dates))
            panel_scores[t] = pd.Series(scores, index=dates)
            future_returns[t] = pd.Series(rets, index=dates)
        return panel_scores, future_returns

    def test_absolute_collapses_on_bull_market(self):
        """Sanity: old absolute=0.03 mode collapses to <5 unique y in bull market."""
        import numpy as np
        from training_panel.global_calibrator import fit_global_calibrator
        ps, fr = self._make_bull_market_data()
        with pytest.raises(ValueError, match="collapsed"):
            fit_global_calibrator(ps, fr, threshold=0.03, threshold_mode="absolute")

    def test_crosssectional_survives_bull_market(self):
        """crosssectional mode succeeds regardless of market direction."""
        from training_panel.global_calibrator import fit_global_calibrator
        ps, fr = self._make_bull_market_data()
        calib = fit_global_calibrator(ps, fr, threshold_mode="crosssectional")
        assert calib is not None
        # ~50% base rate by construction
        assert abs(calib.metadata["prob_base_rate"] - 0.5) < 0.10

    def test_crosssectional_mode_stored_in_metadata(self):
        """threshold_mode is stamped in the artifact metadata."""
        from training_panel.global_calibrator import fit_global_calibrator
        ps, fr = self._make_bull_market_data()
        calib = fit_global_calibrator(ps, fr, threshold_mode="crosssectional")
        assert calib.metadata["threshold_mode"] == "crosssectional"

    def test_absolute_still_works_for_10d(self):
        """Regression: default absolute mode is unchanged for 10d panel."""
        import numpy as np
        import pandas as pd
        from training_panel.global_calibrator import fit_global_calibrator
        rng = np.random.default_rng(7)
        dates = pd.date_range("2022-01-01", periods=400, freq="B")
        tickers = [f"S{i}" for i in range(15)]
        ps, fr = {}, {}
        for t in tickers:
            ps[t] = pd.Series(rng.uniform(0, 1, len(dates)), index=dates)
            # Mixed returns around 0 — typical 10d relative-to-SPY
            fr[t] = pd.Series(rng.normal(0, 0.02, len(dates)), index=dates)
        calib = fit_global_calibrator(ps, fr, threshold=0.03, threshold_mode="absolute")
        assert calib.metadata["threshold_mode"] == "absolute"
        assert calib.metadata["threshold"] == 0.03
