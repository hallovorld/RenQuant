"""Tests for M2 v3 horizon blender helpers (the 5 audit fixes).

Verifies the unit-level invariants of:
  Fix 4: per-date cross-sectional rank target
  Fix 5: winsorize at [0.5%, 99.5%]
  Plus the script imports cleanly and exposes the right API.

The end-to-end blender run is too heavy for pytest (panel prep is ~15min).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _load_v3_module():
    spec = importlib.util.spec_from_file_location(
        "_v3", REPO_ROOT / "scripts" / "train_horizon_blender_v3.py",
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── Module structure ────────────────────────────────────────────────────────

class TestV3ModuleStructure:
    def test_module_imports(self):
        v3 = _load_v3_module()
        assert callable(v3.main)
        assert callable(v3._train_blender_v3)
        assert callable(v3._winsorize)
        assert callable(v3._per_date_rank)

    def test_horizon_set_unchanged(self):
        v3 = _load_v3_module()
        assert v3.HORIZONS == (10, 20, 60)
        assert set(v3.HORIZON_ARTIFACTS.keys()) == {10, 20, 60}


# ── Fix 5: winsorize ─────────────────────────────────────────────────────────

class TestWinsorize:
    def test_clips_extremes(self):
        v3 = _load_v3_module()
        df = pd.DataFrame({"x": list(range(100))})
        out = v3._winsorize(df, ["x"], lo_q=0.05, hi_q=0.95)
        assert out["x"].min() >= np.quantile(range(100), 0.05) - 1e-9
        assert out["x"].max() <= np.quantile(range(100), 0.95) + 1e-9

    def test_does_not_mutate_input(self):
        """Winsorize must return a copy, not mutate caller's df."""
        v3 = _load_v3_module()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 100.0]})
        before = df["x"].max()
        v3._winsorize(df, ["x"], lo_q=0.0, hi_q=0.5)
        assert df["x"].max() == before, "winsorize mutated input"

    def test_handles_inf_values(self):
        v3 = _load_v3_module()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.inf, -np.inf]})
        out = v3._winsorize(df, ["x"], lo_q=0.1, hi_q=0.9)
        # Inf values get clipped to finite quantiles
        # (pandas .quantile ignores inf; here we check no NaN spillover)
        assert out["x"].notna().all()

    def test_preserves_other_columns(self):
        v3 = _load_v3_module()
        df = pd.DataFrame({"x": [1.0, 2.0, 100.0], "y": ["a", "b", "c"]})
        out = v3._winsorize(df, ["x"], lo_q=0.0, hi_q=0.5)
        assert (out["y"].values == np.array(["a", "b", "c"])).all()


# ── Fix 4: per-date rank target ─────────────────────────────────────────────

class TestPerDateRank:
    def test_within_date_pct_rank(self):
        """Each date's values get ranked into [0, 1] within the date."""
        v3 = _load_v3_module()
        # 2 dates, 3 tickers each. Highest fwd_return → rank 1.0.
        s = pd.Series([0.01, 0.02, 0.03, -0.01, 0.0, 0.05])
        d = pd.Series(["d1", "d1", "d1", "d2", "d2", "d2"])
        out = v3._per_date_rank(s, d)
        # On d1: 0.01<0.02<0.03 → ranks 1/3, 2/3, 3/3
        # On d2: -0.01<0.0<0.05 → ranks 1/3, 2/3, 3/3
        np.testing.assert_allclose(out, [1/3, 2/3, 1.0, 1/3, 2/3, 1.0], rtol=1e-6)

    def test_ranks_are_pct_in_zero_one(self):
        v3 = _load_v3_module()
        rng = np.random.default_rng(0)
        n = 1000
        s = pd.Series(rng.normal(0, 1, n))
        d = pd.Series(rng.choice(["d1", "d2", "d3"], n))
        out = v3._per_date_rank(s, d)
        assert out.min() > 0.0 and out.max() <= 1.0

    def test_handles_ties(self):
        v3 = _load_v3_module()
        s = pd.Series([0.5, 0.5, 0.5])
        d = pd.Series(["d1", "d1", "d1"])
        out = v3._per_date_rank(s, d)
        # All three values tied → average rank 0.5/3 + 1.5/3 + 2.5/3 / 3 with pct
        # pandas default 'average' method — all three get rank 2 → pct = 2/3
        assert all(abs(v - 2/3) < 1e-9 for v in out)


# ── Reference / fix annotations present in script ────────────────────────────

class TestFixAnnotations:
    """Each of the 5 fixes is documented in the script with a reference.
    Catches the case where someone removes a fix without updating docs.
    """
    def _src(self):
        return (REPO_ROOT / "scripts" / "train_horizon_blender_v3.py").read_text()

    def test_fix1_standardscaler_in_pipeline(self):
        s = self._src()
        assert "StandardScaler" in s
        assert "Pipeline" in s
        # Pipeline holds both scaler and estimator
        assert '("scaler", StandardScaler())' in s

    def test_fix2_purged_cv_with_embargo(self):
        s = self._src()
        assert "PurgedKFold" in s
        assert "López de Prado 2018 Ch.7" in s
        assert "embargo_days" in s

    def test_fix3_elasticnet(self):
        s = self._src()
        assert "ElasticNetCV" in s
        assert "Zou & Hastie 2005" in s
        assert "l1_ratio" in s

    def test_fix4_per_date_rank(self):
        s = self._src()
        assert "_per_date_rank" in s
        assert "Cao et al. 2007" in s

    def test_fix5_winsorize(self):
        s = self._src()
        assert "_winsorize" in s
        # Default percentile range for the audit
        assert "0.005" in s and "0.995" in s

    def test_aa_sanity_present(self):
        s = self._src()
        # Per CLAUDE.md principle 5.2
        assert "A/A" in s or "shuffled" in s
        assert "rng.shuffle" in s or "permutation" in s

    def test_baseline_blends_present(self):
        s = self._src()
        # Equal-weight + 1/IC weighted (DeMiguel reference)
        assert "DeMiguel" in s
        assert "Equal-weight" in s
        assert "1/IC" in s
