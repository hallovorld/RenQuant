"""Look-ahead leak tests for momentum features (2026-05-18 user audit).

Pin: NONE of the 5 momentum features may use any price at date > t when
computing the feature value at date t. A leak would inflate val_IC by
peeking at the label.

Strategy: build a synthetic price series with a SPIKE on day N. The
feature value at date N-1 MUST NOT see day N's spike (it has no info
about day N). The feature value at date N CAN see day N's data.

Also: assert equivalence with pandas_ta_classic library calls so we
can't regress to hand-implemented (buggy) versions.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def _load_module():
    """Import build_momentum_features as a module (script is not a package)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_momentum_features", REPO / "scripts/build_momentum_features.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_momentum_features"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mom_mod():
    return _load_module()


# ── Pure function: no-leakage on synthetic spike ────────────────────────

class TestNoLeakageOnSpike:
    """Inject a 10% SPIKE on day N. The feature at day N-1 MUST be the
    same as if day N didn't exist (causality)."""

    def _build_series_with_spike(self, length: int = 300, spike_day: int = 250,
                                  spike_pct: float = 0.10) -> pd.Series:
        np.random.seed(42)
        # Stable mean-reverting walk
        returns = np.random.normal(0.0, 0.005, length)
        # Insert spike
        returns[spike_day] = spike_pct
        prices = np.cumprod(1 + returns) * 100
        return pd.Series(prices, name="close")

    def test_mom_12_1_no_leak(self, mom_mod):
        close = self._build_series_with_spike(length=300, spike_day=250)
        feat_full = mom_mod.compute_mom_12_1(close)
        # Recompute using only data through day N-1 (spike_day=250 → use 0..249)
        close_truncated = close.iloc[:250]
        feat_truncated = mom_mod.compute_mom_12_1(close_truncated)
        # feat at day 249 should be IDENTICAL between full and truncated
        v_full = feat_full.iloc[249]
        v_truncated = feat_truncated.iloc[249]
        if not pd.isna(v_full):
            assert abs(v_full - v_truncated) < 1e-9, \
                f"LEAK: mom_12_1[249] differs full={v_full:.6f} truncated={v_truncated:.6f}"

    def test_mom_3m_no_leak(self, mom_mod):
        close = self._build_series_with_spike(length=300, spike_day=250)
        feat_full = mom_mod.compute_mom_3m(close)
        feat_truncated = mom_mod.compute_mom_3m(close.iloc[:250])
        v_full = feat_full.iloc[249]
        v_truncated = feat_truncated.iloc[249]
        if not pd.isna(v_full):
            assert abs(v_full - v_truncated) < 1e-9

    def test_dist_52w_high_no_leak(self, mom_mod):
        close = self._build_series_with_spike(length=300, spike_day=250)
        feat_full = mom_mod.compute_dist_52w_high(close)
        feat_truncated = mom_mod.compute_dist_52w_high(close.iloc[:250])
        v_full = feat_full.iloc[249]
        v_truncated = feat_truncated.iloc[249]
        if not pd.isna(v_full):
            assert abs(v_full - v_truncated) < 1e-9

    def test_abs_vol_30d_no_leak(self, mom_mod):
        close = self._build_series_with_spike(length=300, spike_day=250)
        feat_full = mom_mod.compute_abs_vol_30d(close)
        feat_truncated = mom_mod.compute_abs_vol_30d(close.iloc[:250])
        v_full = feat_full.iloc[249]
        v_truncated = feat_truncated.iloc[249]
        if not pd.isna(v_full):
            assert abs(v_full - v_truncated) < 1e-9


class TestLibraryEquivalence:
    """Verify our feature implementations match canonical library calls."""

    def test_mom_3m_matches_pandas_ta_roc(self, mom_mod):
        """mom_3m should equal pandas_ta.roc(close, length=63) / 100."""
        import pandas_ta_classic as ta
        np.random.seed(0)
        close = pd.Series(np.cumprod(1 + np.random.normal(0.0005, 0.015, 200)) * 100)
        ours = mom_mod.compute_mom_3m(close)
        canonical = ta.roc(close, length=63) / 100.0
        # Drop NaN for fair comparison
        mask = ours.notna() & canonical.notna()
        diff = (ours[mask] - canonical[mask]).abs().max()
        assert diff < 1e-9, f"mom_3m diverges from pandas_ta.roc: max diff = {diff:.2e}"

    def test_dist_52w_high_in_expected_range(self, mom_mod):
        """dist_52w_high MUST be in [-1, 0] by construction."""
        np.random.seed(0)
        close = pd.Series(np.cumprod(1 + np.random.normal(0.0005, 0.015, 300)) * 100)
        feat = mom_mod.compute_dist_52w_high(close).dropna()
        assert (feat <= 0).all(), \
            f"dist_52w_high should be ≤0 (close ≤ rolling_max); got max={feat.max()}"
        assert (feat >= -1).all(), \
            f"dist_52w_high should be ≥-1 (close > 0); got min={feat.min()}"

    def test_abs_vol_30d_positive(self, mom_mod):
        """Annualized vol is always non-negative."""
        np.random.seed(0)
        close = pd.Series(np.cumprod(1 + np.random.normal(0.0005, 0.015, 200)) * 100)
        feat = mom_mod.compute_abs_vol_30d(close).dropna()
        assert (feat >= 0).all()

    def test_abs_vol_30d_high_when_vol_high(self, mom_mod):
        """Series with σ=10x larger should yield ~10x larger vol estimate."""
        np.random.seed(0)
        low_vol = pd.Series(np.cumprod(1 + np.random.normal(0.0005, 0.005, 200)) * 100)
        np.random.seed(0)
        high_vol = pd.Series(np.cumprod(1 + np.random.normal(0.0005, 0.05, 200)) * 100)
        v_low = mom_mod.compute_abs_vol_30d(low_vol).iloc[-1]
        v_high = mom_mod.compute_abs_vol_30d(high_vol).iloc[-1]
        ratio = v_high / v_low
        assert 5 < ratio < 15, f"high/low vol ratio expected ~10, got {ratio:.2f}"


class TestSectorMomCorrectness:
    """sector_mom_30d should equal ret - mean(ret_within_sector_same_date)."""

    def test_within_sector_demean(self, mom_mod):
        panel = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "date":   ["2026-01-01"] * 4,
            "sector": ["Tech", "Tech", "Energy", "Energy"],
            "ret_30d": [0.10, 0.06, -0.02, 0.04],
        })
        panel["date"] = pd.to_datetime(panel["date"])
        out = mom_mod.compute_sector_mom_30d(panel)
        # Tech mean = (0.10 + 0.06) / 2 = 0.08; A's = 0.10 - 0.08 = +0.02
        assert abs(out.iloc[0] - 0.02) < 1e-9, f"A: got {out.iloc[0]:+.4f}"
        assert abs(out.iloc[1] - (-0.02)) < 1e-9
        # Energy mean = (-0.02 + 0.04) / 2 = 0.01; C's = -0.02 - 0.01 = -0.03
        assert abs(out.iloc[2] - (-0.03)) < 1e-9
        assert abs(out.iloc[3] - 0.03) < 1e-9


class TestSourceMarker:
    """Pin that the script uses pandas_ta_classic (not hand-implemented)."""

    def test_uses_pandas_ta_classic(self):
        src = (REPO / "scripts/build_momentum_features.py").read_text()
        assert "import pandas_ta_classic" in src
        assert "ta.roc" in src

    def test_2026_05_18_audit_marker(self):
        src = (REPO / "scripts/build_momentum_features.py").read_text()
        assert "2026-05-18 user audit" in src
