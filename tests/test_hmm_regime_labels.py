"""Regression tests for kernel/hmm_regime_labels.py.

Pin the HMM-style stateless classifier so bull_regime_ic (user-mandated
PatchTST swap criterion 2026-05-18) is stable across refactors.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from kernel.hmm_regime_labels import (compute_hmm_regime_labels,
                                        per_hmm_regime_ic, bull_regime_ic,
                                        BEAR_VOL_20D_THR, BEAR_RET_20D_THR)


@pytest.mark.skipif(not (REPO / "data/ohlcv/SPY/1d.parquet").exists(),
                     reason="SPY parquet not available")
class TestRegimeClassifier:
    """Real SPY data — verify classifier emits expected regimes in known
    historical periods."""

    def test_emits_4_regime_labels(self):
        labels = compute_hmm_regime_labels(REPO / "data/ohlcv/SPY/1d.parquet")
        unique = set(labels["regime"].unique())
        expected = {"BULL_CALM", "BULL_VOLATILE", "BEAR", "CHOPPY"}
        assert unique.issubset(expected), f"unexpected regime: {unique - expected}"
        # Must emit at least 2 distinct regimes over 10 years of SPY
        assert len(unique) >= 3

    def test_covid_period_labeled_bear(self):
        labels = compute_hmm_regime_labels(REPO / "data/ohlcv/SPY/1d.parquet")
        covid = labels[(labels.date >= "2020-02-15") & (labels.date < "2020-05-01")]
        bear_pct = (covid["regime"] == "BEAR").mean()
        assert bear_pct > 0.5, f"COVID period only {bear_pct:.0%} BEAR — detector broken"

    def test_2022_bear_includes_bear_days(self):
        labels = compute_hmm_regime_labels(REPO / "data/ohlcv/SPY/1d.parquet")
        bear22 = labels[(labels.date >= "2022-06-01") & (labels.date < "2022-11-01")]
        bear_days = (bear22["regime"] == "BEAR").sum()
        # PRIME DIRECTIVE: the 2022-05-14 detector bug labeled this period
        # BULL_CALM 100% of bars. Pin ≥10 BEAR days as regression guard.
        assert bear_days >= 10, f"2022 bear only {bear_days} BEAR days — detector regression"

    def test_bull_volatile_is_dominant_in_calm_2023(self):
        """2023 was a calm bull year — should mostly be BULL_VOLATILE
        (or BULL_CALM if Hurst trends well)."""
        labels = compute_hmm_regime_labels(REPO / "data/ohlcv/SPY/1d.parquet")
        y2023 = labels[(labels.date >= "2023-01-01") & (labels.date < "2024-01-01")]
        non_bear_pct = (y2023["regime"] != "BEAR").mean()
        assert non_bear_pct > 0.9, f"only {non_bear_pct:.0%} non-BEAR in 2023 calm year"


class TestPerRegimeIC:
    """Synthetic data: known-direction IC per regime."""

    def test_per_regime_ic_recovers_signed_correlation(self):
        rng = np.random.default_rng(0)
        # 30 dates × 10 tickers; 15 dates BULL_CALM (positive corr),
        # 15 dates BEAR (negative corr)
        preds, labels_data, dates, regime_labels = [], [], [], []
        for d in range(30):
            x = rng.normal(0, 1, 10)
            if d < 15:
                y = x + rng.normal(0, 0.2, 10)  # positive corr
                regime = "BULL_CALM"
            else:
                y = -x + rng.normal(0, 0.2, 10)  # negative corr
                regime = "BEAR"
            preds.extend(x); labels_data.extend(y)
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
            dates.extend([date] * 10)
            regime_labels.append({"date": date, "regime": regime})

        preds_df = pd.DataFrame({"date": dates, "pred": preds, "label": labels_data})
        hmm = pd.DataFrame(regime_labels)
        out = per_hmm_regime_ic(preds_df, hmm, min_days_per_regime=5)
        assert "BULL_CALM" in out and out["BULL_CALM"] > 0.5
        assert "BEAR" in out and out["BEAR"] < -0.5

    def test_under_sampled_regime_excluded(self):
        rng = np.random.default_rng(0)
        preds, labels_data, dates, regime_labels = [], [], [], []
        # 10 BULL_VOLATILE days + 3 CHOPPY days
        for d in range(10):
            x = rng.normal(0, 1, 10)
            y = x + rng.normal(0, 0.1, 10)
            preds.extend(x); labels_data.extend(y)
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
            dates.extend([date] * 10)
            regime_labels.append({"date": date, "regime": "BULL_VOLATILE"})
        for d in range(10, 13):  # only 3 CHOPPY
            x = rng.normal(0, 1, 10)
            preds.extend(x); labels_data.extend(x)
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
            dates.extend([date] * 10)
            regime_labels.append({"date": date, "regime": "CHOPPY"})
        preds_df = pd.DataFrame({"date": dates, "pred": preds, "label": labels_data})
        hmm = pd.DataFrame(regime_labels)
        out = per_hmm_regime_ic(preds_df, hmm, min_days_per_regime=5)
        assert "BULL_VOLATILE" in out
        assert "CHOPPY" not in out  # excluded for under-sampling


class TestBullRegimeIc:
    """The user-mandated swap criterion."""

    def test_aggregates_only_bull_regimes(self):
        per_regime = {
            "BULL_CALM": 0.10, "BULL_VOLATILE": 0.05,
            "BEAR": -0.20, "CHOPPY": -0.05,
        }
        out = bull_regime_ic(per_regime)
        # Mean of BULL_CALM and BULL_VOLATILE only
        assert abs(out - 0.075) < 1e-9

    def test_ignores_bull_strong_label_if_present(self):
        """BULL_STRONG appears in golden config but not emitted by detector;
        if it ever appears, it should NOT contribute (detector code is
        source of truth)."""
        per_regime = {
            "BULL_CALM": 0.10, "BULL_VOLATILE": 0.05,
            "BULL_STRONG": 0.50,  # phantom label
        }
        out = bull_regime_ic(per_regime)
        # BULL_STRONG NOT mixed in
        assert abs(out - 0.075) < 1e-9

    def test_returns_nan_when_no_bull_regime(self):
        per_regime = {"BEAR": -0.10, "CHOPPY": +0.02}
        out = bull_regime_ic(per_regime)
        assert np.isnan(out)

    def test_single_bull_regime_ok(self):
        per_regime = {"BULL_VOLATILE": 0.05, "BEAR": -0.10}
        assert abs(bull_regime_ic(per_regime) - 0.05) < 1e-9


class TestSourceContracts:
    def test_thresholds_match_kernel_regime_defaults(self):
        """Hardcoded thresholds in this stateless classifier MUST match the
        stateful detector defaults in kernel/regime.py. Otherwise this
        module silently disagrees with prod regime detector."""
        prod_src = (REPO / "backtesting/renquant_104/kernel/regime.py").read_text()
        # Check the canonical default values appear in regime.py
        assert '"bear_vol_threshold",    0.35' in prod_src
        assert '"bear_return_threshold", -0.08' in prod_src
        assert '"bear_vol_threshold_5d",    0.25' in prod_src
        assert '"bear_return_threshold_5d", -0.04' in prod_src
        # Our module mirrors these
        assert BEAR_VOL_20D_THR == 0.35
        assert BEAR_RET_20D_THR == -0.08
