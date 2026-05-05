"""Model-layer acceptance tests — load production artifacts + verify.

Runs against `backtesting/renquant_104/artifacts/*.json`. If an artifact
is missing (e.g. fresh checkout pre-train), the relevant test is
skipped, not failed — operator can still run the suite to check
whatever IS on disk.

Each test verifies:
  1. SCHEMA — required keys present, types correct
  2. MEANINGFUL DATA — values within physically plausible ranges,
     no constant outputs, no NaN-leaf collapse pattern
  3. CROSS-ARTIFACT alignment when multiple artifacts are present

User mandate (2026-05-04): every output artifact has acceptance tests.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO / "backtesting" / "renquant_104" / "artifacts"

sys.path.insert(0, str(REPO / "tests"))
from acceptance import protocol as P  # noqa: E402


# ── panel-LTR ────────────────────────────────────────────────────────────────

class TestPanelLTRArtifact:
    @property
    def path(self) -> Path:
        return ARTIFACTS / "panel-ltr.json"

    def test_artifact_exists_and_is_well_formed(self):
        if not self.path.exists():
            pytest.skip(f"{self.path.name} not present — skip")
        P.assert_panel_ltr_artifact(self.path)

    def test_meaningful_oos_ic(self):
        if not self.path.exists():
            pytest.skip("artifact missing")
        payload = json.loads(self.path.read_text())
        P.assert_panel_ltr_meaningful(payload)

    def test_feature_cols_nontrivial(self):
        """A real panel-LTR has ≥10 features. Single-feature artifacts
        are debug stubs and should not ship."""
        if not self.path.exists():
            pytest.skip("artifact missing")
        payload = json.loads(self.path.read_text())
        fc = payload.get("feature_cols", [])
        # transformer shim path may have shorter feature lists
        if payload.get("kind") == "panel_transformer":
            pytest.skip("transformer shim — feature_cols spec differs")
        assert len(fc) >= 10, \
            f"panel-ltr feature_cols={len(fc)} suspiciously low; debug stub?"


# ── Calibrator (the NaN-leaf collapse target) ───────────────────────────────

class TestPanelCalibratorArtifact:
    @property
    def path(self) -> Path:
        return ARTIFACTS / "panel-rank-calibration.json"

    def test_artifact_exists_and_is_well_formed(self):
        if not self.path.exists():
            pytest.skip("calibrator artifact not present — skip")
        P.assert_calibrator_artifact(self.path)

    def test_calibrator_not_collapsed(self):
        """The 2026-05-04 NaN-leaf incident: calibrator collapsed to
        ~constant output because >50% of training rows had all-NaN
        intraday features. This test catches that class of failure."""
        if not self.path.exists():
            pytest.skip("artifact missing")
        payload = json.loads(self.path.read_text())
        P.assert_calibrator_meaningful(payload)


# ── Data-scan preflight report ──────────────────────────────────────────────

class TestDataScanReportArtifact:
    @property
    def path(self) -> Path:
        return ARTIFACTS / "training_data_scan.json"

    def test_report_exists_and_is_well_formed(self):
        if not self.path.exists():
            pytest.skip("data_scan report not present (training hasn't run since opt-in)")
        P.assert_data_scan_report(self.path)

    def test_daily_ohlcv_coverage_above_threshold(self):
        """Sanity: production daily OHLCV coverage must be ≥80%. Below
        that and the panel has too many missing-row tickers."""
        if not self.path.exists():
            pytest.skip("report missing")
        payload = json.loads(self.path.read_text())
        cov = payload["alignment"]["watchlist_coverage_pct"]
        assert cov >= 0.80, (
            f"daily OHLCV watchlist coverage {cov:.1%} below 80% — "
            f"too many tickers missing daily history. Refresh "
            f"`data/ohlcv/` and re-train."
        )


# ── Per-ticker policy artifacts (a sample of 5 random tickers) ──────────────

class TestPerTickerPolicies:
    @property
    def models_dir(self) -> Path:
        return REPO / "backtesting" / "renquant_104" / "models"

    def _sample_tickers(self, n: int = 5) -> list[str]:
        if not self.models_dir.exists():
            return []
        # Pick 5 deterministic tickers from sorted dirs (reproducible).
        all_dirs = sorted(d.name for d in self.models_dir.iterdir() if d.is_dir())
        if not all_dirs:
            return []
        # Take first, last, and 3 evenly spaced
        if len(all_dirs) <= n:
            return all_dirs
        step = len(all_dirs) // n
        return [all_dirs[i * step] for i in range(n)]

    def test_sample_policies_well_formed(self):
        tickers = self._sample_tickers()
        if not tickers:
            pytest.skip("models/ directory empty or missing")
        for t in tickers:
            policy_path = self.models_dir / t / f"{t}-policy-metadata.json"
            if not policy_path.exists():
                continue
            P.assert_per_ticker_policy(policy_path)

    def test_sample_policies_meaningful(self):
        tickers = self._sample_tickers()
        if not tickers:
            pytest.skip("no policies")
        for t in tickers:
            policy_path = self.models_dir / t / f"{t}-policy-metadata.json"
            if not policy_path.exists():
                continue
            payload = json.loads(policy_path.read_text())
            P.assert_per_ticker_policy_meaningful(payload, label=t)


# ── Cross-artifact alignment ────────────────────────────────────────────────

class TestCrossArtifactAlignment:
    def test_panel_ltr_calibrator_ngboost_aligned(self):
        panel_path = ARTIFACTS / "panel-ltr.json"
        ngb_path   = ARTIFACTS / "ngboost-head.json"
        cal_path   = ARTIFACTS / "panel-rank-calibration.json"
        if not panel_path.exists():
            pytest.skip("panel-ltr missing")
        if json.loads(panel_path.read_text()).get("kind") == "panel_transformer":
            pytest.skip("transformer shim — alignment spec differs")
        panel = json.loads(panel_path.read_text())
        ngb   = json.loads(ngb_path.read_text()) if ngb_path.exists() else None
        cal   = json.loads(cal_path.read_text()) if cal_path.exists() else None
        P.assert_cross_artifact_consistency(panel, cal, ngb)
