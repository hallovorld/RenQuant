"""runner.py make_context decomposition — load_context_artifacts tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from adapters.runner_artifacts import load_context_artifacts  # noqa: E402


class TestLoadContextArtifacts:
    def test_missing_artifacts_dir_returns_nones_gracefully(self, tmp_path):
        # No artifacts dir → gmm loader handles missing; corr/earnings None.
        gmm, corr, earnings = load_context_artifacts(tmp_path, {"regime": {}})
        assert corr is None and earnings is None

    def test_malformed_corr_treated_as_missing(self, tmp_path):
        ad = tmp_path / "artifacts" / "prod"
        ad.mkdir(parents=True)
        (ad.parent / "prod" / "watchlist-correlation.json").write_text("{bad json")
        (tmp_path / "artifacts" / "prod" / "earnings-calendar.json").write_text("{}")
        gmm, corr, earnings = load_context_artifacts(
            tmp_path, {"regime": {"gmm_artifact": "prod/none.json"}})
        assert corr is None  # malformed → None, no crash

    def test_malformed_earnings_treated_as_missing(self, tmp_path):
        ad = tmp_path / "artifacts" / "prod"
        ad.mkdir(parents=True)
        (ad / "earnings-calendar.json").write_text("{not valid")
        gmm, corr, earnings = load_context_artifacts(
            tmp_path, {"regime": {"gmm_artifact": "prod/none.json",
                                  "correlation_artifact": "prod/none.json"}})
        assert earnings is None

    def test_valid_earnings_loaded(self, tmp_path):
        ad = tmp_path / "artifacts" / "prod"
        ad.mkdir(parents=True)
        (ad / "earnings-calendar.json").write_text(json.dumps({"MU": ["2026-07-01"]}))
        _, _, earnings = load_context_artifacts(
            tmp_path, {"regime": {"gmm_artifact": "prod/none.json",
                                  "correlation_artifact": "prod/none.json",
                                  "earnings_artifact": "prod/earnings-calendar.json"}})
        assert earnings == {"MU": ["2026-07-01"]}
