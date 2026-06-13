"""sim.py decomposition slice 4 — point-in-time leakage guard tests."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from adapters.sim_leakage_guards import (  # noqa: E402
    corr_leakage_present,
    gmm_leakage_present,
)


class TestCorrLeakage:
    def test_as_of_after_start_is_leakage(self):
        assert corr_leakage_present("2026-06-10", "2024-01-01") is True

    def test_as_of_before_start_clean(self):
        assert corr_leakage_present("2023-12-31", "2024-01-01") is False

    def test_none_inputs_clean(self):
        assert corr_leakage_present(None, "2024-01-01") is False
        assert corr_leakage_present("2026-01-01", None) is False

    def test_unparseable_clean(self):
        assert corr_leakage_present("garbage", "2024-01-01") is False

    def test_tz_aware_normalized(self):
        assert corr_leakage_present("2026-06-10T00:00:00+00:00", "2024-01-01") is True


class TestGmmLeakage:
    def test_extractor_as_of_after_start(self):
        art = {"as_of_date": "2026-06-10"}
        assert gmm_leakage_present(art, "2024-01-01", lambda a: a["as_of_date"]) is True

    def test_legacy_unstamped_clean(self):
        assert gmm_leakage_present({}, "2024-01-01", lambda a: a.get("as_of_date")) is False

    def test_none_artifact_clean(self):
        assert gmm_leakage_present(None, "2024-01-01", lambda a: None) is False
