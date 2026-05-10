"""Tests for kernel.walk_forward.manifest (P1, 2026-05-10).

Pins:
    * schema validation rejects malformed entries
    * datetime ISO round-trip preserves cutoff/trained timestamps
    * read_manifest sorts by cutoff_date ascending
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_STRATEGY_DIR = (
    Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
)
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


class TestSchemaValidation:
    def test_rejects_missing_keys(self, tmp_path):
        from kernel.walk_forward.manifest import read_manifest
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "cadence_days": 21,
            "training_window_years": 3.0,
            "retrains": [
                {"cutoff_date": "2024-01-01T00:00:00",
                 "artifact_uri": "fake://m"},   # missing trained_date
            ],
        }))
        with pytest.raises(ValueError, match="missing key"):
            read_manifest(path)

    def test_rejects_empty_artifact_uri(self, tmp_path):
        from kernel.walk_forward.manifest import read_manifest
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "cadence_days": 21,
            "training_window_years": 3.0,
            "retrains": [
                {"cutoff_date": "2024-01-01T00:00:00",
                 "trained_date": "2024-01-02T00:00:00",
                 "artifact_uri": ""},
            ],
        }))
        with pytest.raises(ValueError, match="empty artifact_uri"):
            read_manifest(path)

    def test_rejects_trained_before_cutoff(self, tmp_path):
        from kernel.walk_forward.manifest import read_manifest
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "cadence_days": 21,
            "training_window_years": 3.0,
            "retrains": [
                {"cutoff_date": "2024-01-01T00:00:00",
                 "trained_date": "2023-12-31T23:59:59",
                 "artifact_uri": "fake://m"},
            ],
        }))
        with pytest.raises(ValueError, match="leakage"):
            read_manifest(path)


class TestRoundTrip:
    def test_iso_datetime_preserved(self, tmp_path):
        from kernel.walk_forward import (
            RetrainEntry, WalkForwardManifest,
            read_manifest, write_manifest,
        )
        entry = RetrainEntry(
            cutoff_date=pd.Timestamp("2024-04-01T00:00:00"),
            trained_date=pd.Timestamp("2024-04-02T03:44:12"),
            artifact_uri="fake://X/panel-ltr.json",
        )
        manifest = WalkForwardManifest(
            cadence_days=21, training_window_years=3.0, retrains=[entry],
        )
        out = tmp_path / "wf.json"
        write_manifest(manifest, out)
        loaded = read_manifest(out)
        assert loaded.cadence_days == 21
        assert loaded.training_window_years == 3.0
        assert loaded.retrains[0].cutoff_date == entry.cutoff_date
        assert loaded.retrains[0].trained_date == entry.trained_date
        assert loaded.retrains[0].artifact_uri == entry.artifact_uri

    def test_write_creates_parent_dir(self, tmp_path):
        from kernel.walk_forward import (
            WalkForwardManifest, read_manifest, write_manifest,
        )
        manifest = WalkForwardManifest(
            cadence_days=21, training_window_years=3.0, retrains=[],
        )
        nested = tmp_path / "a" / "b" / "c" / "wf.json"
        write_manifest(manifest, nested)
        assert nested.exists()
        assert read_manifest(nested).cadence_days == 21


class TestSortedByCutoff:
    def test_read_sorts_ascending(self, tmp_path):
        from kernel.walk_forward.manifest import read_manifest
        path = tmp_path / "wf.json"
        # Insert OUT OF ORDER on disk
        path.write_text(json.dumps({
            "cadence_days": 21, "training_window_years": 3.0,
            "retrains": [
                {"cutoff_date": "2024-03-01T00:00:00",
                 "trained_date": "2024-03-02T00:00:00",
                 "artifact_uri": "fake://C"},
                {"cutoff_date": "2024-01-01T00:00:00",
                 "trained_date": "2024-01-02T00:00:00",
                 "artifact_uri": "fake://A"},
                {"cutoff_date": "2024-02-01T00:00:00",
                 "trained_date": "2024-02-02T00:00:00",
                 "artifact_uri": "fake://B"},
            ],
        }))
        m = read_manifest(path)
        cutoffs = [e.cutoff_date for e in m.retrains]
        assert cutoffs == sorted(cutoffs)
        assert m.retrains[0].artifact_uri == "fake://A"
        assert m.retrains[-1].artifact_uri == "fake://C"
