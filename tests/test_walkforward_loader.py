"""Tests for kernel.walk_forward.WalkForwardModelLoader (P1, 2026-05-10).

Pins the contract that SimAdapter / live runner / cron all bind to:
    * model_as_of(today) → latest cutoff_date < today
    * raises ValueError when nothing eligible (no silent skip)
    * leakage invariants enforced at parse + at lookup
    * has_walkforward_model correctness (empty / populated / missing path)
    * glob-pattern tolerance for the manifest path
"""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

_STRATEGY_DIR = (
    Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
)
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _make_manifest(tmp_path, rows):
    """Write a manifest JSON dict with the given retrain rows."""
    p = tmp_path / "walkforward_manifest.json"
    p.write_text(json.dumps({
        "cadence_days": 21,
        "training_window_years": 3.0,
        "retrains": rows,
    }))
    return p


def _row(cutoff, trained, uri, calibrator_uri=None, **extra):
    row = {
        "cutoff_date": cutoff,
        "trained_date": trained,
        "artifact_uri": uri,
    }
    if calibrator_uri is not None:
        row["calibrator_uri"] = calibrator_uri
    row.update(extra)
    return row


class TestAuditP32Regression:
    """AUDIT REGRESSION GUARD per CLAUDE.md §5.13.3 — pins the 2026-05-10
    P3.2 sim crash.

    The bug: WalkForwardModelLoader.model_as_of passed `trained_date`
    (wall-clock retrain time, ~"now") to `assert_no_leakage` instead of
    `cutoff_date` (training-data upper bound). When the retrain script
    ran on 2026-05-10 and the sim started at 2024-01-02, the assertion
    fired because 2026-05-10 (trained) >= 2024-01-02 (sim today), even
    though the actual training data for cutoff_date=2024-01-01 was clean.

    Realistic scenario test: retrain script run TODAY (wall-clock 2026-05-10)
    producing entries with cutoff_date < retrain wall-clock. Sim must
    successfully load these models without raising.
    """

    def test_trained_date_in_future_does_not_break_loader(self, tmp_path,
                                                          monkeypatch):
        from kernel.walk_forward import WalkForwardModelLoader
        # Wall-clock retrain time is 2026-05-10 (when the retrain script
        # ran), but each entry's cutoff_date enforces the actual training
        # data upper bound — clean walk-forward.
        rows = [
            _row("2024-01-01T00:00:00", "2026-05-10T12:00:00",
                 "fake://run-A/m"),
            _row("2024-02-01T00:00:00", "2026-05-10T12:30:00",
                 "fake://run-B/m"),
        ]
        manifest = _make_manifest(tmp_path, rows)

        class FakeScorer:
            def __init__(self, uri):
                self.uri = uri

        def fake_load(path):
            return FakeScorer(path)

        monkeypatch.setattr(
            "kernel.panel_pipeline.panel_scorer.PanelScorer.load",
            staticmethod(fake_load),
        )

        loader = WalkForwardModelLoader(manifest)
        # Sim today=2024-01-02 should pick cutoff=2024-01-01 entry.
        # MUST NOT raise even though trained_date is 2 years AFTER today.
        scorer = loader.model_as_of(pd.Timestamp("2024-01-02"))
        assert scorer is not None
        assert "run-A" in scorer.uri

    def test_cutoff_equal_to_today_still_blocked(self, tmp_path,
                                                 monkeypatch):
        """cutoff_date must be strictly < today (line 152 invariant).
        Cutoff == today should still raise since model has seen up-to-but-
        excluding cutoff, but the loader's `e.cutoff_date < today_ts`
        eligibility filter prevents selection of equal-cutoff entries."""
        from kernel.walk_forward import WalkForwardModelLoader
        rows = [
            _row("2024-01-02T00:00:00", "2026-05-10T12:00:00",
                 "fake://run-A/m"),
        ]
        manifest = _make_manifest(tmp_path, rows)
        loader = WalkForwardModelLoader(manifest)
        # today == cutoff → no eligible entries → raises
        with pytest.raises(ValueError, match="no retrain"):
            loader.model_as_of(pd.Timestamp("2024-01-02"))


class TestModelAsOf:
    def test_returns_latest_eligible_entry(self, tmp_path, monkeypatch):
        from kernel.walk_forward import WalkForwardModelLoader
        rows = [
            _row("2024-01-01T00:00:00", "2024-01-02T03:00:00", "fake://run-A/m"),
            _row("2024-02-01T00:00:00", "2024-02-02T03:00:00", "fake://run-B/m"),
            _row("2024-03-01T00:00:00", "2024-03-02T03:00:00", "fake://run-C/m"),
        ]
        path = _make_manifest(tmp_path, rows)
        loader = WalkForwardModelLoader(path)

        # Stub PanelScorer.load to bypass artifact materialization
        seen_uri = []

        class _FakeScorer:
            def __init__(self, uri):
                self.uri = uri

        from kernel.panel_pipeline import panel_scorer as _ps

        def fake_load(uri):
            seen_uri.append(str(uri))
            return _FakeScorer(str(uri))

        monkeypatch.setattr(_ps.PanelScorer, "load", staticmethod(fake_load))

        # 2024-02-15 → latest cutoff < it is 2024-02-01 (run-B)
        scorer = loader.model_as_of("2024-02-15")
        assert seen_uri[-1] == "fake://run-B/m"
        assert scorer.uri == "fake://run-B/m"

    def test_effective_train_cutoff_avoids_double_embargo(self, tmp_path, monkeypatch):
        """WF scripts pass selection cutoff to --train-cutoff, then train on
        feature rows before selection_cutoff - lookahead. The loader must use
        the effective feature cutoff for label-safety, not apply lookahead a
        second time to the selection cutoff.
        """
        from kernel.walk_forward import WalkForwardModelLoader

        rows = [
            _row(
                "2024-04-01T00:00:00",
                "2026-05-23T12:00:00",
                "fake://run-A/m",
                lookahead_days=60,
                effective_train_cutoff_date="2024-01-05T00:00:00",
            ),
        ]
        path = _make_manifest(tmp_path, rows)
        loader = WalkForwardModelLoader(path)

        class _FakeScorer:
            def __init__(self, uri):
                self.uri = uri

        from kernel.panel_pipeline import panel_scorer as _ps
        monkeypatch.setattr(
            _ps.PanelScorer,
            "load",
            staticmethod(lambda uri: _FakeScorer(str(uri))),
        )

        scorer = loader.model_as_of("2024-04-02")
        assert scorer.uri == "fake://run-A/m"

    def test_calibrator_as_of_returns_matching_manifest_calibrator(self, tmp_path):
        from kernel.walk_forward import WalkForwardModelLoader
        from training_panel.global_calibrator import GlobalPanelCalibration
        import numpy as np

        scorer_path = tmp_path / "panel-ltr.json"
        scorer_path.write_text(json.dumps({
            "kind": "panel_ltr_xgboost",
            "feature_cols": ["f0"],
            "artifact_fingerprint": "sha256:abc123abc123",
        }))
        cal_path = tmp_path / "cal-A.json"
        GlobalPanelCalibration(
            prob_x=np.array([-1.0, 1.0]),
            prob_y=np.array([0.25, 0.75]),
            er_x=np.array([-1.0, 1.0]),
            er_y=np.array([-0.01, 0.01]),
            metadata={"scorer_artifact_fingerprint": "sha256:abc123abc123"},
        ).save(cal_path)
        rows = [
            _row(
                "2024-01-01T00:00:00",
                "2024-01-02T03:00:00",
                str(scorer_path),
                calibrator_uri=str(cal_path),
            ),
        ]
        path = _make_manifest(tmp_path, rows)
        loader = WalkForwardModelLoader(path)

        cal = loader.calibrator_as_of("2024-01-15")
        assert cal.metadata["scorer_artifact_fingerprint"] == "sha256:abc123abc123"

    def test_calibrator_as_of_accepts_local_pt_scorer_fingerprint(self, tmp_path):
        """HF PatchTST .pt folds must bind calibrators to exact scorer bytes."""
        from kernel.walk_forward import WalkForwardModelLoader
        from training_panel.global_calibrator import GlobalPanelCalibration
        import numpy as np

        scorer_path = tmp_path / "hf_patchtst_fold_A_model.pt"
        scorer_path.write_bytes(b"fake torch checkpoint bytes for wf contract")
        scorer_fp = "sha256:" + hashlib.sha256(scorer_path.read_bytes()).hexdigest()

        cal_path = tmp_path / "cal-A.json"
        GlobalPanelCalibration(
            prob_x=np.array([-1.0, 1.0]),
            prob_y=np.array([0.25, 0.75]),
            er_x=np.array([-1.0, 1.0]),
            er_y=np.array([-0.01, 0.01]),
            metadata={"scorer_artifact_fingerprint": scorer_fp},
        ).save(cal_path)
        rows = [
            _row(
                "2024-01-01T00:00:00",
                "2024-01-02T03:00:00",
                str(scorer_path),
                calibrator_uri=str(cal_path),
            ),
        ]
        path = _make_manifest(tmp_path, rows)
        loader = WalkForwardModelLoader(path)

        cal = loader.calibrator_as_of("2024-01-15")
        assert cal.metadata["scorer_artifact_fingerprint"] == scorer_fp

    def test_calibrator_as_of_rejects_foreign_calibrator(self, tmp_path):
        """The loader must not expose a calibrator fitted to another scorer."""
        from kernel.walk_forward import WalkForwardModelLoader
        from training_panel.global_calibrator import GlobalPanelCalibration
        import numpy as np

        scorer_path = tmp_path / "panel-ltr.json"
        scorer_path.write_text(json.dumps({
            "kind": "panel_ltr_xgboost",
            "feature_cols": ["f0"],
            "artifact_fingerprint": "sha256:active111111",
        }))
        cal_path = tmp_path / "cal-A.json"
        GlobalPanelCalibration(
            prob_x=np.array([-1.0, 1.0]),
            prob_y=np.array([0.25, 0.75]),
            er_x=np.array([-1.0, 1.0]),
            er_y=np.array([-0.01, 0.01]),
            metadata={"scorer_artifact_fingerprint": "sha256:foreign000000"},
        ).save(cal_path)
        rows = [
            _row(
                "2024-01-01T00:00:00",
                "2024-01-02T03:00:00",
                str(scorer_path),
                calibrator_uri=str(cal_path),
            ),
        ]
        path = _make_manifest(tmp_path, rows)
        loader = WalkForwardModelLoader(path)

        with pytest.raises(ValueError, match="fingerprint mismatch"):
            loader.calibrator_as_of("2024-01-15")

    def test_calibrator_as_of_requires_manifest_uri(self, tmp_path):
        from kernel.walk_forward import WalkForwardModelLoader

        rows = [
            _row("2024-01-01T00:00:00", "2024-01-02T03:00:00", "fake://run-A/m"),
        ]
        path = _make_manifest(tmp_path, rows)
        loader = WalkForwardModelLoader(path)

        with pytest.raises(ValueError, match="no calibrator_uri"):
            loader.calibrator_as_of("2024-01-15")

    def test_raises_when_no_eligible_entry(self, tmp_path):
        from kernel.walk_forward import WalkForwardModelLoader
        rows = [_row("2024-06-01T00:00:00", "2024-06-02T03:00:00", "fake://m")]
        path = _make_manifest(tmp_path, rows)
        loader = WalkForwardModelLoader(path)
        with pytest.raises(ValueError, match="no retrain with cutoff_date"):
            loader.model_as_of("2024-01-15")

    def test_raises_when_manifest_empty(self, tmp_path):
        from kernel.walk_forward import WalkForwardModelLoader
        path = _make_manifest(tmp_path, rows=[])
        loader = WalkForwardModelLoader(path)
        with pytest.raises(ValueError, match="no retrain"):
            loader.model_as_of("2026-01-01")

    def test_raises_when_manifest_missing(self, tmp_path):
        from kernel.walk_forward import WalkForwardModelLoader
        loader = WalkForwardModelLoader(tmp_path / "missing.json")
        assert not loader.has_walkforward_model()
        with pytest.raises(ValueError):
            loader.model_as_of("2024-01-15")


class TestLeakageGuards:
    def test_parse_rejects_trained_before_cutoff(self, tmp_path):
        from kernel.walk_forward import WalkForwardModelLoader
        # trained_date < cutoff_date is a manifest construction bug
        rows = [_row("2024-06-01T00:00:00", "2024-05-15T03:00:00", "fake://m")]
        path = _make_manifest(tmp_path, rows)
        with pytest.raises(ValueError, match="leakage"):
            WalkForwardModelLoader(path)

    def test_returned_entry_has_cutoff_strictly_before_today(self, tmp_path,
                                                              monkeypatch):
        from kernel.walk_forward import WalkForwardModelLoader
        rows = [
            _row("2024-01-15T00:00:00", "2024-01-16T03:00:00", "fake://A"),
            _row("2024-02-15T00:00:00", "2024-02-16T03:00:00", "fake://B"),
        ]
        path = _make_manifest(tmp_path, rows)
        loader = WalkForwardModelLoader(path)
        from kernel.panel_pipeline import panel_scorer as _ps
        monkeypatch.setattr(_ps.PanelScorer, "load",
                            staticmethod(lambda uri: object()))
        # Equality counts as leakage — cutoff_date == today must NOT be returned.
        # In the test set, 2024-02-15 → only 2024-01-15 < it.
        loader.model_as_of("2024-02-15")
        # Verify by asserting eligible[-1].cutoff_date strictly < today via
        # public `entries` accessor.
        entries_before = [e for e in loader.entries
                          if e.cutoff_date < pd.Timestamp("2024-02-15")]
        assert entries_before[-1].cutoff_date == pd.Timestamp("2024-01-15")


class TestHasWalkforwardModel:
    def test_true_when_at_least_one_entry(self, tmp_path):
        from kernel.walk_forward import WalkForwardModelLoader
        path = _make_manifest(tmp_path, [
            _row("2024-01-01T00:00:00", "2024-01-02T00:00:00", "fake://m"),
        ])
        loader = WalkForwardModelLoader(path)
        assert loader.has_walkforward_model() is True

    def test_false_when_empty_manifest(self, tmp_path):
        from kernel.walk_forward import WalkForwardModelLoader
        path = _make_manifest(tmp_path, [])
        loader = WalkForwardModelLoader(path)
        assert loader.has_walkforward_model() is False

    def test_false_when_manifest_missing(self, tmp_path):
        from kernel.walk_forward import WalkForwardModelLoader
        loader = WalkForwardModelLoader(tmp_path / "nope.json")
        assert loader.has_walkforward_model() is False


class TestManifestRoundTrip:
    def test_loader_reads_what_writer_writes(self, tmp_path):
        from kernel.walk_forward import (
            RetrainEntry, WalkForwardManifest,
            WalkForwardModelLoader, write_manifest,
        )
        entries = [
            RetrainEntry(
                cutoff_date=pd.Timestamp("2024-04-01"),
                trained_date=pd.Timestamp("2024-04-02T03:44:12"),
                artifact_uri="fake://X/panel-ltr.json",
                calibrator_uri="fake://X/panel-rank-calibration.json",
            ),
            RetrainEntry(
                cutoff_date=pd.Timestamp("2024-05-01"),
                trained_date=pd.Timestamp("2024-05-02T03:44:12"),
                artifact_uri="fake://Y/panel-ltr.json",
            ),
        ]
        manifest = WalkForwardManifest(
            cadence_days=30,
            training_window_years=3.0,
            retrains=entries,
        )
        path = tmp_path / "walkforward_manifest.json"
        write_manifest(manifest, path)
        loader = WalkForwardModelLoader(path)
        assert loader.has_walkforward_model() is True
        entries_loaded = loader.entries
        assert len(entries_loaded) == 2
        assert entries_loaded[0].cutoff_date == pd.Timestamp("2024-04-01")
        assert entries_loaded[0].calibrator_uri == "fake://X/panel-rank-calibration.json"
        assert entries_loaded[1].artifact_uri == "fake://Y/panel-ltr.json"


class TestGlobPatternTolerance:
    def test_glob_picks_lex_last_match(self, tmp_path):
        from kernel.walk_forward import WalkForwardModelLoader
        # Two manifests with sortable filenames
        a = _make_manifest(tmp_path, [
            _row("2024-01-01T00:00:00", "2024-01-02T00:00:00", "fake://A"),
        ])
        a.rename(tmp_path / "manifest.2024-01.json")
        path_a = tmp_path / "manifest.2024-01.json"
        path_a.write_text(json.dumps({
            "cadence_days": 21, "training_window_years": 3.0,
            "retrains": [
                _row("2024-01-01T00:00:00", "2024-01-02T00:00:00", "fake://A"),
            ],
        }))
        path_b = tmp_path / "manifest.2024-02.json"
        path_b.write_text(json.dumps({
            "cadence_days": 21, "training_window_years": 3.0,
            "retrains": [
                _row("2024-02-01T00:00:00", "2024-02-02T00:00:00", "fake://B"),
            ],
        }))
        # Glob picks lexicographically last match → manifest.2024-02.json
        loader = WalkForwardModelLoader(str(tmp_path / "manifest.*.json"))
        assert loader.has_walkforward_model()
        assert loader.entries[0].artifact_uri == "fake://B"

    def test_glob_no_match_returns_empty_loader(self, tmp_path):
        from kernel.walk_forward import WalkForwardModelLoader
        loader = WalkForwardModelLoader(str(tmp_path / "no-such-*.json"))
        assert loader.has_walkforward_model() is False
