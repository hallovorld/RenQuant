"""Campaign B2 pins: the umbrella WF gate leg verifies via the M6 dispatch.

RQ#444 F-2/F-10 + orchestrator#296 BT-1: the umbrella kernel loader (the
LIVE promote-gate leg, ``run_wf_gate.py`` + ``adapters/sim.py``) used to
carry its own 12-char-prefix fingerprint matcher plus a recompute imported
from the umbrella panel_scorer's STALE local ``model_content_sha256`` copy
— one of three divergent verifiers of the one WF-stamp contract (the
2026-05-27 / 06-22 / 07-01 stamp-mismatch incident generator). These tests
pin the repoint:

1. no local matcher fork — verification helpers are IMPORTS ONLY from
   ``renquant_pipeline.kernel.panel_pipeline.fingerprint_dispatch``;
2. incident-fixture regressions — a mismatched stamp still fails closed; a
   legacy-stamped fold still passes via the dispatch legacy route; a
   corrupt v1 stamp fails closed regardless of any flag;
3. the historical 12-char prefix acceptance survives ONLY behind the
   ``accept_legacy_stamps`` migration-window flag (default ON; measured
   2026-07-04: ZERO currently-green artifacts rely on it);
4. the two umbrella stamping scripts (``fit_calibrator_alpha158_fund`` /
   ``stamp_walkforward_fingerprints``) resolve identity stamped-value-first
   with the EXPLICIT legacy engine as fallback — never the venv-coupled
   bare name (the pipeline#160 hazard), never the umbrella kernel copy.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO / "backtesting" / "renquant_104"
for _p in (_REPO, _STRATEGY_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import kernel.walk_forward.loader as u_loader  # noqa: E402
from kernel.walk_forward.loader import WalkForwardModelLoader  # noqa: E402
from training_panel.global_calibrator import GlobalPanelCalibration  # noqa: E402

fingerprint_dispatch = pytest.importorskip(
    "renquant_pipeline.kernel.panel_pipeline.fingerprint_dispatch",
    reason="pinned renquant-pipeline (>= M6 stage-1 dispatch) not on sys.path",
)

_LEGACY_STAMP = "sha256:" + "a1" * 32


def _write_manifest(tmp_path, scorer_path, cal_path):
    p = tmp_path / "walkforward_manifest.json"
    p.write_text(json.dumps({
        "retrains": [{
            "cutoff_date": "2024-01-01T00:00:00",
            "trained_date": "2024-01-02T03:00:00",
            "artifact_uri": str(scorer_path),
            "calibrator_uri": str(cal_path),
        }],
    }))
    return p


def _write_calibrator(cal_path, metadata):
    GlobalPanelCalibration(
        prob_x=np.array([-1.0, 1.0]),
        prob_y=np.array([0.25, 0.75]),
        er_x=np.array([-1.0, 1.0]),
        er_y=np.array([-0.01, 0.01]),
        metadata=metadata,
    ).save(cal_path)


def _write_stamped_scorer(tmp_path, stamp=_LEGACY_STAMP, **extra):
    scorer_path = tmp_path / "panel-ltr.json"
    payload = {
        "kind": "panel_ltr_xgboost",
        "feature_cols": ["f0"],
        "model_content_fingerprint": stamp,
    }
    payload.update(extra)
    scorer_path.write_text(json.dumps(payload))
    return scorer_path


class TestNoLocalMatcherFork:
    """Re-fork guard: the umbrella loader carries no matcher of its own."""

    def test_local_matcher_names_are_gone(self):
        for name in (
            "_fingerprints_match",
            "_any_fingerprints_match",
            "_normalize_fingerprint",
            "_scorer_fingerprints_from_payload",
            "_calibrator_scorer_fingerprints",
        ):
            assert not hasattr(u_loader, name), (
                f"kernel.walk_forward.loader.{name} exists again — a local "
                "matcher re-fork (campaign B2 removed it; verification is "
                "IMPORTS ONLY from the pipeline fingerprint dispatch)"
            )

    def test_dispatch_resolver_returns_the_pipeline_module(self):
        assert u_loader._fingerprint_dispatch() is fingerprint_dispatch

    def test_no_stale_panel_scorer_import_in_verification(self):
        source = Path(u_loader.__file__).read_text()
        assert "panel_pipeline.panel_scorer import model_content_sha256" not in source, (
            "the loader re-imported the umbrella panel_scorer's stale local "
            "model_content_sha256 copy (RQ#444 F-10)"
        )


class TestIncidentFixtureRegressions:
    """The 05-27/06-22/07-01 incident shapes, pinned on the LIVE gate leg."""

    def test_mismatched_stamp_still_fails_closed(self, tmp_path):
        scorer_path = _write_stamped_scorer(tmp_path)
        cal_path = tmp_path / "cal.json"
        _write_calibrator(cal_path, {
            "scorer_model_content_fingerprint": "sha256:" + "0f" * 32,
        })
        loader = WalkForwardModelLoader(
            _write_manifest(tmp_path, scorer_path, cal_path),
        )
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            loader.calibrator_as_of("2024-01-15")

    def test_legacy_stamped_fold_still_passes_via_dispatch(self, tmp_path):
        scorer_path = _write_stamped_scorer(tmp_path)
        cal_path = tmp_path / "cal.json"
        _write_calibrator(cal_path, {
            "scorer_model_content_fingerprint": _LEGACY_STAMP,
        })
        loader = WalkForwardModelLoader(
            _write_manifest(tmp_path, scorer_path, cal_path),
        )
        cal = loader.calibrator_as_of("2024-01-15")
        assert cal.metadata["scorer_model_content_fingerprint"] == _LEGACY_STAMP

    def test_corrupt_v1_stamp_fails_closed_at_fold_read(self, tmp_path):
        scorer_path = _write_stamped_scorer(
            tmp_path,
            stamp="sha256:" + "d0" * 32,
            booster_raw_json="{\"learner\":{}}",
            fingerprint_schema_version=1,
        )
        cal_path = tmp_path / "cal.json"
        _write_calibrator(cal_path, {
            "scorer_model_content_fingerprint": "sha256:" + "d0" * 32,
            "scorer_fingerprint_schema_version": 1,
        })
        loader = WalkForwardModelLoader(
            _write_manifest(tmp_path, scorer_path, cal_path),
        )
        with pytest.raises(ValueError):
            loader.calibrator_as_of("2024-01-15")


class TestPrefixAcceptanceIsFlagGoverned:
    """12-char prefixes: ON behind the migration-window flag, retired by
    ``accept_legacy_stamps=False`` (M6 stage-2 step 4)."""

    def _prefix_fixture(self, tmp_path):
        scorer_path = _write_stamped_scorer(tmp_path)
        cal_path = tmp_path / "cal.json"
        _write_calibrator(cal_path, {
            "scorer_model_content_fingerprint":
                _LEGACY_STAMP[:len("sha256:") + 12],
        })
        return _write_manifest(tmp_path, scorer_path, cal_path)

    def test_prefix_accepted_while_flag_on(self, tmp_path):
        loader = WalkForwardModelLoader(self._prefix_fixture(tmp_path))
        assert loader.calibrator_as_of("2024-01-15") is not None

    def test_flag_off_retires_versionless_stamps_and_prefixes(self, tmp_path):
        loader = WalkForwardModelLoader(
            self._prefix_fixture(tmp_path), accept_legacy_stamps=False,
        )
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            loader.calibrator_as_of("2024-01-15")


def _load_script(mod_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, _REPO / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestStampingScriptsUseCanonicalIdentity:
    """B2 sites 2+3: stamped-value precedence + explicit legacy fallback."""

    def test_scorer_identity_prefers_the_stamp(self, tmp_path):
        stamp_mod = _load_script(
            "b2_stamp_wf", "scripts/stamp_walkforward_fingerprints.py",
        )
        p = _write_stamped_scorer(tmp_path)
        content_fp, file_fp, schema = stamp_mod._scorer_identity(p)
        assert content_fp == _LEGACY_STAMP
        assert file_fp.startswith("sha256:")
        assert schema is None

    def test_scorer_identity_unstamped_falls_back_to_legacy_engine(
        self, tmp_path,
    ):
        from renquant_common.model_fingerprint import (
            _legacy_model_content_sha256,
        )
        stamp_mod = _load_script(
            "b2_stamp_wf", "scripts/stamp_walkforward_fingerprints.py",
        )
        p = tmp_path / "panel-ltr.json"
        payload = {"kind": "panel_ltr_xgboost", "feature_cols": ["f0"],
                   "booster_raw_json": "{}"}
        p.write_text(json.dumps(payload))
        content_fp, _file_fp, schema = stamp_mod._scorer_identity(p)
        assert content_fp == _legacy_model_content_sha256(payload)
        assert schema is None

    def test_calibrator_binding_propagates_v1_schema_version(self, tmp_path):
        stamp_mod = _load_script(
            "b2_stamp_wf", "scripts/stamp_walkforward_fingerprints.py",
        )
        scorer = tmp_path / "panel-ltr.json"
        scorer.write_text(json.dumps({
            "kind": "panel_ltr_xgboost",
            "feature_cols": ["f0"],
            "model_content_fingerprint": "sha256:" + "e2" * 32,
            "fingerprint_schema_version": 1,
        }))
        cal = tmp_path / "cal.json"
        cal.write_text(json.dumps({"metadata": {}}))
        changed = stamp_mod._stamp_calibrator_binding(
            scorer_path=scorer, calibrator_path=cal, dry_run=False,
        )
        assert changed
        meta = json.loads(cal.read_text())["metadata"]
        assert meta["scorer_fingerprint_schema_version"] == 1
        assert meta["scorer_model_content_fingerprint"] == "sha256:" + "e2" * 32

    def test_calibrator_binding_stays_versionless_for_legacy_folds(
        self, tmp_path,
    ):
        stamp_mod = _load_script(
            "b2_stamp_wf", "scripts/stamp_walkforward_fingerprints.py",
        )
        scorer = _write_stamped_scorer(tmp_path)
        cal = tmp_path / "cal.json"
        cal.write_text(json.dumps({"metadata": {}}))
        stamp_mod._stamp_calibrator_binding(
            scorer_path=scorer, calibrator_path=cal, dry_run=False,
        )
        meta = json.loads(cal.read_text())["metadata"]
        assert "scorer_fingerprint_schema_version" not in meta
        assert meta["scorer_model_content_fingerprint"] == _LEGACY_STAMP

    def test_fit_calibrator_fingerprint_is_stamp_first_legacy_fallback(
        self, tmp_path,
    ):
        from renquant_common.model_fingerprint import (
            _legacy_model_content_sha256,
        )
        fit_mod = _load_script(
            "b2_fit_calib", "scripts/fit_calibrator_alpha158_fund.py",
        )
        # Stamped: the stamp wins.
        p = _write_stamped_scorer(tmp_path)
        assert fit_mod._artifact_fingerprint(
            p, json.loads(p.read_text()),
        ) == _LEGACY_STAMP
        # Unstamped: the EXPLICIT legacy engine (identical to the retired
        # umbrella panel_scorer copy — tables proven equal), never v1.
        payload = {"kind": "panel_ltr_xgboost", "feature_cols": ["f0"],
                   "booster_raw_json": "{}"}
        q = tmp_path / "unstamped.json"
        q.write_text(json.dumps(payload))
        assert fit_mod._artifact_fingerprint(q, payload) == (
            _legacy_model_content_sha256(payload)
        )

    def test_scripts_do_not_import_the_stale_kernel_copy(self):
        for rel in (
            "scripts/fit_calibrator_alpha158_fund.py",
            "scripts/stamp_walkforward_fingerprints.py",
        ):
            source = (_REPO / rel).read_text()
            assert "panel_scorer import model_content_sha256" not in source, (
                f"{rel} re-imported the umbrella kernel's stale "
                "model_content_sha256 copy (RQ#444 F-10)"
            )
