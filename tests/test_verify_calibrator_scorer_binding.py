"""Tests for scripts/verify_calibrator_scorer_binding.py — the runtime-
authoritative scorer/calibrator BINDING gate added 2026-07-01.

Loads the script as a module (no strategy venv / renquant_pipeline import
required) via importlib, then exercises `check_binding` with injected
fixture loader/match functions — same dependency-injection pattern as
tests/test_check_model_bundle_consistency.py in renquant-orchestrator (this
repo's convention for testing shell-embedded logic via an extracted,
importable Python module rather than leaving it un-unit-testable inline).

Covers:
  • matching fixture pair            → status=pass, match=True
  • deliberately mismatched pair     → status=fail, match=False
  • missing fingerprints on either side → status=fail (not silently pass)
  • missing scorer/calibrator artifact → status=error (exit 2 == fail-closed)
  • loader import failure            → status=error (exit 2 == fail-closed,
    never a silent skip — the exact failure mode that let the 2026-07-01
    incident through)
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_SCRIPT = REPO / "scripts" / "verify_calibrator_scorer_binding.py"
_spec = importlib.util.spec_from_file_location("bindinggate", _SCRIPT)
bindinggate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bindinggate)  # type: ignore[union-attr]


# Minimal stand-ins for the real `_fingerprint_values` /
# `_any_fingerprints_match` from job_panel_scoring.py — same field-name +
# prefix-match semantics, kept intentionally simple for fixture tests.
def _fingerprint_values(metadata: dict | None) -> list[str]:
    if not metadata:
        return []
    out = []
    for key in ("model_content_fingerprint", "scorer_model_content_fingerprint",
                "artifact_fingerprint", "scorer_artifact_fingerprint"):
        v = metadata.get(key)
        if v:
            out.append(str(v))
    return out


def _any_fingerprints_match(expected: list[str], actual: list[str]) -> bool:
    exp_norm = {e.strip().lower().removeprefix("sha256:") for e in expected}
    act_norm = {a.strip().lower().removeprefix("sha256:") for a in actual}
    return bool(exp_norm & act_norm)


class _FakeScorer:
    def __init__(self, metadata: dict):
        self.metadata = metadata


class _FakePanelScorerCls:
    """Stand-in for PanelScorer with an injectable metadata-by-path map."""

    def __init__(self, metadata_by_path: dict[str, dict]):
        self._by_path = metadata_by_path

    def load(self, path):
        key = str(path)
        if key not in self._by_path:
            raise FileNotFoundError(f"no fixture scorer at {key}")
        return _FakeScorer(self._by_path[key])


MATCH_FP = "sha256:aaaabbbbccccddddeeeeffff00001111"
OTHER_FP = "sha256:99998888777766665555444433332222"


def _write_calibrator(tmp_path: Path, fp: str | None) -> Path:
    cal_path = tmp_path / "panel-rank-calibration.json"
    metadata = {"scorer_model_content_fingerprint": fp} if fp else {}
    cal_path.write_text(json.dumps({"metadata": metadata}))
    return cal_path


def _make_scorer_fixture(tmp_path: Path, fp: str | None):
    scorer_path = tmp_path / "panel-ltr.json"
    scorer_path.write_text("{}")  # content unused — loader is faked
    metadata = {"model_content_fingerprint": fp} if fp else {}
    loader = _FakePanelScorerCls({str(scorer_path): metadata})
    return scorer_path, loader


def test_matching_fingerprints_pass(tmp_path):
    scorer_path, loader = _make_scorer_fixture(tmp_path, MATCH_FP)
    cal_path = _write_calibrator(tmp_path, MATCH_FP)
    res = bindinggate.check_binding(
        scorer_path, cal_path,
        panel_scorer_cls=loader,
        fingerprint_values=_fingerprint_values,
        any_fingerprints_match=_any_fingerprints_match,
    )
    assert res["status"] == "pass"
    assert res["match"] is True


def test_mismatched_fingerprints_fail(tmp_path):
    scorer_path, loader = _make_scorer_fixture(tmp_path, MATCH_FP)
    cal_path = _write_calibrator(tmp_path, OTHER_FP)
    res = bindinggate.check_binding(
        scorer_path, cal_path,
        panel_scorer_cls=loader,
        fingerprint_values=_fingerprint_values,
        any_fingerprints_match=_any_fingerprints_match,
    )
    assert res["status"] == "fail"
    assert res["match"] is False
    assert "BINDING MISMATCH" in res["reason"]


def test_missing_fingerprints_fail_not_silently_pass(tmp_path):
    scorer_path, loader = _make_scorer_fixture(tmp_path, None)
    cal_path = _write_calibrator(tmp_path, None)
    res = bindinggate.check_binding(
        scorer_path, cal_path,
        panel_scorer_cls=loader,
        fingerprint_values=_fingerprint_values,
        any_fingerprints_match=_any_fingerprints_match,
    )
    assert res["status"] == "fail"
    assert res["match"] is False


def test_missing_scorer_artifact_errors_fail_closed(tmp_path):
    _, loader = _make_scorer_fixture(tmp_path, MATCH_FP)
    cal_path = _write_calibrator(tmp_path, MATCH_FP)
    res = bindinggate.check_binding(
        tmp_path / "does-not-exist.json", cal_path,
        panel_scorer_cls=loader,
        fingerprint_values=_fingerprint_values,
        any_fingerprints_match=_any_fingerprints_match,
    )
    assert res["status"] == "error"
    assert res["match"] is False


def test_missing_calibrator_artifact_errors_fail_closed(tmp_path):
    scorer_path, loader = _make_scorer_fixture(tmp_path, MATCH_FP)
    res = bindinggate.check_binding(
        scorer_path, tmp_path / "does-not-exist.json",
        panel_scorer_cls=loader,
        fingerprint_values=_fingerprint_values,
        any_fingerprints_match=_any_fingerprints_match,
    )
    assert res["status"] == "error"
    assert res["match"] is False


def test_scorer_load_failure_errors_fail_closed(tmp_path):
    scorer_path = tmp_path / "panel-ltr.json"
    scorer_path.write_text("{}")
    cal_path = _write_calibrator(tmp_path, MATCH_FP)

    class _RaisingLoader:
        def load(self, path):
            raise ValueError("corrupt artifact")

    res = bindinggate.check_binding(
        scorer_path, cal_path,
        panel_scorer_cls=_RaisingLoader(),
        fingerprint_values=_fingerprint_values,
        any_fingerprints_match=_any_fingerprints_match,
    )
    assert res["status"] == "error"
    assert res["match"] is False


def test_import_failure_fails_closed_not_silently_skipped(tmp_path, monkeypatch):
    """The exact 2026-07-01 failure mode: a check that silently no-ops when
    its dependency isn't importable must instead be a hard gate failure."""
    scorer_path = tmp_path / "panel-ltr.json"
    scorer_path.write_text("{}")
    cal_path = _write_calibrator(tmp_path, MATCH_FP)

    def _boom():
        raise ImportError("renquant_pipeline not on PYTHONPATH")

    monkeypatch.setattr(bindinggate, "load_runtime_authorities", _boom)
    res = bindinggate.check_binding(scorer_path, cal_path)
    assert res["status"] == "error"
    assert res["match"] is False
    assert "not importable" in res["reason"]
    assert "failing CLOSED" in res["reason"]


def test_main_exit_codes(tmp_path, monkeypatch, capsys):
    scorer_path, loader = _make_scorer_fixture(tmp_path, MATCH_FP)
    cal_path = _write_calibrator(tmp_path, MATCH_FP)
    monkeypatch.setattr(
        bindinggate, "load_runtime_authorities",
        lambda: (loader, _fingerprint_values, _any_fingerprints_match),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["verify_calibrator_scorer_binding.py",
         "--scorer", str(scorer_path), "--calibrator", str(cal_path)],
    )
    assert bindinggate.main() == 0

    mismatched_cal = _write_calibrator(tmp_path, OTHER_FP)
    monkeypatch.setattr(
        "sys.argv",
        ["verify_calibrator_scorer_binding.py",
         "--scorer", str(scorer_path), "--calibrator", str(mismatched_cal)],
    )
    assert bindinggate.main() == 1
