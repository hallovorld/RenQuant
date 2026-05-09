"""Walk-forward gate enforcement tests (P0 #1, post E55 revert).

Per CLAUDE.md §5.9 + roadmap rewrite: every promote requires walk-forward
3-cut Sharpe + §5.2 sanity battery (shuffled-label + time-shift placebo)
recorded in artifact metadata BEFORE the artifact may be swapped into
active.

This test class pins the gate behavior:
  - missing wf_gate_metadata        → promote() raises
  - passed=False                    → promote() raises
  - run_at older than 14 days       → promote() raises
  - run_at unparseable              → promote() succeeds with warning
  - RQ_ALLOW_NO_WF=1 override       → promote() succeeds with WARN log
  - all clean (passed + recent)     → promote() succeeds normally

Reference: kernel/model_acceptance.py::_check_wf_gate
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.model_acceptance import promote, _check_wf_gate


def _wf_meta(passed: bool = True, age_days: float = 1.0) -> dict:
    """Build a wf_gate_metadata dict with the given pass status + age."""
    ran = datetime.datetime.utcnow() - datetime.timedelta(days=age_days)
    return {
        "passed": passed,
        "wf_3cut_sharpe_mean": 0.55,
        "wf_3cut_sharpe_std": 0.20,
        "wf_3cut_apy_mean": 0.075,
        "sanity_shuffled_ic": -0.0024,
        "sanity_placebo_ic": 0.012,
        "run_at": ran.isoformat(),
        "gate_version": 1,
    }


def _staging_artifact(tmp_path: Path, *, wf: dict | None = None) -> Path:
    """Write a minimal valid staging artifact, optionally with wf_gate_metadata."""
    art = {"kind": "panel_ltr_xgboost", "feature_cols": ["f1", "f2"]}
    if wf is not None:
        art["metadata"] = {"wf_gate_metadata": wf}
    p = tmp_path / "staging.json"
    p.write_text(json.dumps(art))
    return p


# ── Direct gate-function tests ────────────────────────────────────────────────

class TestCheckWFGate:
    """Direct unit tests of the _check_wf_gate helper."""

    def test_missing_metadata_raises(self):
        with pytest.raises(ValueError, match="missing wf_gate_metadata"):
            _check_wf_gate({"kind": "panel_ltr_xgboost"}, Path("/fake.json"))

    def test_passed_false_raises(self):
        data = {"metadata": {"wf_gate_metadata": _wf_meta(passed=False)}}
        with pytest.raises(ValueError, match="passed=False"):
            _check_wf_gate(data, Path("/fake.json"))

    def test_passed_true_succeeds(self):
        data = {"metadata": {"wf_gate_metadata": _wf_meta(passed=True)}}
        # No raise = pass
        _check_wf_gate(data, Path("/fake.json"))

    def test_stale_run_at_raises(self):
        data = {"metadata": {"wf_gate_metadata": _wf_meta(passed=True, age_days=20)}}
        with pytest.raises(ValueError, match="stale"):
            _check_wf_gate(data, Path("/fake.json"))

    def test_recent_run_at_succeeds(self):
        data = {"metadata": {"wf_gate_metadata": _wf_meta(passed=True, age_days=5)}}
        _check_wf_gate(data, Path("/fake.json"))

    def test_metadata_at_root_also_works(self):
        """Some artifacts store wf_gate_metadata at top-level, not inside metadata."""
        data = {"wf_gate_metadata": _wf_meta(passed=True)}
        _check_wf_gate(data, Path("/fake.json"))

    def test_emergency_override_bypasses(self, monkeypatch):
        monkeypatch.setenv("RQ_ALLOW_NO_WF", "1")
        # Even with NO metadata, override should bypass
        _check_wf_gate({}, Path("/fake.json"))
        # And with passed=False, also bypasses
        data = {"metadata": {"wf_gate_metadata": _wf_meta(passed=False)}}
        _check_wf_gate(data, Path("/fake.json"))

    def test_unparseable_run_at_warns_not_raises(self):
        wf = _wf_meta(passed=True)
        wf["run_at"] = "not-a-date"
        data = {"metadata": {"wf_gate_metadata": wf}}
        # Should pass — unparseable run_at logs warning but doesn't raise
        _check_wf_gate(data, Path("/fake.json"))


# ── Integration: promote() end-to-end ─────────────────────────────────────────

class TestPromoteEndToEnd:
    """Verify promote() refuses bad WF metadata and succeeds with good."""

    def test_promote_refuses_without_wf_metadata(self, tmp_path):
        staging = _staging_artifact(tmp_path, wf=None)
        active = tmp_path / "active.json"
        with pytest.raises(ValueError, match="missing wf_gate_metadata"):
            promote(staging, active)
        # Staging should NOT be moved
        assert staging.exists()
        assert not active.exists()

    def test_promote_refuses_with_failed_wf(self, tmp_path):
        staging = _staging_artifact(tmp_path, wf=_wf_meta(passed=False))
        active = tmp_path / "active.json"
        with pytest.raises(ValueError, match="passed=False"):
            promote(staging, active)
        assert staging.exists()
        assert not active.exists()

    def test_promote_succeeds_with_passed_wf(self, tmp_path):
        staging = _staging_artifact(tmp_path, wf=_wf_meta(passed=True))
        active = tmp_path / "active.json"
        promote(staging, active)
        # active is now populated, staging is gone
        assert not staging.exists()
        assert active.exists()
        loaded = json.loads(active.read_text())
        assert loaded.get("kind") == "panel_ltr_xgboost"
        wf = loaded.get("metadata", {}).get("wf_gate_metadata")
        assert wf and wf.get("passed") is True

    def test_promote_with_override_succeeds_without_metadata(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RQ_ALLOW_NO_WF", "1")
        staging = _staging_artifact(tmp_path, wf=None)
        active = tmp_path / "active.json"
        # Should succeed despite missing wf_gate_metadata
        promote(staging, active)
        assert active.exists()

    def test_promote_refuses_stale_wf(self, tmp_path):
        staging = _staging_artifact(tmp_path, wf=_wf_meta(passed=True, age_days=20))
        active = tmp_path / "active.json"
        with pytest.raises(ValueError, match="stale"):
            promote(staging, active)

    def test_promote_preserves_prior_on_swap(self, tmp_path):
        """Prior active artifact must be preserved in .previous.json."""
        # Set up an existing active first
        active = tmp_path / "active.json"
        active.write_text(json.dumps({"kind": "old", "feature_cols": ["x"]}))

        staging = _staging_artifact(tmp_path, wf=_wf_meta(passed=True))
        promote(staging, active)

        previous = active.with_suffix(".previous.json")
        assert previous.exists()
        loaded_prev = json.loads(previous.read_text())
        assert loaded_prev.get("kind") == "old"
