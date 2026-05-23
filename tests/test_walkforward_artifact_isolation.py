"""AUDIT REGRESSION GUARD — pins the 2026-05-10 incident.

train_walkforward_panel.py::configure_panel_cutoff originally only set
`cfg["panel_ltr"]["artifact_path"]` for the per-cutoff retrain. But
SaveArtifactTask (pp_panel_training.py:2684-2699) reads
`cfg["ranking"]["panel_scoring"]["artifact_path"]` FIRST and falls back
to training-side only when inference-side is unset. Result: every
walkforward retrain overwrote the production
`panel-ltr.alpha158_fund.json` artifact.

Per CLAUDE.md §5.13.3, this regression guard pins the invariant:
configure_panel_cutoff MUST set BOTH keys to the per-cutoff path.

Per §5.13.13 (side configs are loaded weapons), the function also
asserts the path contains "walkforward" so a future variant cannot
silently re-introduce production-overwriting behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# scripts/ is at repo root, not on the package path; insert it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.train_walkforward_panel import (  # noqa: E402
    build_retrain_entry,
    configure_panel_cutoff,
    infer_label_lookahead_days,
    make_calibrator_path,
)


PROD_PATH = "artifacts/panel-ltr.alpha158_fund.json"
WF_PATH = "artifacts/walkforward/2024-01-01/panel-ltr.json"


class TestConfigurePanelCutoffSetsBothPaths:
    """The 2026-05-10 audit class: both training-side AND inference-side
    artifact_path keys must point at the per-cutoff path."""

    def test_both_keys_redirected(self):
        cfg = {
            "panel_ltr": {"artifact_path": PROD_PATH},
            "ranking": {"panel_scoring": {"artifact_path": PROD_PATH}},
        }
        out = configure_panel_cutoff(
            cfg, pd.Timestamp("2024-01-01"), Path(WF_PATH),
        )
        assert out["panel_ltr"]["artifact_path"] == WF_PATH
        assert out["ranking"]["panel_scoring"]["artifact_path"] == WF_PATH

    def test_train_cutoff_set(self):
        cfg = {
            "panel_ltr": {"artifact_path": PROD_PATH},
            "ranking": {"panel_scoring": {"artifact_path": PROD_PATH}},
        }
        out = configure_panel_cutoff(
            cfg, pd.Timestamp("2024-01-01"), Path(WF_PATH),
        )
        assert out["panel_ltr"]["train_cutoff"] == "2024-01-01T00:00:00"

    def test_global_calibration_auto_refresh_disabled(self):
        cfg = {
            "panel_ltr": {"artifact_path": PROD_PATH},
            "ranking": {"panel_scoring": {"artifact_path": PROD_PATH}},
        }
        out = configure_panel_cutoff(
            cfg, pd.Timestamp("2024-01-01"), Path(WF_PATH),
        )
        assert (out["ranking"]["panel_scoring"]
                   ["global_calibration"]["auto_refresh"] is False)

    def test_creates_keys_on_empty_cfg(self):
        cfg: dict = {}
        out = configure_panel_cutoff(
            cfg, pd.Timestamp("2024-01-01"), Path(WF_PATH),
        )
        assert out["panel_ltr"]["artifact_path"] == WF_PATH
        assert out["ranking"]["panel_scoring"]["artifact_path"] == WF_PATH


def test_train_walkforward_panel_supports_cutoff_parallelism():
    src = (_REPO_ROOT / "scripts" / "train_walkforward_panel.py").read_text()
    assert '"--jobs"' in src
    assert "ThreadPoolExecutor" in src
    assert "as_completed" in src
    assert "entries_by_cutoff" in src


def test_train_walkforward_panel_stamps_per_fold_calibrator_uri():
    entry = build_retrain_entry(
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        "artifacts/walkforward_v2/2024-01-01/panel-ltr.json",
        lookahead_days=60,
        calibrator_uri="artifacts/walkforward_v2/2024-01-01/panel-rank-calibration.json",
    )
    assert entry.calibrator_uri.endswith("panel-rank-calibration.json")


def test_make_calibrator_path_sits_next_to_scorer():
    path = make_calibrator_path(
        Path("artifacts/walkforward_v2/2024-01-01/panel-ltr.json")
    )
    assert path == Path("artifacts/walkforward_v2/2024-01-01/panel-rank-calibration.json")


def test_infer_label_lookahead_days_from_label_name():
    assert infer_label_lookahead_days("fwd_5d_excess") == 5
    assert infer_label_lookahead_days("fwd_60d_excess") == 60


class TestNonWalkforwardPathRejected:
    """Sanity guard per §5.13.3 — refuse production-shaped paths."""

    @pytest.mark.parametrize("bad_path", [
        "artifacts/panel-ltr.alpha158_fund.json",       # exact prod path
        "artifacts/panel-ltr.json",                      # legacy prod path
        "artifacts/some/other/place.json",              # any non-walkforward
        "/tmp/scratch.json",                             # absolute outside
    ])
    def test_rejects_path_without_walkforward_segment(self, bad_path):
        cfg: dict = {}
        with pytest.raises(AssertionError, match="walkforward"):
            configure_panel_cutoff(
                cfg, pd.Timestamp("2024-01-01"), Path(bad_path),
            )

    @pytest.mark.parametrize("good_path", [
        "artifacts/walkforward/2024-01-01/panel-ltr.json",
        "artifacts/walkforward/2025-06-30/panel-ltr.json",
        "/Users/foo/repo/artifacts/walkforward/2024-12-31/panel-ltr.json",
    ])
    def test_accepts_path_with_walkforward_segment(self, good_path):
        cfg: dict = {}
        configure_panel_cutoff(
            cfg, pd.Timestamp("2024-01-01"), Path(good_path),
        )


class TestAuditRegression20260510:
    """AUDIT REGRESSION GUARD — exact 2026-05-10 incident pattern.

    Reproduces the original bug: with only the training-side key
    redirected, the writer falls through to the inference-side key
    which still points at production. This test asserts that pattern
    is impossible after the fix.
    """

    def test_bug_pattern_no_longer_possible(self):
        """The original buggy behavior was: only set panel_ltr.artifact_path,
        leave ranking.panel_scoring.artifact_path untouched. SaveArtifactTask
        prefers inference-side, so the writer wrote to the production path.

        After the fix, configure_panel_cutoff MUST also override the
        inference-side key; this test pins that invariant.
        """
        prod_inference_path = "artifacts/panel-ltr.alpha158_fund.json"
        cfg = {
            "panel_ltr": {"artifact_path": prod_inference_path},
            "ranking": {"panel_scoring": {
                "artifact_path": prod_inference_path,
            }},
        }
        out = configure_panel_cutoff(
            cfg, pd.Timestamp("2024-01-01"),
            Path("artifacts/walkforward/2024-01-01/panel-ltr.json"),
        )
        # The post-condition that prevents the bug class:
        # inference-side path must NOT equal production path after redirect.
        assert (out["ranking"]["panel_scoring"]["artifact_path"]
                != prod_inference_path), (
            "Bug 2026-05-10 pattern: inference-side artifact_path was "
            "not redirected by configure_panel_cutoff — writer would "
            "overwrite production"
        )
