"""Umbrella ArtifactResolver tests — single path-resolution authority.

Design: renquant-orchestrator
doc/research/2026-06-12-engineering-architecture-deep-plan.md §III.5;
mirrors renquant-pipeline PR #115 (keep the two copies behavior-identical:
divergence between them is the bug class the module kills).

Pinned here beyond the core invariants: all four previously-ad-hoc
umbrella resolution sites now delegate to the authority —
  1. artifact_contract._resolve_path (was: prefix-string heuristic)
  2. preflight._resolve_artifact_path (was: strategy_dir only)
  3. LoadScorerTask._resolve_artifact_path (was: strategy_dir only)
  4. job_panel_scoring global-calibration closure (was: strategy_dir only)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.artifact_resolver import (  # noqa: E402
    default_repo_root,
    locate_artifact,
    resolve_artifact,
)


@pytest.fixture()
def layout(tmp_path):
    repo_root = tmp_path
    strategy_dir = repo_root / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True)
    return repo_root, strategy_dir


class TestCoreInvariants:

    def test_strategy_dir_first(self, layout):
        repo_root, strategy_dir = layout
        ref = "artifacts/prod/panel-ltr.json"
        for root, body in ((strategy_dir, b"strategy"), (repo_root, b"root")):
            (root / ref).parent.mkdir(parents=True, exist_ok=True)
            (root / ref).write_bytes(body)
        got = resolve_artifact(ref, strategy_dir=strategy_dir, repo_root=repo_root)
        assert got.source == "strategy_dir"
        assert got.path.read_bytes() == b"strategy"

    def test_repo_root_fallback(self, layout):
        repo_root, strategy_dir = layout
        ref = "data/only_at_root.json"
        (repo_root / ref).parent.mkdir(parents=True)
        (repo_root / ref).write_bytes(b"{}")
        got = resolve_artifact(ref, strategy_dir=strategy_dir, repo_root=repo_root)
        assert got.source == "repo_root"

    def test_fail_closed_lists_candidates(self, layout):
        repo_root, strategy_dir = layout
        with pytest.raises(FileNotFoundError, match="fail-closed"):
            resolve_artifact("nope/missing.json",
                             strategy_dir=strategy_dir, repo_root=repo_root)

    def test_sha256_is_content_digest(self, layout):
        repo_root, strategy_dir = layout
        (strategy_dir / "m.json").write_bytes(b"payload")
        got = resolve_artifact("m.json", strategy_dir=strategy_dir,
                               repo_root=repo_root)
        assert got.sha256 == hashlib.sha256(b"payload").hexdigest()[:16]

    def test_default_repo_root_convention(self, layout):
        repo_root, strategy_dir = layout
        assert default_repo_root(strategy_dir) == repo_root

    def test_locate_never_raises(self, layout):
        repo_root, strategy_dir = layout
        p = locate_artifact("missing.json", strategy_dir=strategy_dir,
                            repo_root=repo_root)
        assert p == strategy_dir / "missing.json"


class TestArtifactContractDelegation:
    """Site 1: the prefix-string heuristic is dead."""

    def test_prefixed_ref_resolves_by_existence_not_prefix(self, layout):
        from kernel.artifact_contract import _resolve_path

        repo_root, strategy_dir = layout
        # A "data/"-prefixed ref that exists ONLY under strategy_dir: the
        # old heuristic would blindly return the (missing) repo-root path.
        ref = "data/special/calib.json"
        (strategy_dir / ref).parent.mkdir(parents=True)
        (strategy_dir / ref).write_bytes(b"{}")
        assert _resolve_path(strategy_dir, ref) == strategy_dir / ref

    def test_repo_root_ref_still_resolves(self, layout):
        from kernel.artifact_contract import _resolve_path

        repo_root, strategy_dir = layout
        ref = "backtesting/renquant_104/artifacts/x.json"
        (repo_root / ref).parent.mkdir(parents=True)
        (repo_root / ref).write_bytes(b"{}")
        assert _resolve_path(strategy_dir, ref) == repo_root / ref


class TestPreflightDelegation:
    """Site 2: preflight sees the loaders' repo-root fallback."""

    def test_repo_root_artifact_visible_to_preflight(self, layout):
        from kernel.preflight import _resolve_artifact_path

        repo_root, strategy_dir = layout
        (repo_root / "artifacts").mkdir()
        (repo_root / "artifacts" / "panel.json").write_bytes(b"{}")
        p = _resolve_artifact_path(strategy_dir, "artifacts/panel.json")
        assert p == repo_root / "artifacts" / "panel.json"

    def test_missing_reports_strategy_dir_candidate(self, layout):
        from kernel.preflight import _resolve_artifact_path

        repo_root, strategy_dir = layout
        p = _resolve_artifact_path(strategy_dir, "artifacts/none.json")
        assert p == strategy_dir / "artifacts" / "none.json"


class TestLoadScorerDelegation:
    """Sites 3+4: scorer + calibrator resolve through the authority."""

    def _ctx(self, strategy_dir):
        class _Ctx:
            config = {"_strategy_dir": str(strategy_dir)}
        return _Ctx()

    def test_scorer_ref_gets_repo_root_fallback(self, layout):
        from kernel.panel_pipeline.job_panel_scoring import LoadScorerTask

        repo_root, strategy_dir = layout
        ref = "artifacts/prod/panel-ltr.alpha158_fund.json"
        (repo_root / ref).parent.mkdir(parents=True)
        (repo_root / ref).write_bytes(b"{}")
        p = LoadScorerTask._resolve_artifact_path(
            self._ctx(strategy_dir), {"artifact_path": ref})
        assert p == repo_root / ref

    def test_scorer_strategy_dir_still_wins(self, layout):
        from kernel.panel_pipeline.job_panel_scoring import LoadScorerTask

        repo_root, strategy_dir = layout
        ref = "artifacts/prod/panel-ltr.json"
        for root in (strategy_dir, repo_root):
            (root / ref).parent.mkdir(parents=True, exist_ok=True)
            (root / ref).write_bytes(b"{}")
        p = LoadScorerTask._resolve_artifact_path(
            self._ctx(strategy_dir), {"artifact_path": ref})
        assert p == strategy_dir / ref

    def test_absolute_ref_untouched(self, layout):
        from kernel.panel_pipeline.job_panel_scoring import LoadScorerTask

        repo_root, strategy_dir = layout
        f = repo_root / "abs.json"
        f.write_bytes(b"{}")
        p = LoadScorerTask._resolve_artifact_path(
            self._ctx(strategy_dir), {"artifact_path": str(f)})
        assert p == f
