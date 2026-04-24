"""Artifact snapshot isolation — A/B protection from concurrent retraining.

User-surfaced 2026-04-24: three A/B runs of GOLDEN_v4.1 produced
39.82%, 34.56%, 29.96% because the notebook retrained artifacts
between sims. This module fixes that by freezing artifacts at A/B
start via tmp dir copy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _make_fake_strategy(tmp_path: Path) -> Path:
    """Seed a strategy_dir layout with minimal content for tests."""
    strategy = tmp_path / "renquant_fake"
    strategy.mkdir()
    (strategy / "artifacts").mkdir()
    (strategy / "models").mkdir()
    (strategy / "artifacts" / "spy-gmm-regime.json").write_text('{"version": 1}')
    (strategy / "artifacts" / "panel-ltr.json").write_text('{"n_rows": 47000}')
    (strategy / "models" / "NVDA").mkdir()
    (strategy / "models" / "NVDA" / "NVDA-policy-metadata.json").write_text('{"sharpe": 1.5}')
    (strategy / "strategy_config.json").write_text('{"watchlist": ["NVDA"]}')
    return strategy


class TestSnapshotCreation:
    def test_creates_tmp_dir_with_all_subdirs(self, tmp_path):
        from kernel.artifact_snapshot import snapshot_artifacts
        strategy = _make_fake_strategy(tmp_path)
        snap = snapshot_artifacts(strategy)
        try:
            assert snap.exists()
            assert (snap / "artifacts" / "spy-gmm-regime.json").exists()
            assert (snap / "artifacts" / "panel-ltr.json").exists()
            assert (snap / "models" / "NVDA" / "NVDA-policy-metadata.json").exists()
            assert (snap / "strategy_config.json").exists()
        finally:
            import shutil
            shutil.rmtree(snap, ignore_errors=True)

    def test_captures_git_sha_when_repo(self, tmp_path):
        """Snapshot records HEAD sha for reproducibility."""
        from kernel.artifact_snapshot import snapshot_artifacts
        # Use real strategy_dir (git repo) so `git rev-parse` succeeds
        snap = snapshot_artifacts(_STRATEGY_DIR)
        try:
            sha_file = snap / ".snapshot_sha"
            assert sha_file.exists()
            sha = sha_file.read_text().strip()
            assert len(sha) == 40   # full git hash
        finally:
            import shutil
            shutil.rmtree(snap, ignore_errors=True)

    def test_missing_strategy_dir_raises(self, tmp_path):
        from kernel.artifact_snapshot import snapshot_artifacts
        with pytest.raises(ValueError, match="not found"):
            snapshot_artifacts(tmp_path / "nonexistent")


class TestIsolation:
    """The key guarantee: post-snapshot mutations to the source DON'T
    affect the snapshot."""

    def test_source_mutation_doesnt_affect_snapshot(self, tmp_path):
        from kernel.artifact_snapshot import snapshot_artifacts
        strategy = _make_fake_strategy(tmp_path)
        snap = snapshot_artifacts(strategy)
        try:
            # Simulate concurrent retraining: overwrite source artifact
            (strategy / "artifacts" / "spy-gmm-regime.json").write_text(
                '{"version": 2, "retrained": true}'
            )

            # Snapshot should still have the original
            snap_content = json.loads(
                (snap / "artifacts" / "spy-gmm-regime.json").read_text()
            )
            assert snap_content == {"version": 1}
        finally:
            import shutil
            shutil.rmtree(snap, ignore_errors=True)

    def test_multiple_snapshots_are_independent(self, tmp_path):
        """Two snapshots from the same source don't share state."""
        from kernel.artifact_snapshot import snapshot_artifacts
        strategy = _make_fake_strategy(tmp_path)
        snap1 = snapshot_artifacts(strategy)
        snap2 = snapshot_artifacts(strategy)
        try:
            assert snap1 != snap2
            # Mutate snap1 — snap2 unchanged
            (snap1 / "artifacts" / "spy-gmm-regime.json").write_text('{"mutated": true}')
            snap2_content = json.loads(
                (snap2 / "artifacts" / "spy-gmm-regime.json").read_text()
            )
            assert snap2_content == {"version": 1}
        finally:
            import shutil
            shutil.rmtree(snap1, ignore_errors=True)
            shutil.rmtree(snap2, ignore_errors=True)


class TestContextManager:
    def test_cleanup_on_exit(self, tmp_path):
        from kernel.artifact_snapshot import snapshot_artifacts_ctx
        strategy = _make_fake_strategy(tmp_path)
        snap_ref = None
        with snapshot_artifacts_ctx(strategy) as snap:
            snap_ref = snap
            assert snap.exists()
        # Cleaned up after exit
        assert not snap_ref.exists()

    def test_cleanup_on_exception(self, tmp_path):
        from kernel.artifact_snapshot import snapshot_artifacts_ctx
        strategy = _make_fake_strategy(tmp_path)
        snap_ref = None
        try:
            with snapshot_artifacts_ctx(strategy) as snap:
                snap_ref = snap
                raise RuntimeError("simulated failure")
        except RuntimeError:
            pass
        assert not snap_ref.exists()   # cleaned up despite exception
