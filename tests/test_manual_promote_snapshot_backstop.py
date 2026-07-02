"""Real execution test for manual_promote.sh's snapshot-freshness backstop
(Codex PR #432 round 5) — the emergency-promote path, interactive via
read -p prompts, fed via stdin here rather than mocked away.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _weekly_promote_fixture as fixture  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "manual_promote.sh"


def _write_staging_artifact(root: Path, mod) -> Path:
    staging = (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod"
               / fixture.STAGING_ARTIFACT_NAME)
    staging.parent.mkdir(parents=True, exist_ok=True)
    # Byte-identical to the active artifact the fixture already committed a
    # snapshot for (see _weekly_promote_fixture.build_fixture_repo) — the
    # "fresh" case promotes staging over active with no real content change.
    active = root / mod.STRATEGY_DIR_REL / "artifacts" / "prod" / fixture.ACTIVE_ARTIFACT_NAME
    staging.write_bytes(active.read_bytes())
    return staging


def _run(root: Path, staging: Path) -> subprocess.CompletedProcess:
    stdin = f"{staging}\nemergency_bugfix\ny\n"
    env = {
        "RQ_MANUAL_PROMOTE_REPO_DIR": str(root),
        "RQ_MANUAL_PROMOTE_PYTHON": sys.executable,
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(root),
    }
    return subprocess.run(
        ["bash", str(SCRIPT)], input=stdin, env=env,
        capture_output=True, text=True, timeout=60)


def test_fresh_snapshot_reaches_success(tmp_path):
    root = tmp_path / "repo"
    mod = fixture.build_fixture_repo(root)
    staging = _write_staging_artifact(root, mod)

    result = _run(root, staging)
    assert result.returncode == 0, (
        f"expected success; stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    assert "EMERGENCY PROMOTE complete" in result.stdout
    assert "STALE" not in result.stdout


def test_stale_snapshot_produces_distinct_message_and_exits_nonzero(tmp_path):
    root = tmp_path / "repo"
    mod = fixture.build_fixture_repo(root)
    staging = _write_staging_artifact(root, mod)
    fixture.make_snapshot_stale(root, mod)

    result = _run(root, staging)
    assert result.returncode == 1, (
        f"a stale snapshot must fail non-zero; stdout:\n{result.stdout}")
    assert "Snapshot freshness backstop FAILED" in result.stdout
    assert "run 'make snapshot'" in result.stdout


def test_stale_snapshot_does_not_revert_the_completed_promotion(tmp_path):
    root = tmp_path / "repo"
    mod = fixture.build_fixture_repo(root)
    staging = _write_staging_artifact(root, mod)
    fixture.make_snapshot_stale(root, mod)

    active = root / mod.STRATEGY_DIR_REL / "artifacts" / "prod" / fixture.ACTIVE_ARTIFACT_NAME
    staging_bytes = staging.read_bytes()

    result = _run(root, staging)
    assert result.returncode == 1
    # The emergency promote (staging -> active copy) must have actually
    # happened and stayed in place despite the later stale-snapshot failure.
    assert active.read_bytes() == staging_bytes, (
        "a stale-snapshot finding must not revert the completed emergency promotion")
