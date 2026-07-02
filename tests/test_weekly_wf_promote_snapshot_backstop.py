"""Real execution test for weekly_wf_promote.sh's Step 7 snapshot backstop
(Codex PR #432 round 5): code-reading confirmed the placement was correct,
but nothing had actually RUN the production shell script. This drives the
real script through a fully mocked promotion (Steps 1-6 stubbed; Step 7 is
NOT mocked — it runs the real check_snapshot_freshness against a fixture
repo) and asserts on its actual exit code and notification log.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _weekly_promote_fixture as fixture  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "weekly_wf_promote.sh"


def _run(root: Path, notify_log: Path, lock_file: Path) -> subprocess.CompletedProcess:
    env = {
        "RQ_WEEKLY_PROMOTE_REPO_DIR": str(root),
        "RQ_WEEKLY_PROMOTE_PYTHON": sys.executable,
        "RQ_WEEKLY_PROMOTE_NOTIFY_LOG": str(notify_log),
        "RQ_WEEKLY_PROMOTE_LOCK_FILE": str(lock_file),
        "RQ_WF_GATE_RUNNER": "umbrella",
        # Shims FIRST: notify() must observe via RQ_WEEKLY_PROMOTE_NOTIFY_LOG
        # only — never POST test noise (e.g. a fake "SNAPSHOT STALE" alert)
        # to the real ntfy topic or pop real desktop notifications.
        "PATH": f"{fixture.shim_bin_dir(root)}:/usr/bin:/bin:/usr/local/bin",
        "HOME": str(root),
    }
    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=120)


def test_fresh_snapshot_reaches_success_exactly_once(tmp_path):
    root = tmp_path / "repo"
    fixture.build_fixture_repo(root)
    notify_log = tmp_path / "notify.log"
    lock_file = tmp_path / "weekly.lock"

    result = _run(root, notify_log, lock_file)
    log_tail = (root / "logs" / "weekly_wf_promote").glob("*.log")
    tail_text = "\n".join(p.read_text(encoding="utf-8") for p in log_tail)
    assert result.returncode == 0, f"expected success; stderr/log tail:\n{tail_text[-3000:]}"

    notifications = notify_log.read_text(encoding="utf-8") if notify_log.exists() else ""
    assert "WEEKLY-PROMOTE ✓" in notifications, notifications
    assert "SNAPSHOT STALE" not in notifications, notifications
    assert notifications.count("WEEKLY-PROMOTE ✓") == 1, (
        "success notification must fire exactly once, not double-notify: "
        + notifications)
    assert not lock_file.exists(), "lock file must be released on exit"


def test_stale_snapshot_produces_distinct_alert_suppresses_success_exits_nonzero(tmp_path):
    root = tmp_path / "repo"
    mod = fixture.build_fixture_repo(root)
    fixture.make_snapshot_stale(root, mod)
    notify_log = tmp_path / "notify.log"
    lock_file = tmp_path / "weekly.lock"

    result = _run(root, notify_log, lock_file)
    assert result.returncode == 1, (
        f"a stale snapshot must fail Step 7 non-zero; stdout tail:\n{result.stdout[-2000:]}")

    notifications = notify_log.read_text(encoding="utf-8") if notify_log.exists() else ""
    assert "WEEKLY-PROMOTE — SNAPSHOT STALE" in notifications, notifications
    # The final success notification must be SUPPRESSED, not merely absent
    # alongside a partial run — confirm it genuinely never fired.
    assert "WEEKLY-PROMOTE ✓" not in notifications, (
        "the stale-snapshot alert must suppress the final success "
        "notification, not merely coexist with it: " + notifications)
    assert not lock_file.exists(), "lock file must still be released on a failed run"


def test_stale_snapshot_does_not_revert_the_completed_promotion(tmp_path):
    """The model promotion (Steps 1-6) already succeeded when Step 7 finds
    drift — the review is explicit that this must not be silently undone."""
    root = tmp_path / "repo"
    mod = fixture.build_fixture_repo(root)
    fixture.make_snapshot_stale(root, mod)
    notify_log = tmp_path / "notify.log"
    lock_file = tmp_path / "weekly.lock"

    active_artifact = (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod"
                        / fixture.ACTIVE_ARTIFACT_NAME)
    before = active_artifact.read_text(encoding="utf-8")

    result = _run(root, notify_log, lock_file)
    assert result.returncode == 1

    after = active_artifact.read_text(encoding="utf-8")
    # The retrain stub writes fixed content on every run (same values), so
    # "unchanged" here specifically confirms no rollback/revert path fired —
    # the promoted artifact is the retrain stub's output, present and intact.
    assert after == before, "a stale-snapshot finding must not revert the completed promotion"
