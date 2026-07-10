"""Regression guards for the backup_to_github.sh rc-swallow bug.

Incident 2026-07-10T14:00:05Z: the hourly multirepo backup failed (rc=1,
oversized data/runs.alpaca.db) but ntfy said "failed rc=0" and the script
exited 0, so launchd recorded success and the failure stayed invisible.

Root cause: ``if run_multirepo_backup; then exit 0; fi; BACKUP_RC=$?``
captures the status of the *if construct* (0 when no branch ran), not the
function's return code. The fix captures the rc directly with
``run_multirepo_backup && BACKUP_RC=0 || BACKUP_RC=$?`` (checked context, so
the ERR trap does not double-notify), notifies with the real rc plus the last
JSON line of the module output, and exits nonzero so launchd sees the failure.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "backup_to_github.sh"

STUB_SUBREPO_ENV = """\
renquant_load_subrepo_env() { :; }
renquant_subrepo_root() { echo "$1"; }
renquant_subrepo_src() { echo "$1"; }
renquant_strict_enabled() { return 1; }
"""

# Stub venv python: the import probe (`python -` heredoc) succeeds; the module
# run (`python -m renquant_orchestrator.state_backup ...`) prints a JSON
# summary line (like the real module does, even on failure) and exits MODULE_RC.
STUB_PYTHON_TEMPLATE = """\
#!/bin/sh
if [ "$1" = "-m" ]; then
    echo '{{"committed": false, "error": "stub state-backup failure", "pushed": false}}'
    exit {module_rc}
fi
exit 0
"""

STUB_CURL = """\
#!/bin/sh
echo "$@" >> "$CURL_LOG"
exit 0
"""

STUB_NOOP = "#!/bin/sh\nexit 0\n"


def _make_harness(tmp_path: Path, module_rc: int) -> tuple[Path, dict[str, str], Path]:
    """Copy the real script into an isolated fake repo tree with a stubbed
    .venv python, a stubbed subrepo_env.sh, and network/notification stubs."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, repo / "scripts" / "backup_to_github.sh")
    (repo / "scripts" / "subrepo_env.sh").write_text(STUB_SUBREPO_ENV)

    venv_bin = repo / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    stub_python = venv_bin / "python"
    stub_python.write_text(STUB_PYTHON_TEMPLATE.format(module_rc=module_rc))
    stub_python.chmod(0o755)

    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    curl = stub_bin / "curl"
    curl.write_text(STUB_CURL)
    curl.chmod(0o755)
    notifier = stub_bin / "terminal-notifier"
    notifier.write_text(STUB_NOOP)
    notifier.chmod(0o755)

    curl_log = tmp_path / "curl.log"
    env = dict(os.environ)
    env["PATH"] = f"{stub_bin}:{env['PATH']}"
    env["CURL_LOG"] = str(curl_log)
    env["BACKUP_REPO"] = str(tmp_path / "backup-repo")
    env.pop("RQ_STATE_BACKUP_RUNNER", None)
    env.pop("RQ_STATE_BACKUP_STRICT", None)
    return repo, env, curl_log


def _run_script(repo: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(repo / "scripts" / "backup_to_github.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


class TestBackupRcPropagation:
    def test_multirepo_failure_exits_nonzero_and_notifies_real_rc(self, tmp_path):
        repo, env, curl_log = _make_harness(tmp_path, module_rc=1)

        proc = _run_script(repo, env)

        assert proc.returncode == 1, (
            f"script must propagate the module rc to launchd; "
            f"got rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
        notified = curl_log.read_text()
        assert "rc=1" in notified, f"notification must carry the real rc: {notified}"
        assert "rc=0" not in notified, f"the rc=0 swallow bug is back: {notified}"
        # The last JSON line of the module output rides along for triage.
        assert "stub state-backup failure" in notified

    def test_multirepo_success_still_exits_zero_silently(self, tmp_path):
        repo, env, curl_log = _make_harness(tmp_path, module_rc=0)

        proc = _run_script(repo, env)

        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert not curl_log.exists(), (
            f"no failure notification expected on success: {curl_log.read_text() if curl_log.exists() else ''}"
        )

    def test_script_captures_function_rc_not_if_construct_rc(self):
        sh = SCRIPT.read_text()
        assert "if run_multirepo_backup; then" not in sh, (
            "`if run_multirepo_backup; then exit 0; fi; BACKUP_RC=$?` captures "
            "the if-construct's rc (0), not the function's — rc-swallow regression"
        )
        assert "run_multirepo_backup && BACKUP_RC=0 || BACKUP_RC=$?" in sh
        assert 'exit "$BACKUP_RC"' in sh
