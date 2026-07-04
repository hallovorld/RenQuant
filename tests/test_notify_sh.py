"""Tests for scripts/notify.sh — the canonical shell ntfy sender (campaign B6).

Contract (must mirror renquant_common.notify): topic resolution
($NTFY_TOPIC > $RQ_ROOT/.env parse > "renquant"), RENQUANT_NO_NOTIFY
suppression honored always, --max-time 5, never fails the caller.

No network: curl is stubbed via a PATH shim that records its argv.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

NOTIFY_SH = Path(__file__).resolve().parents[1] / "scripts" / "notify.sh"

SHELLS = [s for s in ("/bin/sh", "/bin/bash", "/bin/zsh") if os.path.exists(s)]


@pytest.fixture()
def curl_stub(tmp_path: Path) -> tuple[Path, Path]:
    """A fake curl on PATH that appends its argv to a log file."""
    log = tmp_path / "curl_args.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "curl"
    stub.write_text(
        "#!/bin/sh\n"
        f'printf \'%s\\n\' "$@" >> "{log}"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return bin_dir, log


def _run(shell: str, snippet: str, *, env: dict[str, str], cwd: Path | None = None):
    # set -u: the helper must be sourceable under the wrappers' strict mode.
    script = f'set -u\n. "{NOTIFY_SH}"\n{snippet}\n'
    return subprocess.run(
        [shell, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=30,
    )


def _base_env(bin_dir: Path, tmp_path: Path) -> dict[str, str]:
    return {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        # Point RQ_ROOT away from the real umbrella so no real .env is read.
        "RQ_ROOT": str(tmp_path),
    }


@pytest.mark.integration
@pytest.mark.parametrize("shell", SHELLS)
def test_no_notify_suppresses_and_returns_zero(shell, curl_stub, tmp_path):
    bin_dir, log = curl_stub
    env = _base_env(bin_dir, tmp_path) | {"RENQUANT_NO_NOTIFY": "1", "NTFY_TOPIC": "t"}
    proc = _run(shell, 'rq_notify "Title" "body"; echo "rc=$?"', env=env)
    assert proc.returncode == 0, proc.stderr
    assert "rc=0" in proc.stdout
    assert "[ntfy suppressed] Title" in proc.stderr
    assert not log.exists(), "curl must not run when suppressed"


@pytest.mark.integration
@pytest.mark.parametrize("shell", SHELLS)
def test_topic_from_env_var_and_headers(shell, curl_stub, tmp_path):
    bin_dir, log = curl_stub
    env = _base_env(bin_dir, tmp_path) | {"NTFY_TOPIC": "my-topic"}
    proc = _run(shell, 'rq_notify "Ti" "bo" 4 "warning,chart"', env=env)
    assert proc.returncode == 0, proc.stderr
    args = log.read_text().splitlines()
    assert "https://ntfy.sh/my-topic" in args
    assert "Title: Ti" in args
    assert "Priority: 4" in args
    assert "Tags: warning,chart" in args
    assert "--max-time" in args and "5" in args


@pytest.mark.integration
@pytest.mark.parametrize("shell", SHELLS)
def test_topic_parsed_from_rq_root_env_file(shell, curl_stub, tmp_path):
    bin_dir, log = curl_stub
    (tmp_path / ".env").write_text('NTFY_TOPIC="file-topic"\nOTHER=x\n', encoding="utf-8")
    env = _base_env(bin_dir, tmp_path)  # no NTFY_TOPIC in env
    proc = _run(shell, 'rq_notify "Ti" "bo"', env=env)
    assert proc.returncode == 0, proc.stderr
    args = log.read_text().splitlines()
    assert "https://ntfy.sh/file-topic" in args
    # .env is parsed, not sourced: OTHER must not leak into the caller env.
    proc2 = _run(shell, 'rq_notify "Ti" "bo"; echo "OTHER=${OTHER:-unset}"', env=env)
    assert "OTHER=unset" in proc2.stdout


@pytest.mark.integration
@pytest.mark.parametrize("shell", SHELLS)
def test_default_topic_when_unconfigured(shell, curl_stub, tmp_path):
    bin_dir, log = curl_stub
    env = _base_env(bin_dir, tmp_path)  # no NTFY_TOPIC, no .env
    proc = _run(shell, 'rq_notify "Ti" "bo"', env=env)
    assert proc.returncode == 0, proc.stderr
    assert "https://ntfy.sh/renquant" in log.read_text().splitlines()


@pytest.mark.integration
@pytest.mark.parametrize("shell", SHELLS)
def test_never_fails_caller_when_curl_is_broken(shell, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    broken = bin_dir / "curl"
    broken.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
    broken.chmod(0o755)
    env = _base_env(bin_dir, tmp_path) | {"NTFY_TOPIC": "t"}
    proc = _run(shell, 'rq_notify "Ti" "bo"; echo "rc=$?"', env=env)
    assert proc.returncode == 0, proc.stderr
    assert "rc=0" in proc.stdout
