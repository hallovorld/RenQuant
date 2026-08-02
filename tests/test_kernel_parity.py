"""Test that umbrella kernel/ and pinned pipeline kernel stay in sync (F-6).

Catches NEW drift: files that were byte-identical (after import normalisation)
but have diverged. Pre-existing drift is allowlisted in check_kernel_parity.py
and does not fail this test. As drifted files are ported/unified, remove them
from the allowlist so re-drift is caught.

Delegates entirely to scripts/check_kernel_parity.py (subprocess) rather than
re-implementing path resolution here, so there is exactly one place that
decides whether the pipeline kernel is available.

``.github/workflows/kernel-parity-ci.yml`` is the ONE job that checks out
``renquant-pipeline`` as a sibling specifically so this comparison has
something real to run against, and sets ``RENQUANT_KERNEL_PARITY_STRICT=1``.
In that job, the script reporting "skipped" means its own checkout step
failed to provide the sibling it promised -- a real environment failure that
must FAIL this test, not skip it. A green skip there would look like the
parity guard ran and passed while it never compared the two kernel trees.
Everywhere else (local dev without the sibling repo, or any other CI job
that doesn't provision it), a skip is the documented, legitimate outcome.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_kernel_parity.py"

_STRICT = os.environ.get("RENQUANT_KERNEL_PARITY_STRICT") == "1"


def test_no_new_kernel_drift():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--verbose"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode == 3:
        # The script itself decided the pipeline kernel isn't available.
        if _STRICT:
            pytest.fail(
                "RENQUANT_KERNEL_PARITY_STRICT=1 but the parity check "
                "skipped instead of comparing the kernels -- the "
                "kernel-parity-ci job is supposed to check out "
                "renquant-pipeline as a sibling; that checkout step must "
                f"have failed or been misconfigured:\n{result.stdout}"
            )
        pytest.skip(f"pipeline kernel not available: {result.stdout.strip()}")

    if result.returncode == 2:
        if _STRICT:
            pytest.fail(
                f"setup error under strict CI mode (RENQUANT_KERNEL_PARITY_STRICT=1):"
                f"\n{result.stdout}\n{result.stderr}"
            )
        pytest.skip(f"setup error: {result.stdout}")

    assert result.returncode == 0, (
        f"New kernel drift detected:\n{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Resolver pin-verification units (2026-08-02 hardening): the resolver must
# refuse any candidate checkout whose HEAD is not the locked pipeline commit
# — the measured failure was a sibling at a14dad11 vs pin 60871e24 reading
# two genuinely-drifted files as converged (a wrong-object measurement).
# ---------------------------------------------------------------------------

def _mk_repo(root: Path, kernel_content: str = "x = 1\n") -> str:
    (root / "src" / "renquant_pipeline" / "kernel").mkdir(parents=True)
    (root / "src" / "renquant_pipeline" / "kernel" / "a.py").write_text(kernel_content)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "commit", "-q", "-m", "c"]):
        subprocess.run(cmd, cwd=root, env=env, check=True, capture_output=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, env=env,
                          check=True, capture_output=True, text=True)
    return head.stdout.strip()


def _load_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ckp_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolver_accepts_a_candidate_at_the_locked_commit(tmp_path, monkeypatch):
    repo = tmp_path / "renquant-pipeline"
    repo.mkdir()
    head = _mk_repo(repo)
    lock = tmp_path / "subrepos.lock.json"
    lock.write_text(
        '{"subrepos": [{"name": "renquant-pipeline", '
        f'"commit": "{head}", "local_path": "{repo}"}}]}}'
    )
    mod = _load_module()
    monkeypatch.setattr(mod, "LOCK_FILE", lock)
    monkeypatch.setattr(mod, "UMBRELLA_ROOT", tmp_path / "nowhere")
    monkeypatch.delenv("RENQUANT_PIPELINE_KERNEL_PATH", raising=False)
    resolved = mod._resolve_pipeline_kernel()
    assert resolved == repo / "src" / "renquant_pipeline" / "kernel"


def test_resolver_refuses_a_candidate_at_the_wrong_commit(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "renquant-pipeline"
    repo.mkdir()
    _mk_repo(repo)
    lock = tmp_path / "subrepos.lock.json"
    lock.write_text(
        '{"subrepos": [{"name": "renquant-pipeline", '
        f'"commit": "{"0" * 40}", "local_path": "{repo}"}}]}}'
    )
    mod = _load_module()
    monkeypatch.setattr(mod, "LOCK_FILE", lock)
    monkeypatch.setattr(mod, "UMBRELLA_ROOT", tmp_path / "nowhere")
    monkeypatch.delenv("RENQUANT_PIPELINE_KERNEL_PATH", raising=False)
    assert mod._resolve_pipeline_kernel() is None
    assert "wrong-object" in capsys.readouterr().err


def test_resolver_prefers_the_pinned_runtime_clone(tmp_path, monkeypatch):
    umbrella = tmp_path / "RenQuant"
    runtime = umbrella / ".subrepo_runtime" / "repos" / "renquant-pipeline"
    runtime.mkdir(parents=True)
    head = _mk_repo(runtime)
    stale = tmp_path / "renquant-pipeline"
    stale.mkdir()
    _mk_repo(stale, kernel_content="y = 2\n")  # different HEAD by content
    lock = tmp_path / "subrepos.lock.json"
    lock.write_text(
        '{"subrepos": [{"name": "renquant-pipeline", '
        f'"commit": "{head}", "local_path": "{stale}"}}]}}'
    )
    mod = _load_module()
    monkeypatch.setattr(mod, "LOCK_FILE", lock)
    monkeypatch.setattr(mod, "UMBRELLA_ROOT", umbrella)
    monkeypatch.delenv("RENQUANT_PIPELINE_KERNEL_PATH", raising=False)
    resolved = mod._resolve_pipeline_kernel()
    assert resolved == runtime / "src" / "renquant_pipeline" / "kernel"
