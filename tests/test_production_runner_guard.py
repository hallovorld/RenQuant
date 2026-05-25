"""Legacy production runner must not bypass the shared trading pipeline."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def test_legacy_production_runner_execute_is_fail_closed_before_artifact_load(tmp_path):
    """`--execute` used to submit Alpaca paper orders directly.

    That path bypasses live.runner, InferencePipeline, QP admission, risk gates,
    and decision_trace DB. It must fail before loading model artifacts so a
    stale artifact path cannot mask the contract violation.
    """
    missing_artifact = tmp_path / "missing-artifact.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "production_runner.py"),
            "--execute",
            "--broker",
            "alpaca-paper",
            "--artifact",
            str(missing_artifact),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "bypasses live.runner" in proc.stderr
    assert str(missing_artifact) not in proc.stderr

