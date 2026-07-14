"""Test that umbrella kernel/ and pinned pipeline kernel stay in sync (F-6).

Catches NEW drift: files that were byte-identical (after import normalisation)
but have diverged. Pre-existing drift is allowlisted in check_kernel_parity.py
and does not fail this test. As drifted files are ported/unified, remove them
from the allowlist so re-drift is caught.

Skips cleanly when the pipeline repo is not checked out locally (CI without
sibling repos).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_kernel_parity.py"
PIPELINE_KERNEL = Path(__file__).resolve().parent.parent.parent / "renquant-pipeline" / "src" / "renquant_pipeline" / "kernel"


@pytest.mark.skipif(
    not PIPELINE_KERNEL.is_dir(),
    reason="renquant-pipeline not checked out locally",
)
def test_no_new_kernel_drift():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--verbose"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 2:
        pytest.skip(f"setup error: {result.stdout}")
    assert result.returncode == 0, (
        f"New kernel drift detected:\n{result.stdout}\n{result.stderr}"
    )
