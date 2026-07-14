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
