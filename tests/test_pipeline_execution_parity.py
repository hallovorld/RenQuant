"""Test that renquant-pipeline and renquant-execution stay in parity on the
G3 Phase A registry items A2 (duplicated constants + ``compute_parent_intent_id``
golden vectors) and A3 (calendar-import inventory upper bound).

Catches drift: ``MIN_FRACTIONAL_NOTIONAL_USD`` diverging between
``renquant_pipeline.kernel.sizing`` and ``renquant_execution.broker``,
``compute_parent_intent_id`` producing different output for the same golden
vectors, or the pipeline kernel's non-canonical ``pandas_market_calendars``
import count growing past the baseline. Delegates entirely to
``scripts/check_pipeline_execution_parity.py`` (subprocess) rather than
re-implementing sibling resolution here, so there is exactly one place that
decides whether renquant-pipeline/renquant-execution (and their own
renquant-common/renquant-base-data/renquant-artifacts dependencies) are
available.

``.github/workflows/pipeline-execution-parity-ci.yml`` is the ONE job that
checks out renquant-pipeline, renquant-execution, and their dependencies as
siblings specifically so this comparison has something real to run against,
and sets ``RENQUANT_PIPELINE_EXECUTION_PARITY_STRICT=1``. In that job, the
script reporting "skipped" means one of its checkout steps failed to
provide a sibling it promised -- a real environment failure that must FAIL
this test, not skip it. A green skip there would look like the parity guard
ran and passed while it never compared pipeline and execution. Everywhere
else (local dev without the sibling repos, or any other CI job that doesn't
provision them), a skip is the documented, legitimate outcome.

Relocated from renquant-orchestrator PR #515
(``tests/test_cross_repo_parity.py``) per Codex review: orchestrator can see
these sibling repos locally on a developer machine, but its own CI has no
job that checks them out, so a green orchestrator build proved none of these
invariants. This repo owns the pins (``subrepos.lock.json``) and the strict
CI job, so it owns the canonical test.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "check_pipeline_execution_parity.py"
)

_STRICT = os.environ.get("RENQUANT_PIPELINE_EXECUTION_PARITY_STRICT") == "1"


def test_pipeline_execution_parity():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--verbose"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode == 3:
        # The script itself decided a sibling isn't available.
        if _STRICT:
            pytest.fail(
                "RENQUANT_PIPELINE_EXECUTION_PARITY_STRICT=1 but the parity "
                "check skipped instead of comparing pipeline/execution -- "
                "the pipeline-execution-parity-ci job is supposed to check "
                "out renquant-pipeline, renquant-execution, and their own "
                "renquant-common/renquant-base-data/renquant-artifacts "
                "dependencies as siblings; one of those checkout steps must "
                f"have failed or been misconfigured:\n{result.stdout}"
            )
        pytest.skip(f"sibling repo(s) not available: {result.stdout.strip()}")

    if result.returncode == 2:
        if _STRICT:
            pytest.fail(
                "setup error under strict CI mode "
                "(RENQUANT_PIPELINE_EXECUTION_PARITY_STRICT=1):"
                f"\n{result.stdout}\n{result.stderr}"
            )
        pytest.skip(f"setup error: {result.stdout}")

    assert result.returncode == 0, (
        f"pipeline/execution parity drift detected:\n{result.stdout}\n{result.stderr}"
    )
