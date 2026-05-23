"""Regression guards for train_104 hardware saturation.

CLAUDE.md §5.10 requires long compute jobs to saturate the current Apple
Silicon machine.  The training entrypoint must not carry stale per-machine
thread counts from older hardware.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
THREAD_ENV_KEYS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]


def test_train_104_sets_thread_env_from_detected_core_count():
    """Importing train_104 must seed BLAS/OpenMP env vars from os.cpu_count()."""
    code = f"""
import json
import os
import runpy

keys = {THREAD_ENV_KEYS!r}
for key in keys:
    os.environ.pop(key, None)

runpy.run_path("scripts/train_104.py", run_name="__train_104_thread_test__")
print(json.dumps({{
    "expected": str(os.cpu_count() or 14),
    "actual": {{key: os.environ.get(key) for key in keys}},
}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr[:500]
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    assert payload["actual"] == {
        key: payload["expected"] for key in THREAD_ENV_KEYS
    }


def test_train_104_does_not_carry_stale_m2_thread_hardcode():
    src = (REPO_ROOT / "scripts" / "train_104.py").read_text()

    assert "M2 Pro has 10 cores" not in src
    assert '("OMP_NUM_THREADS", "10")' not in src
    assert "_os.cpu_count()" in src
