"""Environment fingerprint — env_sha for the run bundle / DRPH fingerprint.

Design: renquant-orchestrator
doc/research/2026-06-12-engineering-architecture-deep-plan.md §III.5
(env reproducibility) + decision 2026-06-12-engineering-before-model-research
milestone "complete provenance"; prototype:
scripts/engineering/env_fingerprint.py (orchestrator PR #116 batch).

Why: the run fingerprint was incomplete without it — same code + same
data + DIFFERENT numpy can score differently (shared-.venv mutation
hazard). With env_sha in every run bundle, "were these two runs
comparable?" includes the dependency set, and a silent venv mutation
shows up as an env_sha change in the bundle diff.

Production differences vs the prototype: fingerprints the RUNNING
interpreter via importlib.metadata (no pip-freeze subprocess), and is
cached per process — the cost is paid once, not per run bundle.
"""
from __future__ import annotations

import functools
import hashlib
import sys


@functools.lru_cache(maxsize=1)
def env_fingerprint() -> dict:
    """{env_sha, python, n_packages} for the running interpreter.

    Deliberately excludes the package LIST from the return value (the
    bundle stays compact); the sha covers name==version for every
    installed distribution plus the python version.
    """
    from importlib import metadata  # noqa: PLC0415

    pkgs = sorted(
        f"{d.metadata['Name']}=={d.version}"
        for d in metadata.distributions()
        if d.metadata["Name"]
    )
    blob = sys.version + "\n" + "\n".join(pkgs)
    return {
        "env_sha": hashlib.sha256(blob.encode()).hexdigest()[:16],
        "python": sys.version.split()[0],
        "n_packages": len(pkgs),
    }
