"""env_sha provenance tests (eng plan §III.5; decision #110 milestone
'complete provenance'). Prototype: orchestrator scripts/engineering/
env_fingerprint.py."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.artifact_contract import build_run_bundle  # noqa: E402
from kernel.env_fingerprint import env_fingerprint  # noqa: E402


class TestEnvFingerprint:

    def test_stable_within_process(self):
        a, b = env_fingerprint(), env_fingerprint()
        assert a == b
        assert len(a["env_sha"]) == 16
        assert a["n_packages"] > 50  # production venv, not a bare interp

    def test_python_version_in_sha_inputs(self):
        fp = env_fingerprint()
        assert fp["python"] == sys.version.split()[0]


class TestBundleCarriesEnv:

    def test_run_bundle_has_env_sha(self, tmp_path):
        bundle = build_run_bundle(
            {"watchlist": ["MU"]}, tmp_path, run_id="r1", run_type="sim")
        assert bundle["env"]["env_sha"] == env_fingerprint()["env_sha"]
        assert bundle["env"]["n_packages"] > 0
