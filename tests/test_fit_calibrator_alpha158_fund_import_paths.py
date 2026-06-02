"""Regression guard: scripts/fit_calibrator_alpha158_fund.py --regime-filter must
import scripts.analyze_manifest_sanity_placebo when launched via its file path.

Source: PR #119 (Track A per-regime calibrator) codex HIGH blocker. Before the
fix, the module top inserted only `REPO/backtesting/renquant_104` into
sys.path. When invoked as `python scripts/fit_calibrator_alpha158_fund.py`,
sys.path[0] = "scripts" (the script's own directory), and `REPO` itself is
NOT on sys.path. The lazy import `from scripts.analyze_manifest_sanity_placebo
import build_regime_series` at line ~354 then raised
`ModuleNotFoundError: No module named 'scripts'` unless the caller had already
exported `PYTHONPATH=$REPO`.

The fix inserts BOTH `REPO` and `REPO/backtesting/renquant_104` at module
load time (same pattern as scripts/analyze_manifest_sanity_placebo.py:28-31).
This test simulates the file-path launch context and asserts the lazy import
succeeds without any PYTHONPATH help.

AUDIT REGRESSION GUARD: do not delete this test without removing the
--regime-filter knob from scripts/fit_calibrator_alpha158_fund.py.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "fit_calibrator_alpha158_fund.py"
SCRIPTS_DIR = REPO / "scripts"


class TestPR119RegimeFilterImportPathRegressionGuard:
    """Pin the --regime-filter import path against PYTHONPATH-leak regressions."""

    def test_script_module_adds_repo_to_sys_path(self):
        """After loading the script module, REPO must be on sys.path so the
        lazy `from scripts.analyze_manifest_sanity_placebo import …` resolves.

        Reproduces the launch context where Python sets sys.path[0] to the
        script's parent dir (scripts/) and nothing else. Pre-fix this test
        fails with ModuleNotFoundError; post-fix it passes.
        """
        assert SCRIPT.exists(), f"script missing: {SCRIPT}"

        # Strip PYTHONPATH so the only sys.path help comes from the script's
        # own top-of-module sys.path.insert calls.
        env = {
            k: v for k, v in os.environ.items()
            if k not in ("PYTHONPATH", "PYTHONSTARTUP")
        }
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        # Mimic `python scripts/fit_calibrator_alpha158_fund.py` by setting
        # sys.path[0] = the script's parent directory, then executing the
        # module body via runpy.run_path. After the body runs, we probe whether
        # the lazy import inside `if regime_filter:` would succeed by attempting
        # the same import statement.
        probe = (
            "import sys, runpy;"
            f"sys.path.insert(0, {str(SCRIPTS_DIR)!r});"
            # Load module body (does NOT call main; argparse only runs under
            # `if __name__ == '__main__'` guard at the bottom of the file).
            # runpy.run_path with run_name != '__main__' avoids triggering main.
            f"runpy.run_path({str(SCRIPT)!r}, run_name='_fit_calib_probe');"
            # Now reproduce the lazy import at line ~354.
            "from scripts.analyze_manifest_sanity_placebo import build_regime_series;"
            "print('OK')"
        )

        proc = subprocess.run(
            [sys.executable, "-c", probe],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO.parent),  # outside the repo so cwd doesn't help
        )

        # Pre-fix failure mode: ModuleNotFoundError: No module named 'scripts'
        assert proc.returncode == 0, (
            "PR #119 --regime-filter import-path regression:\n"
            f"stdout: {proc.stdout!r}\n"
            f"stderr: {proc.stderr!r}"
        )
        assert "OK" in proc.stdout

    def test_module_top_inserts_repo_into_sys_path(self):
        """Direct invariant: top-of-file sys.path insertions must include REPO.

        Lightweight backstop for the subprocess test — even if the subprocess
        harness is bypassed in some environment, this string check fails fast
        when someone reverts the fix.
        """
        text = SCRIPT.read_text()
        # The fix block inserts both REPO and the strategy dir.
        assert 'sys.path.insert(0, _s)' in text, (
            "scripts/fit_calibrator_alpha158_fund.py must keep the REPO + "
            "strategy-dir sys.path insertion block (PR #119 codex fix)."
        )
        assert '(REPO, REPO / "backtesting" / "renquant_104")' in text, (
            "scripts/fit_calibrator_alpha158_fund.py must insert REPO into "
            "sys.path so `from scripts.analyze_manifest_sanity_placebo …` "
            "resolves under the bare file-path launch context."
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
