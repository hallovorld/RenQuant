"""Regression guard for the RFC#210 freshness-fallback promote (PR #559).

Source incident: PR #559 round 1 BLOCKER. Step 4b's fallback-specific pair
promote called the SHARED `renquant_backtesting.forensics.model_acceptance.
promote()` helper, whose internal `_check_wf_gate()` unconditionally refuses
any staging artifact stamped `wf_gate_metadata.passed=False` — exactly what
the RFC#210 fallback stamp intentionally is (`promotion_basis=
freshness_fallback_rfc210`, `passed=False` by design). Every fallback-promote
attempt therefore raised `ValueError: promote: refused — wf_gate_metadata.
passed=False` and Step 4b always terminated in "Fallback promote FAILED",
never actually arming the fallback the whole PR exists to wire up.

This test drives the real production script through a forced WF-gate REJECT
with a stubbed `renquant_backtesting.wf_gate.freshness_fallback` CLI (the
backtesting#102 module the live pin does not carry yet) that stamps the
staging artifact FALLBACK_PROMOTE, and asserts the active artifact actually
receives the promotion_basis stamp — proving Step 4b's own atomic swap runs
instead of the gate-passed `promote()` helper.
"""
from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _weekly_promote_fixture as fixture  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "weekly_wf_promote.sh"

_FRESHNESS_FALLBACK_SHIM = '''
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prod")
    parser.add_argument("--staging", required=True)
    parser.add_argument("--stamp", action="store_true")
    args = parser.parse_args()

    staging_path = Path(args.staging)
    payload = json.loads(staging_path.read_text())
    meta = payload.setdefault("metadata", {})
    gate = meta.setdefault("wf_gate_metadata", {})
    gate["passed"] = False
    if args.stamp:
        meta["promotion_basis"] = "freshness_fallback_rfc210"
        staging_path.write_text(json.dumps(payload))
    print(json.dumps({"verdict": "FALLBACK_PROMOTE", "stamped": bool(args.stamp)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


_MODEL_ACCEPTANCE_SHIM = '''
"""Faithful-enough replica of renquant_backtesting.forensics.model_acceptance
.promote()'s hard gate (_check_wf_gate), for regression-testing PR #559's
fallback-promote fix without depending on the real pinned package version."""
import json
import os
import shutil
from pathlib import Path


def promote(staging_path, active_path):
    staging_path = Path(staging_path)
    active_path = Path(active_path)
    data = json.loads(staging_path.read_text())
    md = data.get("metadata", {}) or {}
    wf = md.get("wf_gate_metadata")
    if not isinstance(wf, dict):
        wf = data.get("wf_gate_metadata")
    if not isinstance(wf, dict) or not bool(wf.get("passed")):
        raise ValueError(
            f"promote: refused — wf_gate_metadata.passed=False on {staging_path.name}")
    previous_path = active_path.with_suffix(".previous.json")
    temp_active = active_path.with_suffix(".incoming.json")
    shutil.copy2(str(staging_path), str(temp_active))
    if active_path.exists():
        os.replace(str(active_path), str(previous_path))
    os.replace(str(temp_active), str(active_path))
    staging_path.unlink(missing_ok=True)
'''


def _write_freshness_fallback_shim(pythonpath_dir: Path) -> None:
    wf_gate_pkg = pythonpath_dir / "renquant_backtesting" / "wf_gate"
    forensics_pkg = pythonpath_dir / "renquant_backtesting" / "forensics"
    wf_gate_pkg.mkdir(parents=True, exist_ok=True)
    forensics_pkg.mkdir(parents=True, exist_ok=True)
    (pythonpath_dir / "renquant_backtesting" / "__init__.py").write_text("", encoding="utf-8")
    (wf_gate_pkg / "__init__.py").write_text("", encoding="utf-8")
    (wf_gate_pkg / "freshness_fallback.py").write_text(_FRESHNESS_FALLBACK_SHIM, encoding="utf-8")
    (forensics_pkg / "__init__.py").write_text("", encoding="utf-8")
    # Faithful-enough replica of the REAL hard gate (_check_wf_gate) so a
    # pre-fix wrapper genuinely reproduces PR #559's BLOCKER instead of
    # silently falling through to the fixture's trivial kernel.model_
    # acceptance stub (a plain copy with no gate at all) via the wrapper's
    # `except Exception: ... from kernel.model_acceptance import promote`
    # fallback — that fallback only fires when THIS import fails, so it must
    # succeed here for the regression to reproduce.
    (forensics_pkg / "model_acceptance.py").write_text(
        _MODEL_ACCEPTANCE_SHIM, encoding="utf-8")


def _force_wf_gate_reject(root: Path) -> None:
    path = root / "scripts" / "run_wf_gate.py"
    path.write_text(f"#!{sys.executable}\nimport sys\nsys.exit(1)\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run(root: Path, notify_log: Path, lock_file: Path, pythonpath_dir: Path) -> subprocess.CompletedProcess:
    env = {
        "RQ_WEEKLY_PROMOTE_REPO_DIR": str(root),
        "RQ_WEEKLY_PROMOTE_PYTHON": sys.executable,
        "RQ_WEEKLY_PROMOTE_NOTIFY_LOG": str(notify_log),
        "RQ_WEEKLY_PROMOTE_LOCK_FILE": str(lock_file),
        "RQ_WF_GATE_RUNNER": "umbrella",
        "PYTHONPATH": str(pythonpath_dir),
        "PATH": f"{fixture.shim_bin_dir(root)}:/usr/bin:/bin:/usr/local/bin",
        "HOME": str(root),
    }
    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=120)


def test_fallback_promote_swaps_active_artifact_instead_of_failing(tmp_path):
    root = tmp_path / "repo"
    mod = fixture.build_fixture_repo(root)
    _force_wf_gate_reject(root)
    pythonpath_dir = tmp_path / "pythonpath_shim"
    _write_freshness_fallback_shim(pythonpath_dir)

    notify_log = tmp_path / "notify.log"
    lock_file = tmp_path / "weekly.lock"
    result = _run(root, notify_log, lock_file, pythonpath_dir)

    log_tail = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (root / "logs" / "weekly_wf_promote").glob("*.log"))
    notifications = notify_log.read_text(encoding="utf-8") if notify_log.exists() else ""

    # Pre-fix, promote() always raised on the passed=False stamp; the
    # fallback-specific block must not hit that failure mode.
    assert "Fallback promote FAILED" not in log_tail, log_tail[-3000:]
    assert "RFC#210 fallback promote failed" not in notifications, notifications

    active_artifact = (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod"
                        / fixture.ACTIVE_ARTIFACT_NAME)
    active_meta = json.loads(active_artifact.read_text(encoding="utf-8")).get("metadata", {})
    assert active_meta.get("promotion_basis") == "freshness_fallback_rfc210", (
        f"active artifact was not swapped by the fallback promote; "
        f"metadata={active_meta}; log tail:\n{log_tail[-3000:]}")
    assert active_meta.get("wf_gate_metadata", {}).get("passed") is False

    # Step 7's snapshot backstop correctly flags the gate-verdict change
    # (passed True -> False) as drift needing `make snapshot`; that is
    # separate, expected behavior — not a regression of this fix — so the
    # overall exit code is not asserted here.
    assert not lock_file.exists(), "lock file must be released on exit"
