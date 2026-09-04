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

_FRESHNESS_FALLBACK_SHIM_TEMPLATE = '''
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

    if {refuse!r}:
        payload = {refuse_payload!r}
        print(payload if payload
              else json.dumps({{"verdict": "REFUSE", "reason": "test-forced-refuse"}}))
        return 1

    staging_path = Path(args.staging)
    payload = json.loads(staging_path.read_text())
    meta = payload.setdefault("metadata", {{}})
    gate = meta.setdefault("wf_gate_metadata", {{}})
    gate["passed"] = {gate_passed!r}
    if args.stamp:
        {promotion_basis_stmt}
        staging_path.write_text(json.dumps(payload))
    print(json.dumps({{"verdict": "FALLBACK_PROMOTE", "stamped": bool(args.stamp)}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

# Default (arm-and-succeed) shim body — kept as a plain constant so the
# original happy-path test reads exactly as before.
_FRESHNESS_FALLBACK_SHIM = _FRESHNESS_FALLBACK_SHIM_TEMPLATE.format(
    refuse_payload="",
    refuse=False, gate_passed=False,
    promotion_basis_stmt='meta["promotion_basis"] = "freshness_fallback_rfc210"')


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


def _write_freshness_fallback_shim(
    pythonpath_dir: Path,
    *,
    refuse: bool = False,
    refuse_payload: str = "",
    gate_passed: bool = False,
    stamp_promotion_basis: bool = True,
) -> None:
    """Write the `renquant_backtesting.wf_gate.freshness_fallback` shim.

    Parametrized so round-2 hermetic coverage can drive every verdict shape
    the real CLI can hand back to Step 4b:
      - ``refuse=True``               -> CLI exits 1, staging file untouched
        (REFUSE verdict; also stands in for module-unavailable in effect).
      - ``gate_passed`` / ``stamp_promotion_basis`` -> what the CLI stamps
        onto the staging artifact when it DOES exit 0 with ``--stamp``.
        The happy path is ``gate_passed=False, stamp_promotion_basis=True``
        (the only combination Step 4b's own swap validation accepts);
        the other combinations reproduce a malformed/buggy fallback CLI so
        Step 4b's validation-before-mutation guard can be pinned.
    """
    wf_gate_pkg = pythonpath_dir / "renquant_backtesting" / "wf_gate"
    forensics_pkg = pythonpath_dir / "renquant_backtesting" / "forensics"
    wf_gate_pkg.mkdir(parents=True, exist_ok=True)
    forensics_pkg.mkdir(parents=True, exist_ok=True)
    (pythonpath_dir / "renquant_backtesting" / "__init__.py").write_text("", encoding="utf-8")
    (wf_gate_pkg / "__init__.py").write_text("", encoding="utf-8")
    promotion_basis_stmt = (
        'meta["promotion_basis"] = "freshness_fallback_rfc210"'
        if stamp_promotion_basis else "pass"
    )
    shim_src = _FRESHNESS_FALLBACK_SHIM_TEMPLATE.format(
        refuse=refuse, refuse_payload=refuse_payload, gate_passed=gate_passed,
        promotion_basis_stmt=promotion_basis_stmt)
    (wf_gate_pkg / "freshness_fallback.py").write_text(shim_src, encoding="utf-8")
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


def _force_wf_gate_reject(root: Path, *, cut_returncode: int = 0) -> None:
    """Stub `run_wf_gate.py` that REJECTS (exit 1) and, like the real gate,
    stamps `metadata.wf_gate_metadata` on the `--artifact` it was given:
    `passed=False` plus three cuts carrying `cut_returncode`. The default 0
    is an EXECUTED reject (the shape every fallback test models);
    `cut_returncode=1` is the 2026-09-01..03 shape — the simulation crashed
    — which Step 4a must refuse to treat as a verdict."""
    path = root / "scripts" / "run_wf_gate.py"
    path.write_text(f"""#!{sys.executable}
import json, sys
args = sys.argv[1:]
artifact = args[args.index("--artifact") + 1] if "--artifact" in args else None
if artifact:
    payload = json.load(open(artifact))
    gate = payload.setdefault("metadata", {{}}).setdefault("wf_gate_metadata", {{}})
    gate["passed"] = False
    gate["wf_reason"] = ("3/3 sim cuts failed execution" if {cut_returncode!r} != 0
                         else "FAIL: test-forced reject")
    gate["cuts"] = [
        {{"start": s, "end": e, "returncode": {cut_returncode!r}, "sharpe": None, "apy": None}}
        for s, e in (("2024-01-02", "2024-12-31"), ("2024-07-01", "2025-06-30"),
                     ("2025-01-02", "2025-12-31"))
    ]
    json.dump(payload, open(artifact, "w"))
sys.exit(1)
""", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run(root: Path, notify_log: Path, lock_file: Path, pythonpath_dir: Path, *, armed_consumer: bool = True) -> subprocess.CompletedProcess:
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
    # Arm the CONSUMER side of the dual-contract check (codex #559 round-1
    # second demand): Step 4b refuses unless the orchestrator emitter
    # contract carries the FALLBACK-PROMOTED action line. Fixture contract by
    # default; the disarm case passes armed_consumer=False.
    orch_run = root / "fixture_orch_run"
    contract = orch_run / "ops" / "renquant104" / "emitter_contract.json"
    contract.parent.mkdir(parents=True, exist_ok=True)
    if armed_consumer:
        contract.write_text(
            '{"lines": [{"job": "weekly-wf-promote", "kind": "action", '
            '"template": "=== weekly_wf_promote FALLBACK-PROMOTED (rfc210) ==="}]}',
            encoding="utf-8")
    else:
        contract.write_text('{"lines": []}', encoding="utf-8")
    env["RQ_ORCH_RUN_DIR"] = str(orch_run)
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

    # Codex review round 2: Step 7's snapshot backstop correctly flags the
    # gate-verdict change (passed True -> False) as drift needing `make
    # snapshot` — but that must not suppress the action/notification
    # contract for a promotion that already happened, nor leave the run
    # exiting through a failure path. The fallback path now handles its own
    # snapshot-staleness follow-up (WARN, not a hard fail) and reaches the
    # FALLBACK-PROMOTED literal + notification unconditionally.
    assert "WEEKLY-PROMOTE — SNAPSHOT STALE" in notifications, notifications
    assert "WEEKLY-FALLBACK-PROMOTE" in notifications, notifications
    assert "=== weekly_wf_promote FALLBACK-PROMOTED (rfc210)" in log_tail, log_tail[-3000:]
    assert "(metadata parse failed)" not in log_tail, (
        "Step 5's GATE_SUMMARY must read back from the active artifact once "
        "Step 4b unlinks the staging path: " + log_tail[-3000:])
    assert result.returncode == 0, (
        f"a successful fallback promotion must exit 0 even when the "
        f"snapshot backstop separately flags staleness; stdout tail:\n"
        f"{log_tail[-3000:]}")
    assert not lock_file.exists(), "lock file must be released on exit"


def test_fallback_promote_swaps_calibrator_together_with_model(tmp_path):
    root = tmp_path / "repo"
    mod = fixture.build_fixture_repo(root)
    _force_wf_gate_reject(root)
    pythonpath_dir = tmp_path / "pythonpath_shim"
    _write_freshness_fallback_shim(pythonpath_dir)

    active_cal = (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod"
                  / fixture.ACTIVE_CALIBRATOR_NAME)
    # Seed a marker the retrain stub's fixed output does NOT carry, so a
    # content match after the run genuinely proves a file swap happened
    # (the retrain stub's calibrator payload is otherwise byte-identical to
    # the fixture's initial active calibrator).
    marker_payload = json.loads(active_cal.read_text(encoding="utf-8"))
    marker_payload["pre_promote_marker"] = True
    active_cal.write_text(json.dumps(marker_payload), encoding="utf-8")

    notify_log = tmp_path / "notify.log"
    lock_file = tmp_path / "weekly.lock"
    result = _run(root, notify_log, lock_file, pythonpath_dir)
    log_tail = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (root / "logs" / "weekly_wf_promote").glob("*.log"))

    active_artifact = (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod"
                        / fixture.ACTIVE_ARTIFACT_NAME)
    active_meta = json.loads(active_artifact.read_text(encoding="utf-8")).get("metadata", {})
    assert active_meta.get("promotion_basis") == "freshness_fallback_rfc210", log_tail[-3000:]

    after_cal = json.loads(active_cal.read_text(encoding="utf-8"))
    assert "pre_promote_marker" not in after_cal, (
        f"active calibrator was not overwritten by the fallback promote "
        f"(marker survived): {after_cal}; log tail:\n{log_tail[-3000:]}")
    assert after_cal.get("kind") == "global_panel_calibration"
    assert result.returncode == 0, log_tail[-3000:]


def test_fallback_verdict_json_preserved_on_disk(tmp_path):
    root = tmp_path / "repo"
    fixture.build_fixture_repo(root)
    _force_wf_gate_reject(root)
    pythonpath_dir = tmp_path / "pythonpath_shim"
    _write_freshness_fallback_shim(pythonpath_dir)

    notify_log = tmp_path / "notify.log"
    lock_file = tmp_path / "weekly.lock"
    _run(root, notify_log, lock_file, pythonpath_dir)

    verdict_files = list((root / "logs" / "weekly_wf_promote").glob("*.fallback_verdict.json"))
    assert len(verdict_files) == 1, (
        f"expected exactly one fallback verdict file preserved per run, "
        f"found: {verdict_files}")
    verdict = json.loads(verdict_files[0].read_text(encoding="utf-8"))
    assert verdict.get("verdict") == "FALLBACK_PROMOTE", verdict
    assert verdict.get("stamped") is True, verdict


def test_module_unavailable_leaves_active_artifacts_unchanged(tmp_path):
    """The pinned backtesting runtime predates #102: the fallback module
    import fails, Step 4b must treat this exactly like REFUSE — production
    stays on the prior artifact/calibrator, no promotion of any kind."""
    root = tmp_path / "repo"
    mod = fixture.build_fixture_repo(root)
    _force_wf_gate_reject(root)
    # No shim written at all: renquant_backtesting.wf_gate.freshness_fallback
    # genuinely does not exist under this PYTHONPATH.
    pythonpath_dir = tmp_path / "pythonpath_shim"
    pythonpath_dir.mkdir(parents=True)

    active_artifact = (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod"
                        / fixture.ACTIVE_ARTIFACT_NAME)
    active_cal = (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod"
                  / fixture.ACTIVE_CALIBRATOR_NAME)
    before_artifact = active_artifact.read_text(encoding="utf-8")
    before_cal = active_cal.read_text(encoding="utf-8")

    notify_log = tmp_path / "notify.log"
    lock_file = tmp_path / "weekly.lock"
    result = _run(root, notify_log, lock_file, pythonpath_dir)
    log_tail = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (root / "logs" / "weekly_wf_promote").glob("*.log"))
    notifications = notify_log.read_text(encoding="utf-8") if notify_log.exists() else ""

    assert "RFC#210 fallback UNAVAILABLE" in log_tail, log_tail[-3000:]
    assert active_artifact.read_text(encoding="utf-8") == before_artifact, (
        "module-unavailable must not touch the active artifact")
    assert active_cal.read_text(encoding="utf-8") == before_cal, (
        "module-unavailable must not touch the active calibrator")
    assert "WEEKLY-REJECT" in notifications, notifications
    assert result.returncode == 1, log_tail[-3000:]
    assert not lock_file.exists(), "lock file must be released on exit"


def test_refuse_verdict_leaves_active_artifacts_unchanged(tmp_path):
    root = tmp_path / "repo"
    mod = fixture.build_fixture_repo(root)
    _force_wf_gate_reject(root)
    pythonpath_dir = tmp_path / "pythonpath_shim"
    _write_freshness_fallback_shim(pythonpath_dir, refuse=True)

    active_artifact = (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod"
                        / fixture.ACTIVE_ARTIFACT_NAME)
    active_cal = (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod"
                  / fixture.ACTIVE_CALIBRATOR_NAME)
    before_artifact = active_artifact.read_text(encoding="utf-8")
    before_cal = active_cal.read_text(encoding="utf-8")

    notify_log = tmp_path / "notify.log"
    lock_file = tmp_path / "weekly.lock"
    result = _run(root, notify_log, lock_file, pythonpath_dir)
    log_tail = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (root / "logs" / "weekly_wf_promote").glob("*.log"))
    notifications = notify_log.read_text(encoding="utf-8") if notify_log.exists() else ""

    assert "RFC#210 fallback verdict: REFUSE" in log_tail, log_tail[-3000:]
    assert active_artifact.read_text(encoding="utf-8") == before_artifact, (
        "a REFUSE verdict must not touch the active artifact")
    assert active_cal.read_text(encoding="utf-8") == before_cal, (
        "a REFUSE verdict must not touch the active calibrator")
    assert "WEEKLY-REJECT" in notifications, notifications
    assert result.returncode == 1, log_tail[-3000:]
    assert not lock_file.exists(), "lock file must be released on exit"


def test_crashed_simulation_alarms_before_the_fallback_is_consulted(tmp_path):
    """2026-09-01..03: the gate's three cuts died inside the sim
    (`cuts[*].returncode = 1`, "3/3 sim cuts failed execution"), the
    wrapper took the ordinary reject branch, the fallback refused on
    prod-fresh, and the run reported "governance nominal, calm notify,
    exit 0" for three days. Step 4a: a crashed simulation is not a verdict —
    alarm WEEKLY-FAIL, exit 1, and never consult the fallback (here armed to
    PROMOTE, so consulting it would be visible as a swapped artifact)."""
    root = tmp_path / "repo"
    mod = fixture.build_fixture_repo(root)
    _force_wf_gate_reject(root, cut_returncode=1)
    pythonpath_dir = tmp_path / "pythonpath_shim"
    _write_freshness_fallback_shim(pythonpath_dir)   # would FALLBACK_PROMOTE if asked

    active_artifact = (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod"
                        / fixture.ACTIVE_ARTIFACT_NAME)
    active_cal = (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod"
                  / fixture.ACTIVE_CALIBRATOR_NAME)
    before_artifact = active_artifact.read_text(encoding="utf-8")
    before_cal = active_cal.read_text(encoding="utf-8")

    notify_log = tmp_path / "notify.log"
    lock_file = tmp_path / "weekly.lock"
    result = _run(root, notify_log, lock_file, pythonpath_dir)
    log_tail = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (root / "logs" / "weekly_wf_promote").glob("*.log"))
    notifications = notify_log.read_text(encoding="utf-8") if notify_log.exists() else ""

    assert "WF-SIM-DID-NOT-RUN|3/3 cuts did not execute" in log_tail, log_tail[-3000:]
    assert "the simulation crashed, so no verdict exists to fall back from" in log_tail
    assert "RFC#210 fallback verdict" not in log_tail, "the fallback must not be consulted"
    assert "Reject disposition" not in log_tail, "a crash is not a reject disposition"
    assert "WEEKLY-FAIL (WF simulation crashed)" in notifications, notifications
    assert "WEEKLY-REJECT" not in notifications, notifications
    assert active_artifact.read_text(encoding="utf-8") == before_artifact
    assert active_cal.read_text(encoding="utf-8") == before_cal
    assert result.returncode == 1, log_tail[-3000:]
    assert not lock_file.exists(), "lock file must be released on exit"


def test_executed_reject_passes_step_4a_and_reaches_the_fallback(tmp_path):
    """Positive control for Step 4a: an executed reject (rc 0 on every cut)
    is a verdict and the run proceeds to the fallback exactly as before."""
    root = tmp_path / "repo"
    fixture.build_fixture_repo(root)
    _force_wf_gate_reject(root, cut_returncode=0)
    pythonpath_dir = tmp_path / "pythonpath_shim"
    _write_freshness_fallback_shim(pythonpath_dir, refuse=True)
    notify_log = tmp_path / "notify.log"
    lock_file = tmp_path / "weekly.lock"
    _run(root, notify_log, lock_file, pythonpath_dir)
    log_tail = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (root / "logs" / "weekly_wf_promote").glob("*.log"))
    assert "WF-SIM-RAN|all 3 cuts executed (returncode 0)" in log_tail, log_tail[-3000:]
    assert "RFC#210 fallback verdict: REFUSE" in log_tail, log_tail[-3000:]


def test_malformed_missing_promotion_basis_refuses_swap(tmp_path):
    """A fallback CLI that exits 0/--stamp but never actually writes the
    promotion_basis stamp must not be trusted — Step 4b's own validation is
    the license, not the CLI's exit code alone."""
    root = tmp_path / "repo"
    mod = fixture.build_fixture_repo(root)
    _force_wf_gate_reject(root)
    pythonpath_dir = tmp_path / "pythonpath_shim"
    _write_freshness_fallback_shim(pythonpath_dir, stamp_promotion_basis=False)

    active_artifact = (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod"
                        / fixture.ACTIVE_ARTIFACT_NAME)
    active_cal = (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod"
                  / fixture.ACTIVE_CALIBRATOR_NAME)
    before_artifact = active_artifact.read_text(encoding="utf-8")
    before_cal = active_cal.read_text(encoding="utf-8")

    notify_log = tmp_path / "notify.log"
    lock_file = tmp_path / "weekly.lock"
    result = _run(root, notify_log, lock_file, pythonpath_dir)
    log_tail = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (root / "logs" / "weekly_wf_promote").glob("*.log"))
    notifications = notify_log.read_text(encoding="utf-8") if notify_log.exists() else ""

    assert "Fallback promote FAILED" in log_tail, log_tail[-3000:]
    assert active_artifact.read_text(encoding="utf-8") == before_artifact, (
        "a missing promotion_basis stamp must not swap the active artifact")
    assert active_cal.read_text(encoding="utf-8") == before_cal, (
        "a missing promotion_basis stamp must not swap the active calibrator")
    assert "RFC#210 fallback promote failed" in notifications, notifications
    assert result.returncode == 1, log_tail[-3000:]
    assert not lock_file.exists(), "lock file must be released on exit"


def test_malformed_passed_not_false_refuses_swap(tmp_path):
    """The fallback license REQUIRES an explicitly rejected candidate
    (passed=False). If the CLI stamps promotion_basis but leaves passed
    True (or any non-False value), Step 4b must refuse the swap — that
    combination should never occur naturally and signals a broken CLI."""
    root = tmp_path / "repo"
    mod = fixture.build_fixture_repo(root)
    _force_wf_gate_reject(root)
    pythonpath_dir = tmp_path / "pythonpath_shim"
    _write_freshness_fallback_shim(pythonpath_dir, gate_passed=True)

    active_artifact = (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod"
                        / fixture.ACTIVE_ARTIFACT_NAME)
    active_cal = (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod"
                  / fixture.ACTIVE_CALIBRATOR_NAME)
    before_artifact = active_artifact.read_text(encoding="utf-8")
    before_cal = active_cal.read_text(encoding="utf-8")

    notify_log = tmp_path / "notify.log"
    lock_file = tmp_path / "weekly.lock"
    result = _run(root, notify_log, lock_file, pythonpath_dir)
    log_tail = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (root / "logs" / "weekly_wf_promote").glob("*.log"))
    notifications = notify_log.read_text(encoding="utf-8") if notify_log.exists() else ""

    assert "Fallback promote FAILED" in log_tail, log_tail[-3000:]
    assert active_artifact.read_text(encoding="utf-8") == before_artifact, (
        "a passed!=False stamp must not swap the active artifact")
    assert active_cal.read_text(encoding="utf-8") == before_cal, (
        "a passed!=False stamp must not swap the active calibrator")
    assert "RFC#210 fallback promote failed" in notifications, notifications
    assert result.returncode == 1, log_tail[-3000:]
    assert not lock_file.exists(), "lock file must be released on exit"


def test_consumer_contract_absent_disarms_loudly(tmp_path):
    """[codex on #559 round 1, second demand] provider armed but the
    orchestrator action-consumer contract absent → a fallback promotion
    would be recorded as a silent-refusal incident; Step 4b must refuse
    loudly and leave both active artifacts byte-unchanged."""
    root = tmp_path / "repo"
    mod = fixture.build_fixture_repo(root)
    _force_wf_gate_reject(root)
    pythonpath_dir = tmp_path / "pythonpath_shim"
    _write_freshness_fallback_shim(pythonpath_dir)
    active_art = (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod"
                  / fixture.ACTIVE_ARTIFACT_NAME)
    active_cal = (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod"
                  / fixture.ACTIVE_CALIBRATOR_NAME)
    before = (active_art.read_bytes(), active_cal.read_bytes())
    notify_log = tmp_path / "notify.log"
    lock_file = tmp_path / "weekly.lock"
    result = _run(root, notify_log, lock_file, pythonpath_dir,
                  armed_consumer=False)
    log_tail = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (root / "logs" / "weekly_wf_promote").glob("*.log"))
    assert result.returncode == 1, log_tail[-2000:]
    assert "RFC#210 fallback DISARMED" in log_tail
    notifications = notify_log.read_text(encoding="utf-8") if notify_log.exists() else ""
    assert "WEEKLY-REJECT" in notifications
    assert (active_art.read_bytes(), active_cal.read_bytes()) == before


# ── 2026-08-04: manual-run session-pin passthrough ──────────────────────────

import re as _re
import subprocess as _sp


def _extract_pin_block() -> str:
    """The EXACT array-building lines from the real wrapper source, so the
    behavior test drifts red the moment the wrapper's logic changes."""
    src = SCRIPT.read_text()
    start = src.index('RETRAIN_EXPECTED_SESSION="${RENQUANT_RETRAIN_EXPECTED_SESSION:-}"')
    end = src.index("if ! bash scripts/daily_retrain_alpha158_fund.sh", start)
    return src[start:end]


def _argv_for(expected_session: str, as_of: str) -> list[str]:
    """Execute the wrapper's own pin-array logic under bash and capture the
    argv the retrainer would receive after the fixed args."""
    harness = (
        "set -euo pipefail\n"
        + _extract_pin_block()
        + '\nprintf "%s\\n" ${RETRAIN_PIN_ARGS[@]+"${RETRAIN_PIN_ARGS[@]}"}\n'
    )
    out = _sp.run(
        ["bash", "-c", harness],
        env={
            "PATH": "/usr/bin:/bin",
            "RENQUANT_RETRAIN_EXPECTED_SESSION": expected_session,
            "RENQUANT_RETRAIN_AS_OF": as_of,
        },
        capture_output=True, text=True, check=True,
    )
    return [l for l in out.stdout.split("\n") if l]


def test_session_pins_expand_to_separate_argv_entries():
    """[codex on #564] The inline ${var:+--flag "$var"} form fused flag and
    value into ONE word. The array form must yield SEPARATE argv entries,
    including for an ISO timestamp with colons."""
    argv = _argv_for("2026-08-03", "2026-08-03T20:00:00-04:00")
    assert argv == [
        "--expected-session", "2026-08-03",
        "--as-of", "2026-08-03T20:00:00-04:00",
    ]


def test_empty_session_pins_add_nothing():
    """Scheduled runs (both envs empty) must add ZERO argv entries —
    byte-identical behavior — and the empty array must not trip set -u."""
    assert _argv_for("", "") == []


def test_only_one_pin_set_threads_just_that_pin():
    assert _argv_for("2026-08-03", "") == ["--expected-session", "2026-08-03"]
    assert _argv_for("", "2026-08-03") == ["--as-of", "2026-08-03"]


def test_retrain_call_threads_the_session_pins_only_when_set():
    """Source-shape guards kept from round 1: the pins ride the SAME
    staging-output invocation, and no tolerance loosening rides along."""
    src = SCRIPT.read_text()
    call_start = src.index("daily_retrain_alpha158_fund.sh \\")
    call_end = src.index("; then", call_start)
    call = src[call_start:call_end]
    for needle in ("--xgb-artifact-out", "--calibrator-out", "RETRAIN_PIN_ARGS"):
        assert needle in call
    assert "--no-freshness-fail-on-stale" not in src


# ── 2026-08-04: the --promote-staged operator mode (source-shape guards) ─────

def test_promote_staged_mode_reuses_the_one_mechanism():
    """The operator mode must be the SAME mechanism, not a fork: dual-
    contract arming check, the fallback CLI with --stamp as its decide
    gate, the SHARED pair-promote script, and the VERBATIM emitter line."""
    src = SCRIPT.read_text()
    idx = src.index('if [ "${1:-}" = "--promote-staged" ]; then')
    mode = src[idx:src.index("\nfi\n", idx)]
    assert "weekly_wf_promote FALLBACK-PROMOTED" in mode          # consumer contract
    assert "import renquant_backtesting.wf_gate.freshness_fallback" in mode  # provider
    assert "--prod \"$ACTIVE_ART\" --staging \"$STAGING_ART\" --stamp" in mode
    assert "scripts/fallback_pair_promote.py" in mode
    assert 'echo "=== weekly_wf_promote FALLBACK-PROMOTED (rfc210) at $(date) — $GATE_SUMMARY ==="' in mode
    # no training and no guard weakening in the mode
    assert "daily_retrain_alpha158_fund.sh" not in mode
    assert "freshness-fail-on-stale" not in mode


def test_promote_staged_routes_only_the_a4t1_refusal_to_the_orchestrator_wrapper():
    """RFC#210 A4-T1 (renquant-backtesting#128 / renquant-orchestrator#1110):
    the direct fallback CLI refuses to stamp the ONE authorized candidate
    exception (it has no ledger) and names that refusal in its verdict. The
    operator mode keeps the ONE standing mechanism first, and routes ONLY that
    named refusal to the orchestrator wrapper (identify -> committed record ->
    atomic consume -> stamp). Every other REFUSE stays a REFUSE, and the
    wrapper is looked up under the PINNED runtime, never a dev checkout."""
    src = SCRIPT.read_text()
    idx = src.index('if [ "${1:-}" = "--promote-staged" ]; then')
    mode = src[idx:src.index("\nfi\n", idx)]
    stamp_call = mode.index("--prod \"$ACTIVE_ART\" --staging \"$STAGING_ART\" --stamp")
    refusal = mode.index('"stamp_refused": "a4t1_candidate_requires_orchestrator_consumption"')
    wrapper = mode.index("$SUBREPO_ROOT/renquant-orchestrator/ops/renquant104/a4t1_promote_staged.sh")
    pair = mode.index("scripts/fallback_pair_promote.py")
    assert stamp_call < refusal < wrapper < pair          # CLI first; wrapper only on the named refusal; then the shared swap
    # $SUBREPO_ROOT already IS .subrepo_runtime/repos (renquant_subrepo_root, subrepo_env.sh):
    # the repo name follows it directly — never "$SUBREPO_ROOT/repos/..." (codex #632 HIGH).
    assert "$SUBREPO_ROOT/repos/" not in mode
    assert "renquant-orchestrator-run/ops/renquant104/a4t1" not in mode   # the wrapper is never taken from the -run dev checkout
    assert mode.count("a4t1_promote_staged.sh") == 1
    # the wrapper runs under the SAME python and logs next to the other verdicts
    assert 'PYTHON="$PYTHON" LOG_DIR="$LOG_DIR" "$A4T1_WRAPPER" "$PS_RUN_ID" "$ACTIVE_ART" "$STAGING_ART"' in mode
    # the pinned PYTHONPATH must resolve renquant_orchestrator for the wrapper
    pp = src[src.index('export PYTHONPATH="$(renquant_subrepo_pythonpath'):]
    pp = pp[:pp.index("\n")]
    assert "renquant-backtesting" in pp and "renquant-orchestrator" in pp


def test_pair_promote_is_one_shared_implementation():
    """Both the scheduled Step 4b path and the operator mode call the ONE
    extracted script; the inline heredoc is gone (no twin swap dances)."""
    src = SCRIPT.read_text()
    assert src.count("scripts/fallback_pair_promote.py") == 2
    assert "def _swap_into_active" not in src   # the CODE lives only in the script (comments may reference it)
    helper = (SCRIPT.parent / "fallback_pair_promote.py").read_text()
    assert "promotion_basis" in helper and "freshness_fallback_rfc210" in helper
    assert "_swap_into_active" in helper


def test_promote_staged_refuses_without_staged_pair():
    src = SCRIPT.read_text()
    idx = src.index('if [ "${1:-}" = "--promote-staged" ]; then')
    mode = src[idx:src.index("\nfi\n", idx)]
    assert "staged pair not found" in mode
    assert "exit 1" in mode


def test_promote_staged_rejects_traversal_run_ids():
    """[codex on #566] EXECUTION-level regression: a traversal-like run id
    must refuse BEFORE any path is constructed — exit 2, no file reads or
    writes outside the usage/refusal echo. The wrapper's mode block is
    executed under bash with stub env; every malformed form refuses."""
    import subprocess
    src = SCRIPT.read_text()
    start = src.index('if [ "${1:-}" = "--promote-staged" ]; then')
    end = src.index("\nfi\n", start) + 4
    mode_block = src[start:end]
    harness = (
        "set -uo pipefail\n"
        'ART_DIR="$TMPDIR_ART"\nLOG_DIR="$TMPDIR_ART"\n'
        'ACTIVE_ART="$ART_DIR/active.json"\nACTIVE_CAL="$ART_DIR/cal.json"\n'
        'PYTHON=/usr/bin/false\nLOG=/dev/null\n'
        "notify() { :; }\n"
        + mode_block
        + '\necho "FELL_THROUGH"\n'
    )
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for bad in ("../../evil", "20260802T170002Z/../x", "foo",
                    "20260802T170002Zx", "..%2F..", "2026-08-02T17:00:02Z"):
            out = subprocess.run(
                ["bash", "-c", harness, "bash", "--promote-staged", bad],
                env={"PATH": "/usr/bin:/bin", "TMPDIR_ART": td},
                capture_output=True, text=True)
            assert out.returncode == 2, (bad, out.returncode, out.stdout, out.stderr)
            assert "RUN_ID must match" in out.stdout, (bad, out.stdout)
        # the canonical form passes validation and proceeds to the staged-
        # pair existence check (exit 1, different message) — proving the
        # validator admits the real format.
        ok = subprocess.run(
            ["bash", "-c", harness, "bash", "--promote-staged", "20260802T170002Z"],
            env={"PATH": "/usr/bin:/bin", "TMPDIR_ART": td},
            capture_output=True, text=True)
        assert ok.returncode == 1, (ok.returncode, ok.stdout, ok.stderr)
        assert "staged pair not found" in ok.stdout


# --- reject-notify disposition (operator directive 2026-08-04) ------------------
#
# A REFUSE because the served model is FRESH is the healthy steady state of
# RFC#210 governance and must notify calm + exit 0. Anything unproven keeps the
# alarm tone + exit 1 (the tests above pin that side: the legacy stub verdict
# {"verdict": ...} and module-unavailable both stay alarms).

def _real_shape_refusal(*, refused_on="prod_stale", prod_ok=False,
                        staleness_days=2, prod_trained="2026-08-02") -> str:
    import json as _json
    return _json.dumps({
        "as_of": "2026-08-04",
        "decision": "REFUSE",
        "policy": "freshness_fallback_rfc210",
        "refused_on": refused_on,
        "checks": [
            {"check": "gate_rejected", "ok": True, "stamped_verdict": False},
            {"check": "prod_stale", "ok": prod_ok, "prod_trained": prod_trained,
             "staleness_days": staleness_days,
             "why": "served model is %dd old" % staleness_days},
        ],
    })


def test_fresh_prod_refusal_notifies_calm_and_exits_zero(tmp_path):
    """The 2026-08-04 live shape: gate reject + prod 2d fresh -> calm + rc 0."""
    root = tmp_path / "repo"
    mod = fixture.build_fixture_repo(root)
    _force_wf_gate_reject(root)
    pythonpath_dir = tmp_path / "pythonpath_shim"
    _write_freshness_fallback_shim(
        pythonpath_dir, refuse=True, refuse_payload=_real_shape_refusal())

    active_artifact = (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod"
                        / fixture.ACTIVE_ARTIFACT_NAME)
    before_artifact = active_artifact.read_text(encoding="utf-8")

    notify_log = tmp_path / "notify.log"
    lock_file = tmp_path / "weekly.lock"
    result = _run(root, notify_log, lock_file, pythonpath_dir)
    log_tail = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (root / "logs" / "weekly_wf_promote").glob("*.log"))
    notifications = notify_log.read_text(encoding="utf-8") if notify_log.exists() else ""

    assert "RFC#210 fallback verdict: REFUSE" in log_tail, log_tail[-3000:]
    # the sentinel's log-contract line is unchanged either way
    assert "WF gate REJECTED staged model — production unchanged." in log_tail
    assert "WEEKLY-REJECT (prod fresh — no action)" in notifications, notifications
    assert "trained 2026-08-02, 2d old" in notifications, notifications
    # the ALARM-tone title must NOT fire (title match is exact-with-colon)
    assert "RenQuant 104 WEEKLY-REJECT:" not in notifications, notifications
    assert result.returncode == 0, log_tail[-3000:]
    assert active_artifact.read_text(encoding="utf-8") == before_artifact
    assert not lock_file.exists(), "lock file must be released on exit"


def test_stale_prod_refusal_keeps_alarm_and_exit_one(tmp_path):
    """REFUSE on another check while prod is genuinely old -> alarm + rc 1,
    with the disposition's reason in the notification body."""
    root = tmp_path / "repo"
    mod = fixture.build_fixture_repo(root)
    _force_wf_gate_reject(root)
    pythonpath_dir = tmp_path / "pythonpath_shim"
    _write_freshness_fallback_shim(
        pythonpath_dir, refuse=True,
        refuse_payload=_real_shape_refusal(
            refused_on="candidate_stale", prod_ok=True, staleness_days=44,
            prod_trained="2026-06-21"))

    notify_log = tmp_path / "notify.log"
    lock_file = tmp_path / "weekly.lock"
    result = _run(root, notify_log, lock_file, pythonpath_dir)
    notifications = notify_log.read_text(encoding="utf-8") if notify_log.exists() else ""

    assert "RenQuant 104 WEEKLY-REJECT:" in notifications, notifications
    assert "candidate_stale" in notifications, notifications
    assert result.returncode == 1
    assert not lock_file.exists(), "lock file must be released on exit"
