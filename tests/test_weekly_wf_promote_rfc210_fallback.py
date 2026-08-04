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
        print(json.dumps({{"verdict": "REFUSE", "reason": "test-forced-refuse"}}))
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
        refuse=refuse, gate_passed=gate_passed,
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


def _force_wf_gate_reject(root: Path) -> None:
    path = root / "scripts" / "run_wf_gate.py"
    path.write_text(f"#!{sys.executable}\nimport sys\nsys.exit(1)\n", encoding="utf-8")
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


# ── 2026-08-04: manual-run session-pin passthrough (source-shape guards) ─────

def test_retrain_call_threads_the_session_pins_only_when_set():
    """The wrapper must pass the retrainer's OWN deterministic-replay pins
    through via env, and pass NOTHING when the envs are empty (scheduled
    runs stay byte-identical). Shape-checked against the source."""
    src = SCRIPT.read_text()
    assert 'RETRAIN_EXPECTED_SESSION="${RENQUANT_RETRAIN_EXPECTED_SESSION:-}"' in src
    assert 'RETRAIN_AS_OF="${RENQUANT_RETRAIN_AS_OF:-}"' in src
    assert '${RETRAIN_EXPECTED_SESSION:+--expected-session "$RETRAIN_EXPECTED_SESSION"}' in src
    assert '${RETRAIN_AS_OF:+--as-of "$RETRAIN_AS_OF"}' in src
    # The pins must ride the SAME retrain invocation that carries the
    # staging outputs — not a second call.
    call_start = src.index("daily_retrain_alpha158_fund.sh \\")
    call_end = src.index("; then", call_start)
    call = src[call_start:call_end]
    for needle in ("--xgb-artifact-out", "--calibrator-out",
                   "RETRAIN_EXPECTED_SESSION:+", "RETRAIN_AS_OF:+"):
        assert needle in call
    # No tolerance loosening rides along: the guard's failure mode is
    # untouched (fail-on-stale never disabled by this wrapper).
    assert "--no-freshness-fail-on-stale" not in src
