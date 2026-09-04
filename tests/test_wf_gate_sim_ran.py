"""A WF-gate reject is only a verdict if the simulation actually ran.

2026-09-01..03: three daily candidates were stamped
`wf_reason = "3/3 sim cuts failed execution"`, `cuts[*].returncode = 1`
(ManifestUriResolutionError inside the sim), and `weekly_wf_promote.sh`
reported each as "Reject disposition: prod FRESH … governance nominal, calm
notify, exit 0". `scripts/wf_gate_sim_ran.py` is the gate that stops that:
exit 0 iff every cut carries the int returncode 0; everything else exits 1.
The promote script must consult it BEFORE the RFC#210 fallback so a crashed
candidate is neither reported calm nor eligible for fallback promotion.

Hermetic: stdlib only, tmp_path only, drives the helper through both its
function and its CLI; the promote script is read as text for the ordering
guard.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
HELPER = _REPO_ROOT / "scripts" / "wf_gate_sim_ran.py"
PROMOTE = _REPO_ROOT / "scripts" / "weekly_wf_promote.sh"

spec = importlib.util.spec_from_file_location("wf_gate_sim_ran", HELPER)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _artifact(cuts, wf_reason="FAIL: zero trades across all WF cuts"):
    return {"kind": "panel_ltr_xgboost", "metadata": {"wf_gate_metadata": {
        "passed": False, "wf_reason": wf_reason, "cuts": cuts}}}


def _write(tmp_path: Path, payload) -> str:
    p = tmp_path / "staging.json"
    p.write_text(json.dumps(payload) if not isinstance(payload, str) else payload)
    return str(p)


def _cut(rc, start="2024-01-02", end="2024-12-31"):
    return {"start": start, "end": end, "returncode": rc, "sharpe": None}


def test_executed_reject_is_a_verdict(tmp_path):
    ok, reason = mod.sim_ran(_write(tmp_path, _artifact([_cut(0), _cut(0), _cut(0)])))
    assert ok and "all 3 cuts executed" in reason


def test_the_2026_09_01_shape_did_not_run(tmp_path):
    """The real stamp of the 09-01..03 candidates: rc 1 on every cut."""
    ok, reason = mod.sim_ran(_write(tmp_path, _artifact(
        [_cut(1), _cut(1, "2024-07-01", "2025-06-30"), _cut(1, "2025-01-02", "2025-12-31")],
        wf_reason="3/3 sim cuts failed execution")))
    assert not ok
    assert reason.startswith("3/3 cuts did not execute")
    assert "returncode=1" in reason and "gate said: 3/3 sim cuts failed execution" in reason


def test_one_crashed_cut_is_enough(tmp_path):
    ok, reason = mod.sim_ran(_write(tmp_path, _artifact([_cut(0), _cut(2), _cut(0)])))
    assert not ok and reason.startswith("1/3 cuts did not execute")


@pytest.mark.parametrize("rc", [None, "0", 0.0, False, True, -1])
def test_only_the_int_zero_proves_execution(tmp_path, rc):
    ok, _ = mod.sim_ran(_write(tmp_path, _artifact([_cut(rc)])))
    assert not ok


def test_missing_evidence_fails_closed(tmp_path):
    assert not mod.sim_ran(str(tmp_path / "absent.json"))[0]
    assert not mod.sim_ran(_write(tmp_path, "{not json"))[0]
    assert not mod.sim_ran(_write(tmp_path, [1, 2]))[0]
    assert not mod.sim_ran(_write(tmp_path, {"metadata": {}}))[0]
    assert not mod.sim_ran(_write(tmp_path, {"metadata": {"wf_gate_metadata": {"cuts": []}}}))[0]
    assert not mod.sim_ran(_write(tmp_path, _artifact(["not-a-dict"])))[0]


def test_cli_exit_codes_and_one_line(tmp_path):
    ran = _write(tmp_path, _artifact([_cut(0), _cut(0), _cut(0)]))
    crashed = str(tmp_path / "crashed.json")
    Path(crashed).write_text(json.dumps(_artifact([_cut(1)], "1/3 sim cuts failed execution")))
    for args, code, prefix in (
        ([ran], 0, "WF-SIM-RAN|"),
        ([crashed], 1, "WF-SIM-DID-NOT-RUN|"),
        ([str(tmp_path / "absent.json")], 1, "WF-SIM-DID-NOT-RUN|"),
        ([], 1, "WF-SIM-UNPROVEN|"),
    ):
        r = subprocess.run([sys.executable, str(HELPER), *args],
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == code, (args, r.stdout, r.stderr)
        lines = r.stdout.splitlines()
        assert len(lines) == 1 and lines[0].startswith(prefix), (args, r.stdout)


def test_promote_script_consults_the_gate_before_the_fallback():
    """Ordering guard on the source: the sim-ran check must sit after the
    gate's REJECTED line and before the RFC#210 fallback is consulted, and
    it must exit 1 — a crashed candidate is never fallback-eligible."""
    text = PROMOTE.read_text()
    rejected = text.index('echo "WF gate REJECTED staged model — consulting the RFC#210 freshness fallback')
    fallback = text.index("FALLBACK_JSON=")
    m = re.search(r'scripts/wf_gate_sim_ran\.py "\$STAGING_ART"', text)
    assert m is not None, "weekly_wf_promote.sh does not call wf_gate_sim_ran.py on the staging artifact"
    assert rejected < m.start() < fallback, "wf_gate_sim_ran.py must run after REJECTED and before the fallback"
    block = text[m.start():fallback]
    assert "exit 1" in block and "notify" in block, "a crashed simulation must alarm and exit 1"
    assert "WEEKLY-FAIL" in block
