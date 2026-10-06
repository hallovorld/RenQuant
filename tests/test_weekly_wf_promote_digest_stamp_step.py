"""Step 3.5b of weekly_wf_promote.sh: the WF manifest's content digests are
stamped AT RUNTIME, after the fingerprint stamp rewrites the corpus bytes and
before the gate resolves them — so no regenerated corpus bytes need to be
committed (the RenQuant#639 rejection) and the 2026-09-13 live-manifest
containment is reproduced by code instead of by hand.

Two layers:
  * source contract on the script (ordering + fail-closed shape), in the
    style of test_wf_gate_sim_ran.py::test_promote_script_consults_the_gate…;
  * an end-to-end run of the REAL stamp script in the exact sequence the
    step performs (stamp, then --check), on a corpus whose bytes are
    rewritten before stamping, including the loud refusal when the bytes
    change under an existing stamp.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROMOTE = REPO / "scripts" / "weekly_wf_promote.sh"
STAMP = REPO / "scripts" / "stamp_wf_manifest_digests.py"


def test_digest_stamp_runs_after_fingerprints_and_before_the_gate():
    text = PROMOTE.read_text()
    fingerprints = text.index("scripts/stamp_walkforward_fingerprints.py")
    gate = text.index("--- Step 4: Walk-forward gate")
    m = re.search(r'scripts/stamp_wf_manifest_digests\.py --manifest "\$WF_MANIFEST_ABS"\s*\\\n'
                  r'\s*\|\| ! "\$PYTHON" scripts/stamp_wf_manifest_digests\.py --manifest "\$WF_MANIFEST_ABS" --check',
                  text)
    assert m is not None, "Step 3.5b must stamp AND --check the manifest digests"
    assert fingerprints < m.start() < gate, (
        "digests must be taken after the fingerprint stamp rewrites the corpus "
        "bytes and before the gate resolves them")
    block = text[m.start():gate]
    assert "exit 1" in block and "WEEKLY-FAIL" in block, "a failed digest stamp must alarm and exit 1"
    # The digest script takes the manifest path literally (the fingerprint
    # script resolves it against the strategy dir) — the step must hand it
    # the strategy-dir path, not the bare relative one.
    assert 'WF_MANIFEST_ABS="$REPO_DIR/backtesting/renquant_104/$WF_MANIFEST"' in text


def _corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    sim = tmp_path / "artifacts" / "sim"
    cut = tmp_path / "artifacts" / "wf" / "2025-01-06"
    cut.mkdir(parents=True)
    sim.mkdir(parents=True)
    art = cut / "panel-ltr.json"
    cal = sim / "walkforward_calibrators" / "2025-01-06" / "panel-rank-calibration.json"
    cal.parent.mkdir(parents=True)
    # Pretty-printed, as the committed corpus is.
    art.write_text(json.dumps({"version": 3, "kind": "panel_ltr_xgboost"}, indent=2) + "\n")
    cal.write_text(json.dumps({"version": 1, "kind": "global_panel_calibration"}, indent=2) + "\n")
    manifest = sim / "walkforward_manifest.json"
    manifest.write_text(json.dumps({
        "retrains": [{
            "date": "2025-01-06",
            "artifact_uri": "artifacts/wf/2025-01-06/panel-ltr.json",
            "calibrator_uri": "artifacts/sim/walkforward_calibrators/2025-01-06/panel-rank-calibration.json",
        }],
    }, indent=2) + "\n")
    return manifest, art, cal


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(STAMP), *args], capture_output=True, text=True,
                          cwd=REPO)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_step_sequence_stamps_the_rewritten_bytes_and_check_passes(tmp_path):
    manifest, art, cal = _corpus(tmp_path)
    # What Step 3.5 does before 3.5b: the fingerprint stamp rewrites the
    # artifact compactly. The digests must describe THESE bytes.
    art.write_text(json.dumps({"version": 3, "kind": "panel_ltr_xgboost", "config_fingerprint": "cfg"}))
    r = _run("--manifest", str(manifest))
    assert r.returncode == 0, r.stdout + r.stderr
    r = _run("--manifest", str(manifest), "--check")
    assert r.returncode == 0, r.stdout + r.stderr
    entry = json.loads(manifest.read_text())["retrains"][0]
    assert entry["artifact_sha256"] == _sha(art)
    assert entry["calibrator_sha256"] == _sha(cal)
    # Repeat pass (next week): byte-for-byte no-op.
    before = manifest.read_bytes()
    assert _run("--manifest", str(manifest)).returncode == 0
    assert manifest.read_bytes() == before


def test_bytes_changed_under_an_existing_stamp_is_a_loud_refusal(tmp_path):
    manifest, art, cal = _corpus(tmp_path)
    assert _run("--manifest", str(manifest)).returncode == 0
    cal.write_text(json.dumps({"version": 1, "kind": "global_panel_calibration", "refit": True}))
    r = _run("--manifest", str(manifest))
    assert r.returncode != 0
    assert "refusing to overwrite a conflicting stamp" in (r.stdout + r.stderr)
    r = _run("--manifest", str(manifest), "--check")
    assert r.returncode != 0
    assert "calibrator_sha256" in (r.stdout + r.stderr)
