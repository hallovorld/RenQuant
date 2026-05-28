"""AUDIT REGRESSION GUARD — multi-repo GBDT trainer byte-identity.

scripts/train_multirepo.py routes the model-side training (booster + CV + artifact)
through the pinned renquant-model engine (legacy_panel_trainer), reusing the
umbrella's data-side loaders. Its artifact MUST be byte-identical to
scripts/train_production_model.py for the same args, excluding the two fields the
legacy script randomizes by design (train_run_id=uuid4, trained_date=utcnow).

Verified manually 2026-05-28 on cutoff 2017-07-01: booster_raw_json byte-identical,
config_fingerprint identical, oos_mean_ic/oos_per_fold_ic identical, zero other
diffs. This test re-proves it on a small slice when the production data is present;
it is skipped in environments without the dataset (e.g. CI).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "data" / "alpha158_291_fundamental_dataset.parquet"
STATS = REPO / "data" / "alpha158_qlib_dataset.stats.json"

pytestmark = pytest.mark.skipif(
    not (DATASET.exists() and STATS.exists()),
    reason="production panel dataset not present (skipped outside the workstation)",
)

RANDOMIZED = {"train_run_id", "trained_date"}


def _run(script: str, out: Path) -> None:
    cmd = [sys.executable, str(REPO / "scripts" / script),
           "--train-cutoff", "2017-01-01", "--side-label", "paritytest",
           "--skip-cv", "--output-path", str(out)]
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"


def test_multirepo_artifact_byte_identical_to_legacy(tmp_path: Path) -> None:
    legacy = tmp_path / "walkforward_legacy.json"   # 'walkforward' satisfies §5.13.13
    multirepo = tmp_path / "walkforward_multirepo.json"
    _run("train_production_model.py", legacy)
    _run("train_multirepo.py", multirepo)

    a = json.loads(legacy.read_text())
    b = json.loads(multirepo.read_text())
    assert a.get("booster_raw_json") == b.get("booster_raw_json"), "booster diverged"
    assert a.get("config_fingerprint") == b.get("config_fingerprint"), "fingerprint diverged"
    diffs = [k for k in (set(a) | set(b)) if k not in RANDOMIZED and a.get(k) != b.get(k)]
    assert not diffs, f"non-identical fields (excluding randomized): {diffs}"
