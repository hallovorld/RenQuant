"""Tests for scripts/pin_import_integrity_sweep.py (GOAL-5 AC5, D1).

Two layers:
  * AST collection unit tests — pure, always run.
  * The #524 regression proof — builds real pinned checkouts from the local
    sibling clones and runs the sweep as a subprocess (fresh interpreter per
    combination, since aliasing mutates sys.modules). Skipped when the
    sibling clones or their SHAs are unavailable (e.g. plain CI runners);
    the CI workflow runs the live sweep against the PR's lock instead.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "pin_import_integrity_sweep.py"
SIBLINGS = Path("/Users/renhao/git/github")

# Frozen regression fixture: the pin combination that shipped the 07-16
# MetaLabelVetoTask outage (orchestrator pre-#524 + pipeline with #203).
ORCH_PRE_524 = "bfb935e4a38d3e7653f6576c5e6a461731cb4acc"
ORCH_POST_524 = "28b1d2baf2c7113afcd3af3507c8fd9047a265da"
PIPELINE_PIN = "7108f51422ef6fc325624c82bc0b3b149b13724b"
BACKTESTING_PIN = "8f6700ab35589d7ada518305ea8cd5aaa598af47"
COMMON_PIN = "6f09cb99ae47915310cb02b21dab39059a38073e"


def _import_mod():
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        import pin_import_integrity_sweep as m  # noqa: PLC0415
        return m
    finally:
        sys.path.pop(0)


class TestCollectAliasedImports:
    def _collect(self, tmp_path: Path, source: str):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "mod.py").write_text(textwrap.dedent(source))
        m = _import_mod()
        return m.collect_aliased_imports(pkg)

    def test_module_level_and_function_local_imports_collected(self, tmp_path):
        got = self._collect(tmp_path, """
            from renquant_pipeline.kernel.meta_label.task_meta_label_veto import X

            def f():
                from kernel.sizing import Y
                import renquant_pipeline.kernel.persistence
        """)
        mods = sorted(t["module"] for t in got)
        assert mods == [
            "kernel.sizing",
            "renquant_pipeline.kernel.meta_label.task_meta_label_veto",
            "renquant_pipeline.kernel.persistence",
        ]

    def test_relative_and_unrelated_imports_ignored(self, tmp_path):
        got = self._collect(tmp_path, """
            from . import sibling
            from .relative.kernel import thing
            import os
            from renquant_common.model_fingerprint import model_content_sha256
        """)
        assert got == []

    def test_importfrom_names_recorded_for_submodule_check(self, tmp_path):
        got = self._collect(tmp_path, """
            from renquant_pipeline.kernel.meta_label import task_snapshot, predictor
        """)
        assert got[0]["names"] == ["task_snapshot", "predictor"]

    def test_syntax_error_is_a_finding(self, tmp_path):
        got = self._collect(tmp_path, "def broken(:\n")
        assert got and got[0].get("syntax_error")


def _clone_at(name: str, sha: str, dest: Path) -> bool:
    src = SIBLINGS / name
    if not src.is_dir():
        return False
    try:
        subprocess.run(["git", "clone", "-q", "--shared", str(src), str(dest / name)],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(dest / name), "checkout", "-q", sha],
                       check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def _run_sweep(lock: Path, siblings: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--lock-file", str(lock),
         "--siblings", str(siblings), "--json"],
        capture_output=True, text=True, timeout=900,
    )


def _fixture_lock(tmp_path: Path, orch_sha: str) -> Path:
    lock = json.loads((REPO / "subrepos.lock.json").read_text())
    for e in lock["subrepos"]:
        if e["name"] == "renquant-orchestrator":
            e["commit"] = orch_sha
    out = tmp_path / "lock.json"
    out.write_text(json.dumps(lock))
    return out


@pytest.mark.skipif(not (SIBLINGS / "renquant-pipeline").is_dir(),
                    reason="local sibling clones unavailable")
class TestRegression524:
    @pytest.fixture(scope="class")
    def checkouts(self, tmp_path_factory):
        base = tmp_path_factory.mktemp("sweep-fixture")
        combos = {}
        for label, orch_sha in (("old", ORCH_PRE_524), ("cur", ORCH_POST_524)):
            sib = base / f"sib-{label}"
            sib.mkdir()
            ok = all([
                _clone_at("renquant-orchestrator", orch_sha, sib),
                _clone_at("renquant-pipeline", PIPELINE_PIN, sib),
                _clone_at("renquant-backtesting", BACKTESTING_PIN, sib),
                _clone_at("renquant-common", COMMON_PIN, sib),
            ])
            if not ok:
                pytest.skip("required SHAs not present in local clones")
            combos[label] = sib
        return combos

    def test_old_combo_fails_naming_the_524_import(self, checkouts, tmp_path):
        res = _run_sweep(_fixture_lock(tmp_path, ORCH_PRE_524), checkouts["old"])
        assert res.returncode == 1, res.stdout + res.stderr
        out = json.loads(res.stdout)
        failed = {f.get("module") for f in out["failures"]}
        assert "renquant_pipeline.kernel.meta_label.task_meta_label_veto" in failed

    def test_current_combo_passes(self, checkouts, tmp_path):
        res = _run_sweep(_fixture_lock(tmp_path, ORCH_POST_524), checkouts["cur"])
        assert res.returncode == 0, res.stdout + res.stderr
        out = json.loads(res.stdout)
        assert out["ok"] and out["n_targets"] > 100
