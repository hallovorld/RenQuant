"""Wiring tests for ``scripts/weekly_retrain_patchtst.sh`` WEEKLY-mode cutoff (S12 B3).

The S12 panel-refresh diagnosis (renquant-orchestrator
``doc/research/2026-07-02-s12-panel-refresh-diagnosis.md`` §4-B3) found the
WEEKLY-mode ``LATEST_CUT`` pinned to the STATIC source manifest's frozen tail
(2026-03-09): after a corpus refresh the retrain advances once, then re-trains
the same cutoff forever and the served pin re-freezes. The fix derives the
cutoff from the refreshed corpus's labeled frontier via the orchestrator-owned
``renquant_orchestrator.patchtst_weekly_cutoff`` (fail-closed); this file pins
the WRAPPER wiring:

- the derived cutoff (not the static tail) lands in the effective source
  manifest handed to ``build_patchtst_wf_manifest``;
- every other WF invocation arg is UNCHANGED (regression);
- a derivation failure aborts BEFORE any training; an orchestrator pin that
  predates the module aborts with an explicit message;
- FULL-manifest mode still consumes the static manifest untouched.

The wrapper runs against a scaffolded temp repo whose ``.venv/bin/python`` is a
shim that logs argv and emulates each delegated module — no real python module
import, no training, no network, no production path is ever touched.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

DERIVED_CUTOFF = "2026-04-06"  # what the (faked) corpus-frontier derivation yields
STATIC_TAIL = "2026-03-09"  # the frozen static-manifest tail (must NOT be trained)

_SHIM = """#!/usr/bin/env bash
# Fake .venv/bin/python for the wrapper under test: log argv, emulate modules.
log="${FAKE_PY_LOG:?}"
printf 'PY %s\\n' "$*" >> "$log"
case "$*" in
  *"import renquant_orchestrator.build_patchtst_wf_manifest"*)
    exit 0;;
  *"import renquant_orchestrator.patchtst_weekly_cutoff"*)
    exit "${FAKE_CUTOFF_IMPORT_RC:-0}";;
  "-m renquant_orchestrator.patchtst_weekly_cutoff "*)
    if [ "${FAKE_CUTOFF_FAIL:-0}" = "1" ]; then
      echo "patchtst_weekly_cutoff: FAIL-CLOSED — corpus is STALE" >&2
      exit 1
    fi
    echo "${FAKE_CUTOFF:?}"
    exit 0;;
  "-m renquant_orchestrator.build_patchtst_wf_manifest "*)
    prev=""
    for a in "$@"; do
      if [ "$prev" = "--source-manifest" ]; then
        printf 'BUILD_SRC_CONTENT %s\\n' "$(cat "$a")" >> "$log"
      fi
      prev="$a"
    done
    exit 0;;
  "-c "*)
    exec "${FAKE_REAL_PYTHON:?}" "$@";;  # the EFFECTIVE_SRC json writer
esac
echo "shim: unexpected invocation: $*" >&2
exit 97
"""


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Scaffold a temp RENQUANT_REPO_ROOT: shim python + real wrapper/env script."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("weekly_retrain_patchtst.sh", "subrepo_env.sh"):
        (repo / "scripts" / name).write_text((REPO / "scripts" / name).read_text())
    shim = repo / ".venv" / "bin" / "python"
    shim.parent.mkdir(parents=True)
    shim.write_text(_SHIM)
    shim.chmod(0o755)
    manifest = repo / "backtesting/renquant_104/artifacts/sim/walkforward_manifest_v2_20260602.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(
        {"retrains": [{"cutoff_date": c} for c in ("2026-02-16", STATIC_TAIL)]}))
    return repo


def _run(fake_repo: Path, tmp_path: Path, **env_over: str) -> tuple[subprocess.CompletedProcess, str]:
    log = tmp_path / "invocations.log"
    log.touch()
    env = dict(
        os.environ,
        RENQUANT_REPO_ROOT=str(fake_repo),
        RQ_PATCHTST_LOCK_FILE=str(tmp_path / "lock"),
        RQ_PATCHTST_REFRESH_CORPUS="0",  # refresh chain is out of scope here
        RQ_PATCHTST_PROMOTE="0",  # promote chain is out of scope here
        FAKE_PY_LOG=str(log),
        FAKE_CUTOFF=DERIVED_CUTOFF,
        FAKE_REAL_PYTHON=sys.executable,
        TMPDIR=str(tmp_path),
        **env_over,
    )
    proc = subprocess.run(
        ["bash", str(fake_repo / "scripts" / "weekly_retrain_patchtst.sh")],
        capture_output=True, text=True, env=env, timeout=60,
    )
    return proc, log.read_text()


def _build_lines(log: str) -> list[str]:
    return [l for l in log.splitlines()
            if l.startswith("PY -m renquant_orchestrator.build_patchtst_wf_manifest ")]


def test_weekly_cutoff_derived_from_corpus_not_static_manifest(
    fake_repo: Path, tmp_path: Path,
) -> None:
    proc, log = _run(fake_repo, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # the derivation was invoked against the corpus, with the static manifest
    # demoted to the lower-bound sanity input
    derive = [l for l in log.splitlines()
              if l.startswith("PY -m renquant_orchestrator.patchtst_weekly_cutoff ")]
    assert len(derive) == 1
    assert f"--corpus {fake_repo}/data/transformer_v4_wl200_clean.parquet" in derive[0]
    assert ("--lower-bound-manifest "
            f"{fake_repo}/backtesting/renquant_104/artifacts/sim/"
            "walkforward_manifest_v2_20260602.json") in derive[0]
    assert "--max-staleness-days 28" in derive[0]
    # the DERIVED cutoff — not the frozen static tail — is what gets trained
    src_content = [l for l in log.splitlines() if l.startswith("BUILD_SRC_CONTENT ")]
    assert len(src_content) == 1
    payload = json.loads(src_content[0].removeprefix("BUILD_SRC_CONTENT "))
    assert payload == {"retrains": [{"cutoff_date": DERIVED_CUTOFF}]}
    assert STATIC_TAIL not in src_content[0]
    assert f"Mode: WEEKLY — train derived corpus-frontier cutoff ({DERIVED_CUTOFF})" in proc.stdout


def test_wf_invocation_args_otherwise_unchanged(fake_repo: Path, tmp_path: Path) -> None:
    """Regression: the cutoff derivation must not perturb the WF build argv."""
    proc, log = _run(fake_repo, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    build = _build_lines(log)
    assert len(build) == 1
    argv = build[0].removeprefix("PY ").split()
    flags = dict(zip(argv[2::2], argv[3::2]))
    art = f"{fake_repo}/backtesting/renquant_104/artifacts"
    assert flags["--output-dir"] == f"{art}/walkforward_patchtst"
    assert flags["--output-manifest"] == f"{art}/walkforward_patchtst_manifest.json"
    assert flags["--cadence-days"] == "0"
    assert flags["--seed"] == "44"
    assert flags["--epochs"] == "5"
    assert flags["--device"] == "cpu"
    # exactly the historical flag set — nothing new leaked into the WF argv
    assert sorted(flags) == [
        "--cadence-days", "--device", "--epochs", "--output-dir",
        "--output-manifest", "--seed", "--source-manifest",
    ]
    # the effective source is the single-cutoff temp manifest, not the static one
    assert flags["--source-manifest"] != str(
        fake_repo / "backtesting/renquant_104/artifacts/sim/walkforward_manifest_v2_20260602.json")


def test_derivation_failure_aborts_before_training(fake_repo: Path, tmp_path: Path) -> None:
    proc, log = _run(fake_repo, tmp_path, FAKE_CUTOFF_FAIL="1")
    assert proc.returncode != 0
    assert _build_lines(log) == []  # fail-closed: no training was attempted
    assert "FAIL-CLOSED" in proc.stdout + proc.stderr


def test_stale_orchestrator_pin_aborts_with_message(fake_repo: Path, tmp_path: Path) -> None:
    proc, log = _run(fake_repo, tmp_path, FAKE_CUTOFF_IMPORT_RC="1")
    assert proc.returncode != 0
    assert "predates the S12 B3" in proc.stdout + proc.stderr
    assert _build_lines(log) == []


def test_full_manifest_mode_still_uses_static_manifest(fake_repo: Path, tmp_path: Path) -> None:
    proc, log = _run(fake_repo, tmp_path, RQ_PATCHTST_FULL_MANIFEST="1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not any(
        "patchtst_weekly_cutoff" in l for l in log.splitlines())  # no derivation
    build = _build_lines(log)
    assert len(build) == 1
    assert ("--source-manifest "
            f"{fake_repo}/backtesting/renquant_104/artifacts/sim/"
            "walkforward_manifest_v2_20260602.json") in build[0]
    assert "--cadence-days 180" in build[0]
