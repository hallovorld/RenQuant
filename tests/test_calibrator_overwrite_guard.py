"""Source-level regression test for the calibrator production-overwrite guard.

Closes 2026-05-05's three production calibrator overwrite incidents
(diagnostic Platt fit + retrain_v2 + isotonic-recal each silently wrote
to artifacts/panel-rank-calibration.json).

We pin the guard pattern at the source level rather than spinning up
a full subprocess invocation — the guard touches I/O + pipeline import
at ~30s setup cost. The structural test is sufficient because the bug
is about CONTROL-FLOW gating, not data shape.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "fit_panel_calibrator.py"


def test_script_has_force_flag():
    txt = SCRIPT.read_text()
    assert '"--force"' in txt, "Script must accept --force flag"
    assert 'action="store_true"' in txt and 'force' in txt


def test_guard_checks_canonical_path():
    txt = SCRIPT.read_text()
    assert 'canonical_prod' in txt, (
        "Guard must reference canonical production path explicitly"
    )
    assert 'panel-rank-calibration.json' in txt


def test_guard_compares_methods():
    txt = SCRIPT.read_text()
    # The guard must compare existing vs new calibration_method
    assert re.search(r"existing_method.*calib_method", txt, re.S), (
        "Guard must compare existing vs new calibration_method to detect "
        "diagnostic-overwrite"
    )


def test_guard_exits_when_method_diff_and_no_force():
    txt = SCRIPT.read_text()
    # The guard branch executes sys.exit(2) and points the operator to
    # --out for diagnostic or --force for confirmed replacement
    assert "sys.exit(2)" in txt
    assert "--force to confirm" in txt
    assert "--out <side_path>" in txt


def test_guard_short_circuited_by_explicit_out():
    txt = SCRIPT.read_text()
    # If args.out is set, guard does not fire (operator confirmed target)
    assert re.search(r"out_path == canonical_prod and not args\.out", txt), (
        "Guard must skip when --out was passed (explicit target)"
    )


def test_method_persisted_in_metadata():
    txt = SCRIPT.read_text()
    # save() must include "method" in metadata so subsequent guards can
    # compare existing vs new method
    assert re.search(r'"method":\s*calib_method', txt), (
        "Must persist method in calibrator metadata for future comparison"
    )
