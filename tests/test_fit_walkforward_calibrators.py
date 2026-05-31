from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parent.parent
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from fit_walkforward_calibrators import _date_window, _fit_one, main  # noqa: E402


def test_date_window_uses_effective_cutoff_before_forward_label() -> None:
    start, end = _date_window(
        pd.Timestamp("2023-10-02"),
        years=0.0,
        lookahead_days=60,
    )
    assert start is None
    assert end == "2023-07-10"


def test_date_window_applies_training_window_before_effective_cutoff() -> None:
    start, end = _date_window(
        pd.Timestamp("2024-01-02"),
        years=1.0,
        lookahead_days=60,
    )
    assert end == "2023-10-10"
    assert start < end


# --- Bug G regression guards (2026-05-31) ---------------------------------
#
# Per-cut calibrator output was compressed to [0.07, 0.13] because the fit
# window inherited training_window_years=3.0. The fix decouples the
# calibrator window from the model-training window. These tests pin:
#   1. `calibrator_window_years=None`  → legacy: uses training_window_years
#   2. `calibrator_window_years=0.0`   → unbounded: no --data-start passed
#   3. `calibrator_window_years=5.0`   → override: 5-year window regardless
#      of manifest's training_window_years
#   4. main() stamps the chosen window into calibrator_policy + per-row.


def _row_fixture(tmp_path: Path) -> dict[str, object]:
    scorer = tmp_path / "scorer.json"
    scorer.write_text("{}")
    return {
        "cutoff_date": "2024-01-02",
        "artifact_uri": str(scorer),
        "lookahead_days": 60,
    }


@pytest.fixture
def fake_subprocess(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Capture the subprocess.run cmd argv that _fit_one would have invoked.

    Real fitters write the artifact JSON; here we synthesize a minimal
    artifact so the post-fit ``_stamp_window_into_artifact`` step can
    open + amend it.  Tests that need to control the artifact's pre-stamp
    contents can override via ``fake.side_effect = ...``.
    """
    def _default_side_effect(cmd, *a, **kw):
        # Parse the --out arg and synthesize a minimal artifact at that path.
        try:
            out_idx = cmd.index("--out") + 1
            out_path = Path(cmd[out_idx])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"method": "platt"}))
        except (ValueError, IndexError):
            pass
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    fake = MagicMock(side_effect=_default_side_effect)
    monkeypatch.setattr("fit_walkforward_calibrators.subprocess.run", fake)
    return fake


def _captured_argv(fake: MagicMock) -> list[str]:
    fake.assert_called_once()
    cmd = fake.call_args.args[0]
    assert isinstance(cmd, list)
    return cmd


def test_fit_one_legacy_path_uses_training_window_years_for_calibrator(
    tmp_path: Path, fake_subprocess: MagicMock,
) -> None:
    """`calibrator_window_years=None` preserves the pre-Bug-G behavior."""
    out = _fit_one(
        _row_fixture(tmp_path),
        calibrator_root=tmp_path / "out",
        training_window_years=3.0,
        calibrator_window_years=None,
        method="platt",
        panel=None,
        raw_label_panel=None,
        overwrite=True,
    )
    argv = _captured_argv(fake_subprocess)
    assert "--data-start" in argv, "legacy path must pass a bounded --data-start"
    assert out["calibrator_window_years"] == 3.0


def test_fit_one_zero_window_omits_data_start_for_full_history(
    tmp_path: Path, fake_subprocess: MagicMock,
) -> None:
    """Bug G fix: calibrator_window_years=0 → full history, no --data-start."""
    out = _fit_one(
        _row_fixture(tmp_path),
        calibrator_root=tmp_path / "out",
        training_window_years=3.0,
        calibrator_window_years=0.0,
        method="platt",
        panel=None,
        raw_label_panel=None,
        overwrite=True,
    )
    argv = _captured_argv(fake_subprocess)
    assert "--data-start" not in argv, (
        "Bug G fix: calibrator_window_years=0 must drop --data-start so the "
        "fitter sees the full history up to data_end"
    )
    assert "--data-end" in argv
    assert out["calibrator_data_start"] is None
    assert out["calibrator_window_years"] == 0.0


def test_fit_one_override_uses_calibrator_window_not_training_window(
    tmp_path: Path, fake_subprocess: MagicMock,
) -> None:
    """5-year calibrator window overrides 3-year training window."""
    out = _fit_one(
        _row_fixture(tmp_path),
        calibrator_root=tmp_path / "out",
        training_window_years=3.0,
        calibrator_window_years=5.0,
        method="platt",
        panel=None,
        raw_label_panel=None,
        overwrite=True,
    )
    argv = _captured_argv(fake_subprocess)
    ds = argv[argv.index("--data-start") + 1]
    # cutoff=2024-01-02, lookahead=60 → effective_cutoff=2023-10-10
    # 5y window → start ≈ 2018-10-12 (legacy 3y would give ≈ 2020-10-11)
    assert ds < "2020-01-01", (
        f"5y calibrator window should produce data_start before 2020, got {ds}"
    )
    assert out["calibrator_window_years"] == 5.0


def test_main_stamps_calibrator_window_in_policy_and_per_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main() persists the chosen window into both the manifest policy AND
    each retrains[] row, so downstream consumers can audit the fit window
    without re-parsing CLI flags."""
    # Stub _fit_one so we don't need real fitter scripts on disk.
    def fake_fit_one(row, **kw):
        return {**row,
                "calibrator_uri": str(tmp_path / "cal.json"),
                "calibrator_data_start": "2018-01-01",
                "calibrator_data_end": "2023-10-10",
                "calibrator_method": kw["method"],
                "calibrator_window_years": kw["calibrator_window_years"]
                                            if kw["calibrator_window_years"] is not None
                                            else kw["training_window_years"]}
    monkeypatch.setattr("fit_walkforward_calibrators._fit_one", fake_fit_one)

    in_manifest = tmp_path / "in.json"
    out_manifest = tmp_path / "out.json"
    in_manifest.write_text(json.dumps({
        "training_window_years": 3.0,
        "retrains": [{
            "cutoff_date": "2024-01-02",
            "artifact_uri": str(tmp_path / "scorer.json"),
            "lookahead_days": 60,
        }],
    }))
    (tmp_path / "scorer.json").write_text("{}")
    monkeypatch.setattr(
        sys, "argv",
        ["fit_walkforward_calibrators.py",
         "--manifest", str(in_manifest),
         "--out-manifest", str(out_manifest),
         "--calibrator-root", str(tmp_path / "out"),
         "--calibrator-window-years", "0.0"],
    )
    assert main() == 0
    out = json.loads(out_manifest.read_text())
    assert out["calibrator_policy"]["calibrator_window_years"] == 0.0
    assert out["calibrator_policy"]["training_window_years"] == 3.0
    assert out["calibrator_policy"]["fit_window"] == "calibrator_window_through_effective_cutoff"
    assert out["retrains"][0]["calibrator_window_years"] == 0.0


def test_main_legacy_default_falls_back_to_training_window_in_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting `--calibrator-window-years` keeps the legacy policy label."""
    def fake_fit_one(row, **kw):
        return {**row,
                "calibrator_uri": str(tmp_path / "cal.json"),
                "calibrator_data_start": "2021-01-10",
                "calibrator_data_end": "2023-10-10",
                "calibrator_method": kw["method"],
                "calibrator_window_years": kw["training_window_years"]}
    monkeypatch.setattr("fit_walkforward_calibrators._fit_one", fake_fit_one)

    in_manifest = tmp_path / "in.json"
    out_manifest = tmp_path / "out.json"
    in_manifest.write_text(json.dumps({
        "training_window_years": 3.0,
        "retrains": [{
            "cutoff_date": "2024-01-02",
            "artifact_uri": str(tmp_path / "scorer.json"),
            "lookahead_days": 60,
        }],
    }))
    (tmp_path / "scorer.json").write_text("{}")
    monkeypatch.setattr(
        sys, "argv",
        ["fit_walkforward_calibrators.py",
         "--manifest", str(in_manifest),
         "--out-manifest", str(out_manifest),
         "--calibrator-root", str(tmp_path / "out")],
    )
    assert main() == 0
    out = json.loads(out_manifest.read_text())
    assert out["calibrator_policy"]["calibrator_window_years"] == 3.0
    assert out["calibrator_policy"]["fit_window"] == "training_window_through_effective_cutoff"


# --- PR #13 review fixes -------------------------------------------------
#
# Two follow-up review findings on the Bug G surgical knob:
#
#   1. HIGH — reuse path could stamp the manifest with a window the cached
#      artifact wasn't fit at, silently making the manifest lie about the
#      calibration data. Fix: gate reuse on the artifact's stamped window.
#
#   2. MEDIUM — --continue-on-failure could exit rc=0 with zero successful
#      fits, and failures were never persisted into the manifest. Fix:
#      persist failures + return rc=2 on zero coverage or < --min-coverage.

# ---- Finding 1: reuse-gate ----------------------------------------------


def _cal_dir(tmp_path: Path) -> Path:
    """Per-cutoff directory under the calibrator root that _calibrator_path
    expects.  cutoff_date='2024-01-02' → 2024-01-02/panel-rank-calibration.json"""
    out = tmp_path / "out" / "2024-01-02"
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_reuse_refuses_when_existing_artifact_lacks_window_stamp(
    tmp_path: Path, fake_subprocess: MagicMock,
) -> None:
    """A pre-Bug-G artifact (no calibrator_window_years field) MUST NOT be
    silently reused under a different requested window — the manifest
    would lie about the fit data."""
    cal_path = _cal_dir(tmp_path) / "panel-rank-calibration.json"
    cal_path.write_text(json.dumps({"method": "platt"}))  # legacy: no window stamp

    with pytest.raises(RuntimeError, match="lacks calibrator_window_years metadata"):
        _fit_one(
            _row_fixture(tmp_path),
            calibrator_root=tmp_path / "out",
            training_window_years=3.0,
            calibrator_window_years=0.0,
            method="platt",
            panel=None,
            raw_label_panel=None,
            overwrite=False,
        )
    # Subprocess must NOT have been invoked — reuse gate fires before fit.
    fake_subprocess.assert_not_called()


def test_reuse_refuses_when_existing_window_differs_from_requested(
    tmp_path: Path, fake_subprocess: MagicMock,
) -> None:
    """Cached artifact at window=3.0 cannot be reused for a request at 0.0."""
    cal_path = _cal_dir(tmp_path) / "panel-rank-calibration.json"
    cal_path.write_text(json.dumps({
        "method": "platt",
        "calibrator_window_years": 3.0,  # legacy 3-year window
    }))

    with pytest.raises(RuntimeError, match=r"existing window 3\.0 != requested 0\.0"):
        _fit_one(
            _row_fixture(tmp_path),
            calibrator_root=tmp_path / "out",
            training_window_years=3.0,
            calibrator_window_years=0.0,  # operator wants the wider window
            method="platt",
            panel=None,
            raw_label_panel=None,
            overwrite=False,
        )
    fake_subprocess.assert_not_called()


def test_reuse_accepts_matching_window(
    tmp_path: Path, fake_subprocess: MagicMock,
) -> None:
    """Cached artifact at window=0.0 IS reusable when request is 0.0."""
    cal_path = _cal_dir(tmp_path) / "panel-rank-calibration.json"
    cal_path.write_text(json.dumps({
        "method": "platt",
        "calibrator_window_years": 0.0,
    }))

    out = _fit_one(
        _row_fixture(tmp_path),
        calibrator_root=tmp_path / "out",
        training_window_years=3.0,
        calibrator_window_years=0.0,
        method="platt",
        panel=None,
        raw_label_panel=None,
        overwrite=False,
    )
    assert out["calibrator_window_years"] == 0.0
    assert out["calibrator_uri"] == str(cal_path)
    fake_subprocess.assert_not_called()


def test_overwrite_short_circuits_reuse_gate(
    tmp_path: Path, fake_subprocess: MagicMock,
) -> None:
    """--overwrite ignores window mismatch — operator opt-in to refit."""
    cal_path = _cal_dir(tmp_path) / "panel-rank-calibration.json"
    cal_path.write_text(json.dumps({"calibrator_window_years": 3.0}))

    # subprocess.run must return success so _fit_one continues to stamp.
    fake_subprocess.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr="",
    )
    # The subprocess "writes" the artifact — simulate that by re-writing
    # the file with no window stamp (the fitters don't stamp; orchestrator does).
    def _side_effect(*a, **kw):
        cal_path.write_text(json.dumps({"method": "platt"}))
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    fake_subprocess.side_effect = _side_effect

    out = _fit_one(
        _row_fixture(tmp_path),
        calibrator_root=tmp_path / "out",
        training_window_years=3.0,
        calibrator_window_years=0.0,
        method="platt",
        panel=None,
        raw_label_panel=None,
        overwrite=True,
    )
    assert out["calibrator_window_years"] == 0.0
    # After refit, the artifact carries the new window stamp.
    stamped = json.loads(cal_path.read_text())
    assert stamped["calibrator_window_years"] == 0.0
    fake_subprocess.assert_called_once()


def test_fresh_fit_stamps_window_into_artifact(
    tmp_path: Path, fake_subprocess: MagicMock,
) -> None:
    """The orchestrator must inject calibrator_window_years into the
    artifact JSON that the fitter subprocess produces, so future reuse
    decisions can audit the fit policy from the artifact alone."""
    cal_path = _cal_dir(tmp_path) / "panel-rank-calibration.json"
    # Fitter writes an artifact without the window field.
    def _side_effect(*a, **kw):
        cal_path.write_text(json.dumps({"method": "platt", "a_unique": 5}))
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    fake_subprocess.side_effect = _side_effect

    _fit_one(
        _row_fixture(tmp_path),
        calibrator_root=tmp_path / "out",
        training_window_years=3.0,
        calibrator_window_years=5.0,
        method="platt",
        panel=None,
        raw_label_panel=None,
        overwrite=True,
    )
    stamped = json.loads(cal_path.read_text())
    assert stamped["calibrator_window_years"] == 5.0
    assert stamped["a_unique"] == 5  # original fields preserved


# ---- Finding 2: zero-coverage rc + failures persistence ------------------


def _two_row_manifest(tmp_path: Path) -> Path:
    in_manifest = tmp_path / "in.json"
    scorer = tmp_path / "scorer.json"
    scorer.write_text("{}")
    in_manifest.write_text(json.dumps({
        "training_window_years": 3.0,
        "retrains": [
            {"cutoff_date": "2024-01-02", "artifact_uri": str(scorer),
             "lookahead_days": 60},
            {"cutoff_date": "2024-04-02", "artifact_uri": str(scorer),
             "lookahead_days": 60},
        ],
    }))
    return in_manifest


def test_main_returns_nonzero_when_continue_on_failure_yields_zero_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero successful fits with --continue-on-failure must exit rc=2.
    Pre-fix: returned rc=0 (silent empty manifest)."""
    def fake_fit_one(row, **kw):
        raise RuntimeError(f"synthetic failure for {row['cutoff_date']}")
    monkeypatch.setattr("fit_walkforward_calibrators._fit_one", fake_fit_one)

    out_manifest = tmp_path / "out.json"
    monkeypatch.setattr(
        sys, "argv",
        ["fit_walkforward_calibrators.py",
         "--manifest", str(_two_row_manifest(tmp_path)),
         "--out-manifest", str(out_manifest),
         "--calibrator-root", str(tmp_path / "out"),
         "--continue-on-failure"],
    )
    rc = main()
    assert rc == 2, "zero-fit + --continue-on-failure must signal failure"
    # Manifest persisted with full failure record.
    out = json.loads(out_manifest.read_text())
    assert len(out["calibrator_failures"]) == 2
    cutoffs = sorted(f["cutoff_date"] for f in out["calibrator_failures"])
    assert cutoffs == ["2024-01-02", "2024-04-02"]


def test_main_persists_partial_failures_with_per_cutoff_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial coverage: 1 fit + 1 failure → failures list keeps the
    failing cutoff's cutoff_date + artifact_uri + error text."""
    def fake_fit_one(row, **kw):
        if row["cutoff_date"] == "2024-04-02":
            raise RuntimeError("synthetic failure")
        return {**row,
                "calibrator_uri": str(tmp_path / "cal.json"),
                "calibrator_data_start": None,
                "calibrator_data_end": "2023-10-10",
                "calibrator_method": kw["method"],
                "calibrator_window_years": 0.0}
    monkeypatch.setattr("fit_walkforward_calibrators._fit_one", fake_fit_one)

    out_manifest = tmp_path / "out.json"
    monkeypatch.setattr(
        sys, "argv",
        ["fit_walkforward_calibrators.py",
         "--manifest", str(_two_row_manifest(tmp_path)),
         "--out-manifest", str(out_manifest),
         "--calibrator-root", str(tmp_path / "out"),
         "--calibrator-window-years", "0.0",
         "--continue-on-failure"],
    )
    rc = main()
    assert rc == 0, "1-of-2 fit is positive coverage; default --min-coverage=0 passes"
    out = json.loads(out_manifest.read_text())
    failures = out["calibrator_failures"]
    assert len(failures) == 1
    assert failures[0]["cutoff_date"] == "2024-04-02"
    assert "synthetic failure" in failures[0]["error"]


def test_min_coverage_gate_returns_nonzero_when_below_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--min-coverage 1.0 requires every cutoff to fit; 1/2 → rc=2."""
    def fake_fit_one(row, **kw):
        if row["cutoff_date"] == "2024-04-02":
            raise RuntimeError("synthetic failure")
        return {**row,
                "calibrator_uri": str(tmp_path / "cal.json"),
                "calibrator_method": kw["method"],
                "calibrator_window_years": 0.0}
    monkeypatch.setattr("fit_walkforward_calibrators._fit_one", fake_fit_one)

    out_manifest = tmp_path / "out.json"
    monkeypatch.setattr(
        sys, "argv",
        ["fit_walkforward_calibrators.py",
         "--manifest", str(_two_row_manifest(tmp_path)),
         "--out-manifest", str(out_manifest),
         "--calibrator-root", str(tmp_path / "out"),
         "--continue-on-failure",
         "--min-coverage", "1.0"],
    )
    rc = main()
    assert rc == 2
    out = json.loads(out_manifest.read_text())
    assert out["calibrator_policy"]["min_coverage"] == 1.0
    assert len(out["calibrator_failures"]) == 1


def test_min_coverage_validates_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    """--min-coverage outside [0, 1] is rejected by argparse."""
    out_manifest = tmp_path / "out.json"
    monkeypatch.setattr(
        sys, "argv",
        ["fit_walkforward_calibrators.py",
         "--manifest", str(_two_row_manifest(tmp_path)),
         "--out-manifest", str(out_manifest),
         "--calibrator-root", str(tmp_path / "out"),
         "--min-coverage", "1.5"],
    )
    with pytest.raises(SystemExit):
        main()
    err = capsys.readouterr().err
    assert "--min-coverage" in err
    assert "1.5" in err


# --- PR #13 follow-up review (commit 3e56a41) -----------------------------
#
# Two additional findings:
#
#   F1 MEDIUM — --limit denominator off-by. After --limit slicing, the
#      coverage denominator is the SCHEDULED rows, not the full manifest.
#      --limit 1 --min-coverage 1.0 on a 2-row manifest fits 1/1 = 100%
#      and must rc=0, not rc=2 from the 1/2 = 50% computation.
#
#   F2 MEDIUM — method is the same contamination class as window. A Platt
#      artifact reused under --method isotonic would have the manifest
#      claim isotonic while the file is Platt.


# ---- F1: --limit denominator ---------------------------------------------


def test_limit_denominator_uses_scheduled_rows_not_full_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--limit 1 on a 2-row manifest with --min-coverage 1.0 fits 1/1 = 100%.

    Pre-fix: denominator was len(payload["retrains"])=2 → 50% → rc=2 even
    though every SCHEDULED cutoff fit successfully.
    """
    def fake_fit_one(row, **kw):
        return {**row,
                "calibrator_uri": str(tmp_path / "cal.json"),
                "calibrator_method": kw["method"],
                "calibrator_window_years": 0.0}
    monkeypatch.setattr("fit_walkforward_calibrators._fit_one", fake_fit_one)

    out_manifest = tmp_path / "out.json"
    monkeypatch.setattr(
        sys, "argv",
        ["fit_walkforward_calibrators.py",
         "--manifest", str(_two_row_manifest(tmp_path)),
         "--out-manifest", str(out_manifest),
         "--calibrator-root", str(tmp_path / "out"),
         "--calibrator-window-years", "0.0",
         "--limit", "1",
         "--min-coverage", "1.0"],
    )
    rc = main()
    assert rc == 0, "1-of-1 scheduled fit at --min-coverage 1.0 must pass"

    # Output manifest preserves all 2 rows (the 1 un-scheduled row passes
    # through untouched), but the coverage gate measured only the scheduled set.
    out = json.loads(out_manifest.read_text())
    assert len(out["retrains"]) == 2
    fit_count = sum(1 for r in out["retrains"] if "calibrator_uri" in r)
    assert fit_count == 1


def test_limit_denominator_still_catches_partial_in_scheduled_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--limit 2 with 1 fit + 1 fail still fails --min-coverage 1.0."""
    def fake_fit_one(row, **kw):
        if row["cutoff_date"] == "2024-04-02":
            raise RuntimeError("synthetic failure")
        return {**row,
                "calibrator_uri": str(tmp_path / "cal.json"),
                "calibrator_method": kw["method"],
                "calibrator_window_years": 0.0}
    monkeypatch.setattr("fit_walkforward_calibrators._fit_one", fake_fit_one)

    out_manifest = tmp_path / "out.json"
    monkeypatch.setattr(
        sys, "argv",
        ["fit_walkforward_calibrators.py",
         "--manifest", str(_two_row_manifest(tmp_path)),
         "--out-manifest", str(out_manifest),
         "--calibrator-root", str(tmp_path / "out"),
         "--continue-on-failure",
         "--limit", "2",
         "--min-coverage", "1.0"],
    )
    rc = main()
    assert rc == 2  # 1/2 of scheduled set < 100%


# ---- F2: method reuse-gate (same class as window) ------------------------


def test_reuse_refuses_when_existing_artifact_lacks_method_stamp(
    tmp_path: Path, fake_subprocess: MagicMock,
) -> None:
    """An artifact stamped with window=0.0 but no calibrator_method MUST
    NOT be silently reused — the manifest would claim a method the file
    wasn't fit at."""
    cal_path = _cal_dir(tmp_path) / "panel-rank-calibration.json"
    cal_path.write_text(json.dumps({
        "calibrator_window_years": 0.0,
        # no calibrator_method
    }))

    with pytest.raises(RuntimeError, match="lacks calibrator_method metadata"):
        _fit_one(
            _row_fixture(tmp_path),
            calibrator_root=tmp_path / "out",
            training_window_years=3.0,
            calibrator_window_years=0.0,
            method="platt",
            panel=None,
            raw_label_panel=None,
            overwrite=False,
        )
    fake_subprocess.assert_not_called()


def test_reuse_refuses_when_existing_method_differs(
    tmp_path: Path, fake_subprocess: MagicMock,
) -> None:
    """A Platt artifact cannot be reused under --method isotonic."""
    cal_path = _cal_dir(tmp_path) / "panel-rank-calibration.json"
    cal_path.write_text(json.dumps({
        "calibrator_window_years": 0.0,
        "calibrator_method": "platt",
    }))

    with pytest.raises(RuntimeError,
                       match=r"existing method 'platt' != requested 'isotonic'"):
        _fit_one(
            _row_fixture(tmp_path),
            calibrator_root=tmp_path / "out",
            training_window_years=3.0,
            calibrator_window_years=0.0,
            method="isotonic",
            panel=None,
            raw_label_panel=None,
            overwrite=False,
        )
    fake_subprocess.assert_not_called()


def test_reuse_accepts_matching_window_and_method(
    tmp_path: Path, fake_subprocess: MagicMock,
) -> None:
    """Both axes match → reuse OK; subprocess not invoked."""
    cal_path = _cal_dir(tmp_path) / "panel-rank-calibration.json"
    cal_path.write_text(json.dumps({
        "calibrator_window_years": 0.0,
        "calibrator_method": "platt",
    }))

    out = _fit_one(
        _row_fixture(tmp_path),
        calibrator_root=tmp_path / "out",
        training_window_years=3.0,
        calibrator_window_years=0.0,
        method="platt",
        panel=None,
        raw_label_panel=None,
        overwrite=False,
    )
    assert out["calibrator_window_years"] == 0.0
    assert out["calibrator_method"] == "platt"
    fake_subprocess.assert_not_called()


def test_fresh_fit_stamps_method_into_artifact(
    tmp_path: Path, fake_subprocess: MagicMock,
) -> None:
    """Post-fit stamping injects BOTH window and method into the JSON."""
    cal_path = _cal_dir(tmp_path) / "panel-rank-calibration.json"

    def _side_effect(cmd, *a, **kw):
        # Fitter writes an artifact without the policy stamps.
        cal_path.write_text(json.dumps({"unique_field": 7}))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    fake_subprocess.side_effect = _side_effect

    _fit_one(
        _row_fixture(tmp_path),
        calibrator_root=tmp_path / "out",
        training_window_years=3.0,
        calibrator_window_years=0.0,
        method="isotonic",
        panel=None,
        raw_label_panel=None,
        overwrite=True,
    )
    stamped = json.loads(cal_path.read_text())
    assert stamped["calibrator_window_years"] == 0.0
    assert stamped["calibrator_method"] == "isotonic"
    assert stamped["unique_field"] == 7  # original fields preserved
