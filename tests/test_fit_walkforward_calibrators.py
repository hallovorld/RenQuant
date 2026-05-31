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
    """Capture the subprocess.run cmd argv that _fit_one would have invoked."""
    fake = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0,
                                                              stdout="", stderr=""))
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
