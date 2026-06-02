"""Track B: pin that ``--include-features`` flows through train_walkforward_panel
to the per-cutoff ``train_production_model.py`` subprocess.

§7.2.1 R3: this test was added in the same PR as the ``--include-features``
flag itself so the gate fires on the FIRST artifact (not retrofitted later).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# scripts/ is at repo root, not on the package path; insert it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.train_walkforward_panel import train_one_cutoff  # noqa: E402


class _FakeProc:
    """Stand-in for ``subprocess.run`` result; pretends the trainer succeeded."""

    def __init__(self) -> None:
        self.returncode = 0
        self.stdout = ""
        self.stderr = ""


def _patched_subprocess(captured_cmds: list[list[str]]):
    """Capture every subprocess.run cmd into ``captured_cmds`` and return fake success."""

    def _run(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        return _FakeProc()

    return _run


def test_include_features_flag_flows_into_train_production_subprocess(tmp_path: Path) -> None:
    """When ``--include-features`` is passed, train_one_cutoff invokes
    ``train_production_model.py --include-features <list>``. This is the
    single point where the WF driver hands the addendum opt-in to the
    per-cutoff trainer.
    """
    captured: list[list[str]] = []
    with patch("scripts.train_walkforward_panel.subprocess.run", _patched_subprocess(captured)):
        # ``fit_calibrator=False`` skips the calibrator subprocess so the
        # captured list contains ONLY the trainer cmd.
        ok, artifact_path, _calibrator_path, err = train_one_cutoff(
            pd.Timestamp("2024-06-01"),
            tmp_path,
            include_features="mom_carry_12_1,beta_dm,rvar_total,idio_vol_market",
            fit_calibrator=False,
        )
    assert ok, err
    assert captured, "no subprocess captured"
    cmd = captured[0]
    assert "--include-features" in cmd
    idx = cmd.index("--include-features")
    assert cmd[idx + 1] == "mom_carry_12_1,beta_dm,rvar_total,idio_vol_market"
    # The cutoff and output path also flow.
    assert "--train-cutoff" in cmd and "2024-06-01" in cmd
    assert any("walkforward" in part for part in cmd), \
        f"cutoff artifact path must contain 'walkforward'; got {cmd}"


def test_default_omits_include_features_flag(tmp_path: Path) -> None:
    """When ``--include-features`` is NOT passed, the subprocess command does
    NOT carry it. Preserves byte-identical CLI for baseline retrains.
    """
    captured: list[list[str]] = []
    with patch("scripts.train_walkforward_panel.subprocess.run", _patched_subprocess(captured)):
        ok, _, _, err = train_one_cutoff(
            pd.Timestamp("2024-06-01"),
            tmp_path,
            fit_calibrator=False,
        )
    assert ok, err
    assert captured
    cmd = captured[0]
    assert "--include-features" not in cmd


def test_include_features_changes_side_label(tmp_path: Path) -> None:
    """The addendum variant uses a distinct side_label so its artifacts are
    discoverable in the data/sim_runs.db training_runs table even when sharing
    cutoff_date with baseline.
    """
    captured: list[list[str]] = []
    with patch("scripts.train_walkforward_panel.subprocess.run", _patched_subprocess(captured)):
        train_one_cutoff(
            pd.Timestamp("2024-06-01"),
            tmp_path,
            include_features="mom_carry_12_1",
            fit_calibrator=False,
        )
    cmd = captured[0]
    idx = cmd.index("--side-label")
    assert cmd[idx + 1].startswith("walkforward_addendum_"), \
        f"expected addendum side_label, got {cmd[idx + 1]}"
