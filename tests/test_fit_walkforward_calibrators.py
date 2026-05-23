from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parent.parent
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from fit_walkforward_calibrators import _date_window  # noqa: E402


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
