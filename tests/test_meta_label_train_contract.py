from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts._meta_label_train import select_path_rule_training_events  # noqa: E402


def test_select_path_rule_training_events_matches_runtime_veto_surface() -> None:
    df = pd.DataFrame([
        {
            "meta_label": 1,
            "any_trigger": 1,
            "trigger_stop_loss": 1,
            "trigger_trailing_stop": 0,
            "trigger_single_day_loss": 0,
            "trigger_max_hold": 0,
            "ticker": "SL",
        },
        {
            "meta_label": 0,
            "any_trigger": 1,
            "trigger_stop_loss": 0,
            "trigger_trailing_stop": 0,
            "trigger_single_day_loss": 0,
            "trigger_max_hold": 0,
            "ticker": "MODEL_EXIT",
        },
        {
            "meta_label": np.nan,
            "any_trigger": 1,
            "trigger_stop_loss": 1,
            "trigger_trailing_stop": 0,
            "trigger_single_day_loss": 0,
            "trigger_max_hold": 0,
            "ticker": "UNLABELED",
        },
        {
            "meta_label": 1,
            "any_trigger": 1,
            "trigger_stop_loss": 0,
            "trigger_trailing_stop": 0,
            "trigger_single_day_loss": 1,
            "trigger_max_hold": 0,
            "ticker": "SDL",
        },
    ])

    out = select_path_rule_training_events(df)

    assert out["ticker"].tolist() == ["SL", "SDL"]
    assert out["meta_label"].tolist() == [1, 1]
