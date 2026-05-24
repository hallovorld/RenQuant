from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
KERNEL = REPO / "backtesting" / "renquant_104"
if str(KERNEL) not in sys.path:
    sys.path.insert(0, str(KERNEL))

from kernel.panel_pipeline.feature_transform import transform_feature_frame  # noqa: E402


def test_raw_feature_space_applies_all_artifact_stats() -> None:
    frame = pd.DataFrame({"alpha": [12.0], "fund": [120.0], "pead": [3.0]})
    meta = {
        "feature_means": [10.0, 100.0, 0.0],
        "feature_stds": [2.0, 10.0, 1.0],
        "feature_norm_kind": ["global_z", "robust_z", "identity"],
    }

    out = transform_feature_frame(frame, ["alpha", "fund", "pead"], meta, source_space="raw")

    assert out.iloc[0].to_dict() == {"alpha": 1.0, "fund": 2.0, "pead": 3.0}


def test_panel_feature_space_only_transforms_raw_panel_columns() -> None:
    frame = pd.DataFrame({"alpha": [0.5], "fund": [120.0], "pead": [3.0]})
    meta = {
        "feature_means": [10.0, 100.0, 0.0],
        "feature_stds": [2.0, 10.0, 1.0],
        "feature_norm_kind": ["global_z", "robust_z", "identity"],
    }

    out = transform_feature_frame(frame, ["alpha", "fund", "pead"], meta, source_space="panel")

    assert out.iloc[0].to_dict() == {"alpha": 0.5, "fund": 2.0, "pead": 3.0}


def test_training_script_panel_matrix_uses_panel_feature_contract() -> None:
    from scripts.train_production_model import panel_training_matrix

    frame = pd.DataFrame({"alpha": [0.5], "fund": [120.0]})

    out = panel_training_matrix(
        frame,
        ["alpha", "fund"],
        pd.Series([10.0, 100.0]).values,
        pd.Series([2.0, 10.0]).values,
        ["global_z", "robust_z"],
    )

    assert out.iloc[0].to_dict() == {"alpha": 0.5, "fund": 2.0}
