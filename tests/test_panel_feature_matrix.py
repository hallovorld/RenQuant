"""Tests for kernel/panel_pipeline/feature_matrix.py (inference-side matrix builder)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _make_feature_frame(dates, values_x1, values_x2):
    return pd.DataFrame(
        {"x1": values_x1, "x2": values_x2},
        index=pd.DatetimeIndex(dates),
    )


def _trained_artifact(tmp_path: Path):
    """Train a tiny PanelLTRModel with feature_cols=[x1, x2, factor_z]."""
    from training_panel.ltr_model import PanelLTRModel

    rng = np.random.default_rng(1)
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=15)
    for d in dates:
        for i in range(5):
            x1 = rng.normal()
            x2 = rng.normal()
            fz = rng.normal()
            rows.append({
                "date": d, "ticker": f"T{i}",
                "x1": x1, "x2": x2, "factor_z": fz,
                "label": x1 + 0.3 * fz + 0.1 * rng.normal(),
                "weight": 1.0,
            })
    panel = pd.DataFrame(rows).sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    grp = panel.groupby("date", sort=True).size().values.astype(np.int32)
    m = PanelLTRModel()
    m.train(panel, grp, feature_cols=["x1", "x2", "factor_z"],
            num_boost_round=20, early_stopping_rounds=None)
    path = tmp_path / "panel_model.json"
    m.save(path, metadata={"training_notes": "feature-matrix test"})
    return path


class TestBuildInferenceMatrix:
    def test_columns_match_feature_cols_exactly(self):
        from kernel.panel_pipeline.feature_matrix import build_inference_matrix

        dates = pd.bdate_range("2024-01-01", periods=5)
        ff = {
            "AAA": _make_feature_frame(dates, [0.1, 0.2, 0.3, 0.4, 0.5], [1.0] * 5),
            "BBB": _make_feature_frame(dates, [1.0, 1.1, 1.2, 1.3, 1.4], [2.0] * 5),
        }
        today = dates[-1]
        mat = build_inference_matrix(ff, factor_frames=None, today=today,
                                     feature_cols=["x1", "x2"])
        assert list(mat.columns) == ["x1", "x2"]
        assert set(mat.index) == {"AAA", "BBB"}

    def test_picks_today_row(self):
        from kernel.panel_pipeline.feature_matrix import build_inference_matrix

        dates = pd.bdate_range("2024-01-01", periods=5)
        ff = {"AAA": _make_feature_frame(dates, [10.0, 20.0, 30.0, 40.0, 50.0], [1.0] * 5)}
        mat = build_inference_matrix(ff, None, today=dates[2],
                                     feature_cols=["x1", "x2"])
        assert mat.loc["AAA", "x1"] == 30.0

    def test_fallback_to_most_recent_row_before_today(self):
        from kernel.panel_pipeline.feature_matrix import build_inference_matrix

        dates = pd.bdate_range("2024-01-01", periods=3)
        ff = {"AAA": _make_feature_frame(dates, [10.0, 20.0, 30.0], [1.0] * 3)}
        # "today" is a weekend after the last bdate — should fall back to dates[-1]
        today = dates[-1] + pd.Timedelta(days=3)
        mat = build_inference_matrix(ff, None, today=today,
                                     feature_cols=["x1", "x2"])
        assert mat.loc["AAA", "x1"] == 30.0

    def test_tickers_without_history_are_skipped(self):
        from kernel.panel_pipeline.feature_matrix import build_inference_matrix

        dates = pd.bdate_range("2024-01-01", periods=3)
        ff = {
            "AAA": _make_feature_frame(dates, [1.0, 2.0, 3.0], [1.0] * 3),
            "BBB": _make_feature_frame(
                pd.bdate_range("2024-06-01", periods=3),  # all after `today`
                [1.0, 2.0, 3.0], [1.0] * 3,
            ),
        }
        mat = build_inference_matrix(ff, None, today=dates[-1],
                                     feature_cols=["x1", "x2"])
        assert list(mat.index) == ["AAA"]

    def test_empty_inputs_returns_empty_frame_with_correct_columns(self):
        from kernel.panel_pipeline.feature_matrix import build_inference_matrix

        mat = build_inference_matrix({}, None, today="2024-01-01",
                                     feature_cols=["x1", "x2"])
        assert mat.empty
        assert list(mat.columns) == ["x1", "x2"]

    def test_missing_columns_filled_with_nan(self):
        from kernel.panel_pipeline.feature_matrix import build_inference_matrix

        dates = pd.bdate_range("2024-01-01", periods=3)
        ff = {"AAA": _make_feature_frame(dates, [1.0, 2.0, 3.0], [1.0] * 3)}
        mat = build_inference_matrix(ff, None, today=dates[-1],
                                     feature_cols=["x1", "x2", "missing_col"])
        assert list(mat.columns) == ["x1", "x2", "missing_col"]
        assert np.isnan(mat.loc["AAA", "missing_col"])

    def test_many_missing_columns_do_not_fragment_frame(self):
        from kernel.panel_pipeline.feature_matrix import build_inference_matrix

        dates = pd.bdate_range("2024-01-01", periods=3)
        ff = {"AAA": _make_feature_frame(dates, [1.0, 2.0, 3.0], [1.0] * 3)}
        missing_cols = [f"missing_{i}" for i in range(180)]
        feature_cols = ["x1", "x2", *missing_cols]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mat = build_inference_matrix(
                ff,
                None,
                today=dates[-1],
                feature_cols=feature_cols,
            )

        perf_warnings = [
            w for w in caught
            if issubclass(w.category, pd.errors.PerformanceWarning)
        ]
        assert perf_warnings == []
        assert list(mat.columns) == feature_cols
        assert mat.loc["AAA", missing_cols].isna().all()

    def test_factor_frames_merged(self):
        from kernel.panel_pipeline.feature_matrix import build_inference_matrix

        dates = pd.bdate_range("2024-01-01", periods=3)
        ff = {"AAA": _make_feature_frame(dates, [1.0, 2.0, 3.0], [10.0, 20.0, 30.0])}
        fac = {
            "AAA": pd.DataFrame({"size_z": [0.5, 0.6, 0.7]},
                                index=pd.DatetimeIndex(dates)),
        }
        mat = build_inference_matrix(ff, fac, today=dates[-1],
                                     feature_cols=["x1", "x2", "size_z"])
        assert mat.loc["AAA", "x1"] == 3.0
        assert mat.loc["AAA", "size_z"] == 0.7

    def test_missingness_indicators_set_when_col_present_and_nan(self):
        from kernel.panel_pipeline.feature_matrix import build_inference_matrix

        dates = pd.bdate_range("2024-01-01", periods=2)
        ff = {"AAA": _make_feature_frame(dates, [1.0, np.nan], [10.0, 20.0])}
        mat = build_inference_matrix(
            ff, None, today=dates[-1],
            feature_cols=["x1", "x2", "x1_is_missing"],
            nan_prone_cols=["x1"],
        )
        assert mat.loc["AAA", "x1_is_missing"] == 1

    def test_missingness_indicators_zero_when_col_present_and_not_nan(self):
        from kernel.panel_pipeline.feature_matrix import build_inference_matrix

        dates = pd.bdate_range("2024-01-01", periods=2)
        ff = {"AAA": _make_feature_frame(dates, [1.0, 5.0], [10.0, 20.0])}
        mat = build_inference_matrix(
            ff, None, today=dates[-1],
            feature_cols=["x1", "x1_is_missing"],
            nan_prone_cols=["x1"],
        )
        assert mat.loc["AAA", "x1_is_missing"] == 0

    def test_missingness_indicator_is_one_when_col_absent(self):
        from kernel.panel_pipeline.feature_matrix import build_inference_matrix

        dates = pd.bdate_range("2024-01-01", periods=2)
        # feature frame has no `x_special` column at all
        ff = {"AAA": _make_feature_frame(dates, [1.0, 5.0], [10.0, 20.0])}
        mat = build_inference_matrix(
            ff, None, today=dates[-1],
            feature_cols=["x1", "x_special", "x_special_is_missing"],
            nan_prone_cols=["x_special"],
        )
        assert mat.loc["AAA", "x_special_is_missing"] == 1
        assert np.isnan(mat.loc["AAA", "x_special"])


class TestRunPanelInference:
    def test_end_to_end_scores_per_ticker(self, tmp_path):
        from kernel.panel_pipeline.feature_matrix import run_panel_inference

        path = _trained_artifact(tmp_path)

        dates = pd.bdate_range("2024-01-01", periods=5)
        ff = {
            "T0": _make_feature_frame(dates, [0.1, 0.2, 0.3, 0.4, 0.5],
                                       [1.0, 1.1, 1.2, 1.3, 1.4]),
            "T1": _make_feature_frame(dates, [-0.1, -0.2, -0.3, -0.4, -0.5],
                                       [0.5, 0.6, 0.7, 0.8, 0.9]),
        }
        fac = {
            "T0": pd.DataFrame({"factor_z": [0.1, 0.2, 0.3, 0.4, 0.5]},
                               index=pd.DatetimeIndex(dates)),
            "T1": pd.DataFrame({"factor_z": [-0.1, -0.2, -0.3, -0.4, -0.5]},
                               index=pd.DatetimeIndex(dates)),
        }
        scores = run_panel_inference(ff, fac, today=dates[-1], artifact_path=path)
        assert isinstance(scores, pd.Series)
        assert set(scores.index) == {"T0", "T1"}
        assert scores.notna().all()

    def test_empty_feature_frames_returns_empty_series(self, tmp_path):
        from kernel.panel_pipeline.feature_matrix import run_panel_inference

        path = _trained_artifact(tmp_path)
        out = run_panel_inference({}, None, today="2024-01-10",
                                  artifact_path=path)
        assert isinstance(out, pd.Series)
        assert out.empty
        assert out.name == "panel_score"

    def test_matrix_columns_aligned_to_artifact(self, tmp_path):
        """Extra columns in inputs should be dropped; missing ones filled with NaN."""
        from kernel.panel_pipeline.feature_matrix import build_inference_matrix
        from kernel.panel_pipeline import PanelScorer

        path = _trained_artifact(tmp_path)
        scorer = PanelScorer.load(path)

        dates = pd.bdate_range("2024-01-01", periods=3)
        ff = {
            "T0": pd.DataFrame(
                {"x1": [0.1, 0.2, 0.3], "x2": [1.0, 2.0, 3.0], "extra": [9, 9, 9]},
                index=pd.DatetimeIndex(dates),
            ),
        }
        fac = {
            "T0": pd.DataFrame({"factor_z": [0.5, 0.6, 0.7]},
                               index=pd.DatetimeIndex(dates)),
        }
        mat = build_inference_matrix(ff, fac, today=dates[-1],
                                     feature_cols=scorer.feature_cols)
        assert list(mat.columns) == scorer.feature_cols
        assert "extra" not in mat.columns
