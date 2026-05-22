"""Regression guards for panel calibrator expected-return label units."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.fit_calibrator_alpha158_fund import (  # noqa: E402
    _infer_raw_er_label,
    _label_scale_diagnostics,
    _load_expected_return_labels,
)


def _standardized_panel(n_dates: int = 20, n_tickers: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(104)
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    for d in dates:
        vals = rng.normal(size=n_tickers)
        vals = (vals - vals.mean()) / vals.std(ddof=1)
        for i, v in enumerate(vals):
            rows.append({"date": d, "ticker": f"T{i:03d}", "fwd_60d_excess": float(v)})
    return pd.DataFrame(rows)


def test_infers_raw_er_label_from_rank_label():
    assert _infer_raw_er_label("fwd_60d_excess") == "fwd_60d_excess_raw"
    assert _infer_raw_er_label("fwd_20d_excess") == "fwd_20d_excess_raw"
    assert _infer_raw_er_label("fwd_60d_excess_raw") == "fwd_60d_excess_raw"


def test_label_diagnostics_identify_cross_sectional_zscore():
    panel = _standardized_panel()
    diag = _label_scale_diagnostics(panel, "fwd_60d_excess")

    assert diag["looks_cross_sectional_standardized"] is True
    assert 0.95 <= diag["per_date_std_median"] <= 1.05
    assert diag["abs_gt_20pct_fraction"] > 0.5


def test_er_label_contract_rejects_normalized_label_without_escape_hatch(tmp_path):
    panel = _standardized_panel()

    with pytest.raises(ValueError, match="EXPECTED-RETURN-LABEL CONTRACT FAIL"):
        _load_expected_return_labels(
            scoring_panel=panel,
            panel_path=tmp_path / "scoring.parquet",
            raw_label_panel_path=tmp_path / "missing.parquet",
            model_label_col="fwd_60d_excess",
            er_label_col="fwd_60d_excess",
            allow_normalized_er_label=False,
        )


def test_er_label_contract_merges_raw_label_panel(tmp_path):
    scoring = _standardized_panel(n_dates=8, n_tickers=20)
    raw = scoring[["ticker", "date"]].copy()
    rng = np.random.default_rng(7)
    raw["fwd_60d_excess_raw"] = rng.normal(0.01, 0.06, len(raw))
    raw_path = tmp_path / "raw_labels.parquet"
    raw.to_parquet(raw_path, index=False)

    merged, label_col, diag, source = _load_expected_return_labels(
        scoring_panel=scoring,
        panel_path=tmp_path / "scoring.parquet",
        raw_label_panel_path=raw_path,
        model_label_col="fwd_60d_excess",
        er_label_col=None,
        allow_normalized_er_label=False,
    )

    assert label_col == "fwd_60d_excess_raw"
    assert source == str(raw_path)
    assert merged["fwd_60d_excess_raw"].notna().all()
    assert diag["looks_cross_sectional_standardized"] is False
    assert diag["std"] < 0.20
