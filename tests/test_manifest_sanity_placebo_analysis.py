import numpy as np
import pandas as pd
import pytest

from scripts.analyze_manifest_sanity_placebo import (
    render_markdown,
    shift_diagnostics,
    summarize_ic,
)


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=90)
    rows = []
    for t_idx, ticker in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]):
        # Stable cross-sectional ordering plus a small time trend makes
        # future labels persistent. The diagnostic should expose that via
        # label_autocorr_ic instead of calling it model alpha.
        base = float(t_idx) * 0.05
        for d_idx, d in enumerate(dates):
            rows.append({
                "ticker": ticker,
                "date": d,
                "fwd_60d_excess_raw": base + d_idx * 0.001,
            })
    return pd.DataFrame(rows)


def test_summarize_ic_reports_cross_sectional_mean() -> None:
    dates = pd.to_datetime(["2024-01-02"] * 6 + ["2024-01-03"] * 6)
    pred = np.tile(np.arange(6), 2)
    label = np.tile(np.arange(6), 2)

    stats = summarize_ic(pred, label, dates, min_names=5)

    assert stats["mean_ic"] == 1.0
    assert stats["n_dates"] == 2
    assert stats["n_rows"] == 12


def test_shift_diagnostics_separates_model_placebo_from_label_persistence() -> None:
    panel = _panel()
    val = panel[panel["date"] >= pd.Timestamp("2024-03-01")].copy()
    # A model that is just today's raw label rank will look good against
    # shifted labels when labels are persistent. We need the autocorr column
    # to reveal this confounder.
    mu = val["fwd_60d_excess_raw"].copy()

    rows = shift_diagnostics(
        panel,
        val,
        mu,
        "fwd_60d_excess_raw",
        shifts=[5],
        min_names=5,
    )

    assert len(rows) == 1
    assert rows[0]["aligned_real_ic"] > 0.99
    assert rows[0]["model_placebo_ic"] > 0.99
    assert rows[0]["label_autocorr_ic"] > 0.99
    assert rows[0]["model_placebo_abs_ratio_to_aligned_real"] > 0.99


def test_shift_diagnostics_compares_placebo_to_aligned_real_sample() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    rows = []
    for rank, ticker in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]):
        for i, d in enumerate(dates):
            rows.append({
                "ticker": ticker,
                "date": d,
                # First 9 dates match ticker rank; final 3 dates invert it.
                # A 3-day placebo can only use the first 9 dates. The
                # diagnostic must therefore compare placebo to real IC on
                # those same 9 dates, not the full 12-date validation sample.
                "fwd_60d_excess_raw": float(rank if i < 9 else -rank),
            })
    panel = pd.DataFrame(rows)
    val = panel.copy()
    mu = pd.Series(val["ticker"].map({
        "AAA": 0.0,
        "BBB": 1.0,
        "CCC": 2.0,
        "DDD": 3.0,
        "EEE": 4.0,
        "FFF": 5.0,
    }).to_numpy(), index=val.index)

    row = shift_diagnostics(
        panel,
        val,
        mu,
        "fwd_60d_excess_raw",
        shifts=[3],
        min_names=5,
    )[0]

    assert row["full_real_ic"] < row["aligned_real_ic"]
    assert row["aligned_real_ic"] == 1.0
    assert row["model_placebo_abs_ratio_to_aligned_real"] == (
        abs(row["model_placebo_ic"]) / abs(row["aligned_real_ic"])
    )


def test_markdown_marks_failed_promotion_evidence() -> None:
    result = {
        "artifact": "a.json",
        "manifest": "m.json",
        "label": "fwd_60d_excess_raw",
        "validation": {
            "start": "2025-01-01",
            "end": "2025-12-31",
            "n_dates": 10,
            "n_rows": 60,
        },
        "real_ic": {"mean_ic": 0.01},
        "interpretation": {
            "promotion_evidence": False,
            "aligned_real_60_ic": 0.01,
            "placebo_60_ic": 0.02,
            "label_autocorr_60_ic": 0.04,
            "primary_warning": "60-day placebo is too large relative to aligned real IC",
        },
        "shift_diagnostics": [{
            "shift_days": 60,
            "aligned_real_ic": 0.01,
            "model_placebo_ic": 0.02,
            "label_autocorr_ic": 0.04,
            "n_rows": 60,
            "n_dates": 10,
        }],
        "by_regime": {
            "BULL_CALM": {
                "mean_ic": 0.01,
                "hit_rate": 0.5,
                "n_dates": 10,
                "n_raw_rows": 60,
                "mean_confidence": 0.8,
            }
        },
    }

    md = render_markdown(result)

    assert "Promotion evidence: `False`" in md
    assert "60-day placebo is too large" in md
    assert "60d aligned real IC" in md


def test_load_sanity_panel_supplements_addendum_from_training_panel(
    tmp_path, monkeypatch
) -> None:
    """Opt-in addendum features (Track B) are absent from the rawlabel sanity
    panel. They must be supplemented column-wise from the production training
    panel, leaving the rawlabel base features untouched so addendum sanity runs
    stay apples-to-apples with the non-addendum (baseline) run.

    Regression guard for the 2026-06-05 Track-B verdict run, where the sanity
    eval crashed with KeyError because mom_carry_12_1/beta_dm/etc. lived only in
    the training panel, not in the rawlabel or transformer panels.
    """
    import scripts.run_wf_gate as wf

    data = tmp_path / "data"
    data.mkdir()
    dates = pd.bdate_range("2024-01-01", periods=4)
    rows = [(t, d) for t in ("AAA", "BBB") for d in dates]
    raw = pd.DataFrame({
        "ticker": [t for t, _ in rows],
        "date": [d for _, d in rows],
        "alpha_base": [0.1 * i for i in range(len(rows))],
        "fwd_60d_excess_raw": [0.2 * i for i in range(len(rows))],
    })
    raw.to_parquet(data / "alpha158_291_fundamental_dataset_rawlabel.parquet")
    # Training panel carries the base feature AND the Track-B addendum column.
    train = raw[["ticker", "date", "alpha_base"]].copy()
    train["mom_carry_12_1"] = [0.5 * i for i in range(len(rows))]
    train.to_parquet(data / "alpha158_291_fundamental_dataset.parquet")

    monkeypatch.setattr(wf, "REPO", tmp_path)

    panel, meta = wf._load_sanity_panel(
        ["alpha_base", "mom_carry_12_1"], "fwd_60d_excess_raw"
    )

    # Addendum column supplemented, every row populated (1:1 key merge).
    assert "mom_carry_12_1" in panel.columns
    assert panel["mom_carry_12_1"].notna().all()
    # Base feature preserved from rawlabel (NOT the training panel copy).
    assert "alpha_base" in panel.columns
    assert meta["supplement_only_missing"] is True
    assert meta["feature_cols_supplied_by_feature_panel"] == ["mom_carry_12_1"]
    assert "alpha158_291_fundamental_dataset.parquet" in meta["sanity_feature_panel"]


def test_load_sanity_panel_rejects_duplicate_training_panel_keys(
    tmp_path, monkeypatch
) -> None:
    import scripts.run_wf_gate as wf

    data = tmp_path / "data"
    data.mkdir()
    dates = pd.bdate_range("2024-01-01", periods=2)
    raw = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "date": dates,
        "alpha_base": [0.1, 0.2],
        "fwd_60d_excess_raw": [0.3, 0.4],
    })
    raw.to_parquet(data / "alpha158_291_fundamental_dataset_rawlabel.parquet")
    train = pd.DataFrame({
        "ticker": ["AAA", "AAA", "AAA"],
        "date": [dates[0], dates[0], dates[1]],
        "mom_carry_12_1": [0.5, 0.6, 0.7],
    })
    train.to_parquet(data / "alpha158_291_fundamental_dataset.parquet")
    monkeypatch.setattr(wf, "REPO", tmp_path)

    with pytest.raises(ValueError, match="duplicate"):
        wf._load_sanity_panel(
            ["alpha_base", "mom_carry_12_1"], "fwd_60d_excess_raw"
        )


def test_load_sanity_panel_rejects_incomplete_training_panel_coverage(
    tmp_path, monkeypatch
) -> None:
    import scripts.run_wf_gate as wf

    data = tmp_path / "data"
    data.mkdir()
    dates = pd.bdate_range("2024-01-01", periods=2)
    raw = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "date": dates,
        "alpha_base": [0.1, 0.2],
        "fwd_60d_excess_raw": [0.3, 0.4],
    })
    raw.to_parquet(data / "alpha158_291_fundamental_dataset_rawlabel.parquet")
    train = pd.DataFrame({
        "ticker": ["AAA"],
        "date": [dates[0]],
        "mom_carry_12_1": [0.5],
    })
    train.to_parquet(data / "alpha158_291_fundamental_dataset.parquet")
    monkeypatch.setattr(wf, "REPO", tmp_path)

    with pytest.raises(ValueError, match="missing values"):
        wf._load_sanity_panel(
            ["alpha_base", "mom_carry_12_1"], "fwd_60d_excess_raw"
        )


def test_load_sanity_panel_drops_tail_edge_coverage_gap(tmp_path, monkeypatch) -> None:
    """A tail-edge coverage gap (rawlabel has keys the training panel lacks, e.g.
    the rawlabel's last date not yet stamped into the training panel) below the
    1% tolerance is DROPPED, not hard-failed — the model scores NaN natively and
    a handful of tail rows is immaterial to a per-regime IC. Regression guard for
    the 2026-06-06 Track-C specialist eval (109/715629 = 0.02% gap).
    """
    import scripts.run_wf_gate as wf

    data = tmp_path / "data"
    data.mkdir()
    dates = pd.bdate_range("2024-01-01", periods=200)
    raw = pd.DataFrame({
        "ticker": ["AAA"] * 200,
        "date": dates,
        "alpha_base": [0.001 * i for i in range(200)],
        "fwd_60d_excess_raw": [0.002 * i for i in range(200)],
    })
    raw.to_parquet(data / "alpha158_291_fundamental_dataset_rawlabel.parquet")
    # Training panel covers all dates EXCEPT the last one → 1/200 = 0.5% gap (< 1%).
    train = pd.DataFrame({
        "ticker": ["AAA"] * 199,
        "date": dates[:199],
        "mom_carry_12_1": [0.5 * i for i in range(199)],
    })
    train.to_parquet(data / "alpha158_291_fundamental_dataset.parquet")
    monkeypatch.setattr(wf, "REPO", tmp_path)

    panel, meta = wf._load_sanity_panel(
        ["alpha_base", "mom_carry_12_1"], "fwd_60d_excess_raw"
    )

    assert len(panel) == 199  # the 1 gap row dropped
    assert panel["mom_carry_12_1"].notna().all()
    assert meta["supplement_only_missing"] is True
