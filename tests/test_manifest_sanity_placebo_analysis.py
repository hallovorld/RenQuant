import numpy as np
import pandas as pd

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
    assert rows[0]["model_placebo_ic"] > 0.99
    assert rows[0]["label_autocorr_ic"] > 0.99


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
            "placebo_60_ic": 0.02,
            "label_autocorr_60_ic": 0.04,
            "primary_warning": "60-day placebo is too large relative to real IC",
        },
        "shift_diagnostics": [{
            "shift_days": 60,
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
