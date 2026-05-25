import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import scripts.run_wf_gate as wf_gate  # noqa: E402


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=420)
    rows = []
    for rank, ticker in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]):
        for d in dates:
            rows.append({
                "date": d,
                "ticker": ticker,
                "f1": float(rank),
                "fwd_60d_excess": float(rank) * 0.01,
                "fwd_60d_excess_raw": float(rank) * 0.01,
            })
    return pd.DataFrame(rows)


def test_run_sanity_battery_stamps_regime_ic_metadata(monkeypatch, tmp_path):
    panel = _panel()
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({
        "kind": "panel_ltr_xgboost",
        "feature_cols": ["f1"],
        "lookahead_days": 60,
    }))

    def fake_score(val, *_args, **_kwargs):
        mu = pd.Series(val["fwd_60d_excess"].to_numpy(), index=val.index)
        return mu, {
            "sanity_eval_scope": "walkforward_manifest",
            "sanity_eval_start": str(pd.Timestamp(val["date"].min()).date()),
            "sanity_eval_end": str(pd.Timestamp(val["date"].max()).date()),
        }

    monkeypatch.setattr(wf_gate, "_load_artifact_payload", lambda _p: {
        "kind": "panel_ltr_xgboost",
        "feature_cols": ["f1"],
        "lookahead_days": 60,
    })
    monkeypatch.setattr(wf_gate, "_load_sanity_panel", lambda _cols, _label: (
        panel.copy(),
        {"feature_panel_merge": False},
    ))
    monkeypatch.setattr(wf_gate, "_score_manifest_sanity", fake_score)

    result = wf_gate.run_sanity_battery(
        artifact,
        artifact_usage={
            "eval_scope": "walkforward_manifest",
            "manifest_path": str(tmp_path / "manifest.json"),
        },
    )

    assert result["sanity_label_col"] == "fwd_60d_excess"
    assert "sanity_regime_ic" in result
    assert result["sanity_placebo_aligned_real_ic"] is not None
    assert any(
        row.get("aligned_real_ic") is not None
        for row in result.get("placebo_shift_diagnostics", [])
    )
    regime_ic = result["sanity_regime_ic"]
    assert regime_ic["min_n_dates"] == 30
    assert regime_ic["min_mean_ic"] == 0.02
    assert regime_ic["regimes"]
    assert any(stats["eligible"] for stats in regime_ic["regimes"].values())
