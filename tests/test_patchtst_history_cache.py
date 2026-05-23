from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
STRATEGY = REPO / "backtesting" / "renquant_104"
if str(STRATEGY) not in sys.path:
    sys.path.insert(0, str(STRATEGY))


def test_panel_history_cache_strips_labels_and_forward_returns():
    from adapters.sim import _drop_inference_forbidden_cols

    raw = pd.DataFrame({
        "date": pd.to_datetime(["2026-03-18"]),
        "ticker": ["AAA"],
        "alpha": [1.0],
        "fwd_5d": [0.1],
        "label": [1],
        "split_label": ["train"],
    })

    clean = _drop_inference_forbidden_cols(raw)

    assert list(clean.columns) == ["date", "ticker", "alpha"]


def test_sim_adapter_attaches_point_in_time_history_slice():
    src = (STRATEGY / "adapters" / "sim.py").read_text()

    assert "self._panel_history_cache[\"date\"] < today_ts" in src
    assert "ctx._panel_history" in src
    assert "[-self._panel_history_seq_len:]" in src
