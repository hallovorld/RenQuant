from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
sys.path.insert(0, str(STRATEGY_DIR))

from kernel.trade_score_diagnostics import (  # noqa: E402
    compute_score_diagnostics,
    render_markdown,
)


def test_rank_score_diagnostic_detects_positive_execution_signal():
    import pandas as pd

    rows = []
    for i in range(40):
        score = i / 39
        rows.append({
            "status": "closed",
            "entry_rank_score": score,
            "entry_mu": score,
            "entry_sigma": 0.2,
            "pnl_pct": score - 0.5,
        })
    payload = compute_score_diagnostics(pd.DataFrame(rows))
    by_score = {m["score_col"]: m for m in payload["metrics"]}

    assert by_score["entry_rank_score"]["spearman"] == pytest.approx(1.0)
    assert by_score["entry_rank_score"]["top_bottom_spread"] > 0


def test_sigma_diagnostic_keeps_negative_direction_visible():
    import pandas as pd

    rows = []
    for i in range(40):
        sigma = 0.1 + i / 100
        rows.append({
            "status": "closed",
            "entry_rank_score": 0.6,
            "entry_mu": 0.03,
            "entry_sigma": sigma,
            "pnl_pct": -sigma,
        })
    payload = compute_score_diagnostics(pd.DataFrame(rows))
    by_score = {m["score_col"]: m for m in payload["metrics"]}

    assert by_score["entry_sigma"]["spearman"] == pytest.approx(-1.0)
    assert by_score["entry_sigma"]["higher_is_better"] is False


def test_open_lots_are_excluded_by_default():
    import pandas as pd

    df = pd.DataFrame([
        {"status": "closed", "entry_rank_score": 0.1, "pnl_pct": -0.1},
        {"status": "closed", "entry_rank_score": 0.9, "pnl_pct": 0.1},
        {"status": "open", "entry_rank_score": 0.9, "pnl_pct": 0.5},
    ])

    payload = compute_score_diagnostics(df, min_n=2)
    assert payload["n_trades"] == 2

    payload_all = compute_score_diagnostics(df, min_n=2, closed_only=False)
    assert payload_all["n_trades"] == 3


def test_markdown_renders_metrics_table():
    import pandas as pd

    payload = compute_score_diagnostics(pd.DataFrame([
        {"status": "closed", "entry_rank_score": i, "pnl_pct": i}
        for i in range(5)
    ]))

    md = render_markdown(payload)
    assert "Trade-Level Score Diagnostics" in md
    assert "entry_rank_score" in md

