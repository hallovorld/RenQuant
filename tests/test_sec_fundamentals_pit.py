"""Point-in-time guards for SEC fundamentals."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.fetch_sec_fundamentals import (  # noqa: E402
    build_quarterly_panel,
    forward_fill_to_daily,
)
from scripts.build_extended_fundamentals import (  # noqa: E402
    build_quarterly_panel as build_extended_quarterly_panel,
)


def test_quarterly_panel_uses_actual_sec_filed_date_not_fixed_lag():
    raw = pd.DataFrame([
        {
            "cik": 1,
            "end": "2024-03-31",
            "filed": "2024-04-25",
            "concept": "NetIncomeLoss",
            "val": 100.0,
        },
        {
            "cik": 1,
            "end": "2024-03-31",
            "filed": "2024-05-03",
            "concept": "Assets",
            "val": 1000.0,
        },
    ])

    panel = build_quarterly_panel(raw, {1: "AAA"})

    assert len(panel) == 1
    # Row combines concepts, so it is available only after the last filed leg.
    assert panel.loc[0, "available_date"] == pd.Timestamp("2024-05-03")
    assert panel.loc[0, "available_date"] != pd.Timestamp("2024-05-15")


def test_quarterly_panel_falls_back_to_45d_when_filed_missing():
    raw = pd.DataFrame([
        {
            "cik": 1,
            "end": "2024-03-31",
            "concept": "NetIncomeLoss",
            "val": 100.0,
        },
    ])

    panel = build_quarterly_panel(raw, {1: "AAA"})

    assert panel.loc[0, "available_date"] == pd.Timestamp("2024-05-15")


def test_daily_forward_fill_starts_on_available_date():
    quarterly = pd.DataFrame([
        {
            "ticker": "AAA",
            "end": pd.Timestamp("2024-03-31"),
            "available_date": pd.Timestamp("2024-05-03"),
            "NetIncomeLoss": 100.0,
        },
    ])
    dates = pd.date_range("2024-05-01", "2024-05-05", freq="D")

    daily = forward_fill_to_daily(quarterly, dates, ["AAA"])
    by_date = daily.set_index("date")

    assert pd.isna(by_date.loc[pd.Timestamp("2024-05-02"), "NetIncomeLoss"])
    assert by_date.loc[pd.Timestamp("2024-05-03"), "NetIncomeLoss"] == 100.0


def test_extended_fundamentals_uses_actual_filed_date():
    raw = pd.DataFrame([
        {
            "ticker": "AAA",
            "end": "2024-03-31",
            "filed": "2024-04-22",
            "concept": "Revenues",
            "val": 200.0,
        },
        {
            "ticker": "AAA",
            "end": "2024-03-31",
            "filed": "2024-05-01",
            "concept": "Assets",
            "val": 1000.0,
        },
    ])

    panel = build_extended_quarterly_panel(raw)

    assert panel.loc[0, "available_date"] == pd.Timestamp("2024-05-01")
