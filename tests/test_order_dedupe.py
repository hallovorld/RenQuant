from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from kernel.pipeline.order_dedupe import dedupe_buy_orders_first_wins  # noqa: E402


def test_dedupe_buy_orders_first_wins_for_dict_and_object_orders() -> None:
    first = {"ticker": "AAPL", "shares": 1}
    duplicate_dict = {"ticker": "AAPL", "shares": 2}
    first_obj = SimpleNamespace(ticker="MSFT", shares=3)
    duplicate_obj = SimpleNamespace(ticker="MSFT", shares=4)
    no_ticker = {"shares": 5}

    kept, skipped = dedupe_buy_orders_first_wins([
        first,
        duplicate_dict,
        first_obj,
        duplicate_obj,
        no_ticker,
    ])

    assert kept == [first, first_obj, no_ticker]
    assert skipped == [duplicate_dict, duplicate_obj]


def test_sim_live_lean_use_shared_buy_dedupe_helper() -> None:
    for rel in (
        "backtesting/renquant_104/adapters/sim.py",
        "backtesting/renquant_104/adapters/runner.py",
        "backtesting/renquant_104/adapters/lean.py",
    ):
        src = (REPO / rel).read_text()
        assert "dedupe_buy_orders_first_wins(" in src, (
            f"{rel} must route same-bar duplicate buys through the shared "
            "first-write-wins helper"
        )
        assert "duplicate_buy_intent" in src or "duplicate same-bar buy intent" in src
