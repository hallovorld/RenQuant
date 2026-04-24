"""Tests for sim/analysis.py — strip_top_n_trades + compare_strip_levels.

User contract: strip_top_n_trades(result, n=3) must report what APY
would be if the top-3 most profitable completed trades had zero P&L.
Answers the "am I riding lucky mega-winners?" question.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


@dataclass
class _FakeResult:
    """Minimal SimResult-shaped fixture for analysis tests."""
    trade_log:    list[dict] = field(default_factory=list)
    total_return: float = 0.0
    apy:          float = 0.0
    equity_df:    pd.DataFrame = field(default_factory=pd.DataFrame)


def _fake_result(pnl_list, years=1.0, total_return=None):
    # Build a trade_log alternating buys + sells — only sells carry pnl_pct
    log = []
    for i, pnl in enumerate(pnl_list):
        log.append({"action": "buy", "ticker": f"T{i}", "date": f"2024-01-{i+1:02d}"})
        log.append({"action": "sell", "ticker": f"T{i}",
                    "date": f"2024-06-{i+1:02d}", "pnl_pct": pnl})
    # Total return = compound growth of trade pnls (rough single-position model)
    if total_return is None:
        growth = 1.0
        for p in pnl_list:
            growth *= (1.0 + p)
        total_return = growth - 1.0
    # Equity df just needs first/last date for year calc
    idx = pd.bdate_range(start="2024-01-01", periods=int(years * 252))
    eq = pd.DataFrame({"portfolio": [100] * len(idx)}, index=idx)
    return _FakeResult(
        trade_log=log, total_return=total_return,
        apy=(1 + total_return) ** (1 / years) - 1,
        equity_df=eq,
    )


class TestStripTopN:
    def test_strip_zero_returns_original(self):
        from sim.analysis import strip_top_n_trades
        r = _fake_result([0.10, 0.20, -0.05, 0.50])
        s = strip_top_n_trades(r, n=0)
        assert s["stripped_total"] == pytest.approx(r.total_return, abs=1e-9)
        assert s["n_stripped"] == 0
        assert s["stripped_trades"] == []

    def test_strip_top1_removes_biggest_winner(self):
        from sim.analysis import strip_top_n_trades
        r = _fake_result([0.10, 0.20, -0.05, 0.50])  # top = 0.50
        s = strip_top_n_trades(r, n=1)
        # Stripped trade should be the 0.50 one
        assert len(s["stripped_trades"]) == 1
        assert s["stripped_trades"][0]["pnl_pct"] == pytest.approx(0.50)
        # Growth factor: original 1.10 * 1.20 * 0.95 * 1.50 = 1.8810
        # Stripped:     1.10 * 1.20 * 0.95           = 1.2540
        # (1 + total_return_stripped) = original / (1 + top_pnl) = 1.8810 / 1.50 = 1.2540
        assert s["stripped_total"] == pytest.approx(0.2540, abs=1e-4)

    def test_strip_top3(self):
        from sim.analysis import strip_top_n_trades
        r = _fake_result([0.05, 0.10, 0.80, -0.10, 1.20, 0.40])
        s = strip_top_n_trades(r, n=3)
        stripped_pnls = sorted(t["pnl_pct"] for t in s["stripped_trades"])
        # Top 3 = 1.20, 0.80, 0.40
        assert stripped_pnls == pytest.approx([0.40, 0.80, 1.20])
        # APY should be significantly below original
        assert s["apy"] < r.apy

    def test_strip_n_larger_than_trades_caps(self):
        from sim.analysis import strip_top_n_trades
        r = _fake_result([0.10, 0.20])
        s = strip_top_n_trades(r, n=100)
        # Can't strip more than exist
        assert s["n_stripped"] == 2
        # With all profitable trades stripped, return → ~0%
        assert abs(s["stripped_total"]) < 1e-6

    def test_empty_trade_log_returns_original(self):
        from sim.analysis import strip_top_n_trades
        r = _fake_result([])
        s = strip_top_n_trades(r, n=3)
        assert s["n_stripped"] == 0
        assert s["stripped_total"] == r.total_return

    def test_median_trade_in_output(self):
        from sim.analysis import strip_top_n_trades
        r = _fake_result([-0.05, 0.00, 0.05, 0.10, 0.20])
        s = strip_top_n_trades(r, n=0)
        # sorted: -0.05, 0, 0.05, 0.10, 0.20 → median is middle = 0.05
        assert s["median_trade"] == pytest.approx(0.05)

    def test_median_stripped_excludes_top_n(self):
        from sim.analysis import strip_top_n_trades
        r = _fake_result([0.05, 0.10, 0.80, -0.10, 1.20, 0.40])
        s = strip_top_n_trades(r, n=3)
        # Remaining (after stripping top 3) = 0.05, 0.10, -0.10 → median = 0.05
        assert s["median_stripped"] == pytest.approx(0.05)


class TestCompareStripLevels:
    def test_prints_ladder(self, capsys):
        from sim.analysis import compare_strip_levels
        r = _fake_result([0.05, 0.10, 0.80, 1.20, 0.40])
        compare_strip_levels(r, levels=[0, 1, 3])
        out = capsys.readouterr().out
        # Should show all 3 levels
        assert "strip" in out and "apy" in out
        assert "0" in out and "3" in out
        # ⭐ appears on the N=0 row
        assert "⭐" in out
