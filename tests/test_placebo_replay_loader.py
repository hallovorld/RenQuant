"""Placebo loader smoke tests (§7.2 R2 battery)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

STRAT = Path(__file__).resolve().parent.parent / "backtesting/renquant_104"
if str(STRAT) not in sys.path:
    sys.path.insert(0, str(STRAT))


def _bar(tickers, fwd):
    from kernel.portfolio_qp.allocator_replay import AllocatorReplayBar
    from kernel.portfolio_qp.constraint_snapshot import ConstraintSnapshot
    n = len(tickers)
    snap = ConstraintSnapshot(
        n=n, tickers=tuple(tickers), w_current=np.zeros(n),
        w_upper_hard=np.full(n, 0.5), w_upper=np.full(n, 0.5), w_lower=0.0,
        dw_max=np.full(n, 1.0), cash_reserve=0.0, turnover_max=None,
        drawdown=0.0, drawdown_limit=0.2, gross_max=None,
        wash_sale_mask=np.zeros(n, dtype=bool),
    )
    return AllocatorReplayBar(
        bar_date="d", snap=snap, mu=np.zeros(n), sigma=np.full(n, 0.1),
        fwd_return=np.array(fwd, dtype=float), regime="BULL_CALM",
        cost_per_trade_bps=0.0,
    )


def test_shuffle_preserves_marginal_breaks_alignment(monkeypatch):
    from kernel.portfolio_qp import placebo_replay_loader as pl
    bars_in = [_bar(["A", "B", "C"], [0.1, 0.2, 0.3])]
    monkeypatch.setattr(pl, "load_replay_bars_from_sim_db",
                        lambda *a, **k: [_bar(["A", "B", "C"], [0.1, 0.2, 0.3])])
    out = pl.load_shuffle_placebo("x", "s", "e", fwd_horizon_days=20)
    # same multiset of returns (marginal preserved), order changed possible
    assert sorted(out[0].fwd_return) == [0.1, 0.2, 0.3]


def test_timeshift_uses_next_bar_returns(monkeypatch):
    from kernel.portfolio_qp import placebo_replay_loader as pl
    def fake(*a, **k):
        return [_bar(["A", "B"], [1.0, 2.0]), _bar(["A", "B"], [3.0, 4.0])]
    monkeypatch.setattr(pl, "load_replay_bars_from_sim_db", fake)
    out = pl.load_timeshift_placebo("x", "s", "e", fwd_horizon_days=20)
    # bar 0 realises bar 1 returns (A=3,B=4)
    assert list(out[0].fwd_return) == [3.0, 4.0]
