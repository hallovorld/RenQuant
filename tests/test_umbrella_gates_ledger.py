"""Umbrella gate dual-write (task_gates ×9) + gate_verdicts ledger mirror.

Design: eng plan S2-PR4 / errata C; mirrors renquant-pipeline #121
(dual-write), #124 (ledger). DELIBERATELY NOT retired here: the sibling
renquant-pipeline pin predates kernel.gate_registry — retiring while the
registry import can fail would silently disable every gate (see
_submit_gate_verdict docstring). Retirement follows the pin advance.
"""
from __future__ import annotations

import datetime
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

# Self-validating harness (codex review, #322): the registry lives in the
# renquant-pipeline sibling; bare `pytest` in this repo has no sibling on
# sys.path, so the lazy import degrades and every ledger assertion goes
# vacuous. Bootstrap the sibling src explicitly and FAIL (not skip) if it
# is missing — the runtime pin guarantees it exists on any valid checkout.
_SIBLING_CANDIDATES = (
    REPO.parent / "renquant-pipeline" / "src",                      # sibling layout
    Path.home() / "git" / "github" / "renquant-pipeline" / "src",   # canonical root (worktrees)
)
_SIBLING_SRC = next((p for p in _SIBLING_CANDIDATES if p.exists()), None)
assert _SIBLING_SRC is not None, (
    f"renquant-pipeline sibling missing (tried {[str(p) for p in _SIBLING_CANDIDATES]}) "
    f"— required for gate-ledger tests (runtime pin guarantees it)")
if str(_SIBLING_SRC) not in sys.path:
    sys.path.insert(0, str(_SIBLING_SRC))


from kernel.persistence import get_connection, record_gate_verdicts  # noqa: E402
from kernel.pipeline.task_gates import (  # noqa: E402
    DrawdownGateTask,
    EMA50GateTask,
    FlattenCooldownGateTask,
    RegimeAlphaGateTask,
    TransitionWindowTask,
    VelocityCrashTask,
)


def _ctx(**kw) -> SimpleNamespace:
    base = dict(
        config={}, counters={}, skip_buys=False, buy_blocked=False,
        bear_only=False, regime="BULL_CALM", confidence=0.9,
        regime_state=None, monitor_state={}, spy_returns=[],
        ohlcv={}, today=datetime.date(2026, 6, 12), gate_registry=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _gates(ctx):
    if getattr(ctx, "gate_registry", None) is None:
        return []
    return [r["gate"] for r in ctx.gate_registry.ledger_rows(run_id="t")]


class TestDualWriteStaysDual:
    """Direct write AND submit — equivalence, no retirement."""

    def test_drawdown(self):
        ctx = _ctx(skip_buys=True)
        DrawdownGateTask().run(ctx)
        assert not ctx.buy_blocked, "retired: job boundary applies the flag"
        assert ctx._gate_block_pending
        assert _gates(ctx) == ["drawdown_circuit"]

    def test_flatten_same_bar(self):
        ctx = _ctx(monitor_state={"flatten_last_date_iso": "2026-06-12",
                                  "flatten_cooldown_bars": 3})
        FlattenCooldownGateTask().run(ctx)
        assert not ctx.buy_blocked and ctx._gate_block_pending
        assert _gates(ctx) == ["flatten_cooldown"]

    def test_transition(self):
        ctx = _ctx(regime_state=SimpleNamespace(in_transition=True))
        TransitionWindowTask().run(ctx)
        assert not ctx.buy_blocked and ctx._gate_block_pending
        assert _gates(ctx) == ["transition_window"]

    def test_regime_alpha(self):
        ctx = _ctx(config={"regime_params": {"BULL_CALM": {"disable_new_buys": True}}})
        RegimeAlphaGateTask().run(ctx)
        assert not ctx.buy_blocked and ctx._gate_block_pending
        assert _gates(ctx) == ["regime_alpha"]

    def test_velocity(self):
        ctx = _ctx(spy_returns=[-0.05, -0.04, -0.03],
                   config={"regime_params": {"BULL_CALM": {
                       "spy_velocity_halt_pct": 0.05,
                       "spy_velocity_lookback_days": 3}}})
        VelocityCrashTask().run(ctx)
        assert not ctx.buy_blocked and ctx._gate_block_pending
        assert _gates(ctx) == ["spy_velocity_crash"]

    def test_ema50_both_branches(self):
        ctx = _ctx(ohlcv={})
        EMA50GateTask().run(ctx)
        assert not ctx.buy_blocked and ctx._gate_block_pending
        rows = ctx.gate_registry.ledger_rows(run_id="t")
        assert rows[0]["inputs"]["data_outage"] is True
        closes = pd.Series([100.0 - i for i in range(60)])
        ctx2 = _ctx(ohlcv={"SPY": pd.DataFrame({"close": closes})})
        EMA50GateTask().run(ctx2)
        assert not ctx2.buy_blocked and ctx2._gate_block_pending
        assert ctx2.gate_registry.ledger_rows(run_id="t")[0]["inputs"]["data_outage"] is False

    def test_non_blocking_silent(self):
        ctx = _ctx()
        DrawdownGateTask().run(ctx)
        TransitionWindowTask().run(ctx)
        assert not ctx.buy_blocked
        assert _gates(ctx) == []


class TestLedgerMirror:

    def test_record_round_trip(self, tmp_path):
        ctx = _ctx(skip_buys=True)
        DrawdownGateTask().run(ctx)
        conn = get_connection({"persistence": {
            "enabled": True, "db_path": str(tmp_path / "runs.db")}})
        n = record_gate_verdicts(conn, run_id="r1",
                                 run_date=datetime.date(2026, 6, 12),
                                 registry=ctx.gate_registry)
        assert n == 1
        row = conn.execute("SELECT gate, scope, verdict FROM gate_verdicts").fetchone()
        assert row == ("drawdown_circuit", "book", "block")

    def test_noop_paths(self, tmp_path):
        conn = get_connection({"persistence": {
            "enabled": True, "db_path": str(tmp_path / "runs.db")}})
        d = datetime.date(2026, 6, 12)
        assert record_gate_verdicts(None, run_id="r", run_date=d, registry=None) == 0
        assert record_gate_verdicts(conn, run_id=None, run_date=d, registry=None) == 0
        assert record_gate_verdicts(conn, run_id="r", run_date=d, registry=None) == 0


class TestUmbrellaChokePoints:

    def test_buy_gates_job_applies_flag(self):
        from kernel.pipeline.job_gates import BuyGatesJob

        ctx = _ctx(skip_buys=True)
        BuyGatesJob().run(ctx)
        assert ctx.buy_blocked

    def test_latch_alone_applies_flag(self, monkeypatch):
        # Registry import broken (pin regression simulation): the plain
        # latch still lands the flag at the boundary — gates never die.
        import builtins
        from kernel.pipeline.job_gates import BuyGatesJob

        real = builtins.__import__

        def _broken(name, *a, **kw):
            if "gate_registry" in name:
                raise ImportError("simulated pin regression")
            return real(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _broken)
        ctx = _ctx(skip_buys=True)
        BuyGatesJob().run(ctx)
        assert ctx.buy_blocked, "degrade-safe retirement invariant"
        assert ctx.gate_registry is None
