"""GateRegistry writer migration #3 — umbrella job_panel_scoring dual-write.

Design: renquant-orchestrator
doc/research/2026-06-12-engineering-architecture-deep-plan.md S2-PR4;
mirrors pipeline migrations #1/#2 (renquant-pipeline #121 + panel_scoring
slice). The three umbrella fail-closed helpers now dual-write a block
verdict via _submit_gate_verdict, which lazy-imports the registry from
renquant-pipeline and degrades LOUDLY (warning, no ledger row, trading
unaffected) when the sibling checkout predates kernel.gate_registry —
telemetry must never block trading during a merged-not-deployed window.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.panel_pipeline.job_panel_scoring import (  # noqa: E402
    _fail_closed_missing_calibrator,
    _fail_closed_ngboost,
    _fail_closed_panel_scoring,
)


def _ctx(**kw) -> SimpleNamespace:
    base = dict(
        candidates=[], buy_blocked=False, skip_buys=False,
        counters={}, gate_registry=None, config={},
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _rows(ctx):
    if getattr(ctx, "gate_registry", None) is None:
        return []
    return ctx.gate_registry.ledger_rows(run_id="t")


class TestDualWrite:

    def test_panel_scoring_fail_closed(self):
        ctx = _ctx()
        _fail_closed_panel_scoring(ctx, "panel_scorer_load_failed")
        assert ctx.buy_blocked and ctx.skip_buys
        rows = _rows(ctx)
        assert rows and rows[0]["gate"] == "panel_scoring_fail_closed"
        assert rows[0]["reason"] == "panel_scorer_load_failed"

    def test_calibrator_fail_closed(self):
        ctx = _ctx()
        _fail_closed_missing_calibrator(ctx, "calibrator_missing")
        assert ctx.buy_blocked
        rows = _rows(ctx)
        assert rows and rows[0]["gate"] == "calibrator_fail_closed"

    def test_ngboost_fail_closed_carries_detail(self):
        ctx = _ctx()
        _fail_closed_ngboost(ctx, "ngb_artifact_unreadable", detail="bad json")
        assert ctx.buy_blocked
        rows = _rows(ctx)
        assert rows and rows[0]["gate"] == "ngboost_fail_closed"
        assert rows[0]["inputs"]["detail"] == "bad json"

    def test_registry_equivalence(self):
        ctx = _ctx()
        _fail_closed_panel_scoring(ctx, "r")
        assert ctx.gate_registry.blocked("book") == ctx.buy_blocked


class TestStaleSiblingDegradesLoudly:
    """Missing gate_registry module: warning + no row, trading unaffected."""

    def test_import_failure_does_not_raise(self, monkeypatch, caplog):
        import builtins
        import logging

        real_import = builtins.__import__

        def _no_registry(name, *a, **kw):
            if "gate_registry" in name:
                raise ImportError("simulated stale sibling checkout")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _no_registry)
        ctx = _ctx()
        with caplog.at_level(logging.WARNING):
            _fail_closed_panel_scoring(ctx, "r")  # must not raise
        assert ctx.buy_blocked, "direct write must survive telemetry outage"
        assert ctx.gate_registry is None
        assert any("gate_registry unavailable" in r.message
                   for r in caplog.records)
