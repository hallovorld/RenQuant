"""Tests for LimitSellsPerBarTask (audit fix MAX-SELLS-PER-BAR, 2026-04-26 round-7).

User spec: "把我有的股票全卖了？这他妈的合理吗？"
Pre-fix, a single bar could exit 3-of-6 holdings simultaneously when
multiple per-ticker models all spiked sell signals on the same day.
Per-ticker rules can't see the portfolio-level effect.

Behavioral contract:
  * Default OFF (max_sells_per_bar=0 → uncapped).
  * Risk exits (stop_loss/trailing/SDL/max_hold/panel_conviction/
    rotation/kelly_trim/joint_sell) are EXEMPT — they always fire.
  * model_sell exits sorted by NGBoost μ ascending (most-bearish first),
    keep top N, drop the rest.
  * Diagnostic: dropped exits stored in ctx.exits_throttled +
    counters["model_sell_throttled"] incremented.
  * Defensive: missing μ → treated as +inf (least urgent → first to drop).
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.exits import ExitSignal, HoldingState   # noqa: E402
from kernel.pipeline.context import InferenceContext   # noqa: E402
from kernel.pipeline.task_limit_sells import LimitSellsPerBarTask  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _holding(*, mu: float | None) -> HoldingState:
    today = datetime.date(2026, 4, 27)
    h = HoldingState(
        entry_price=100.0,
        entry_date=today - datetime.timedelta(days=60),
        high_watermark=110.0,
        sell_streak=3,
        last_streak_inc_date=today,
        shares=10.0,
    )
    h.mu    = mu
    h.sigma = 0.05
    return h


def _exit(exit_type: str = "model_sell", reason: str = "") -> ExitSignal:
    return ExitSignal(
        should_exit=True,
        reason=reason or f"{exit_type} test",
        exit_type=exit_type,
    )


def _ctx(*, exits: list[tuple[str, ExitSignal]],
          holdings: dict[str, HoldingState],
          max_sells_per_bar: int) -> InferenceContext:
    return InferenceContext(
        config={"risk": {"max_sells_per_bar": max_sells_per_bar}},
        today=datetime.date(2026, 4, 27),
        holdings=holdings,
        exits=list(exits),
        counters={},
    )


# ── Default OFF ───────────────────────────────────────────────────────────────

class TestDefaultOff:
    def test_zero_means_uncapped(self):
        """max_sells_per_bar=0 → no-op even with 100 model_sells."""
        exits = [(f"T{i}", _exit("model_sell")) for i in range(100)]
        holdings = {f"T{i}": _holding(mu=0.05) for i in range(100)}
        ctx = _ctx(exits=exits, holdings=holdings, max_sells_per_bar=0)

        LimitSellsPerBarTask().run(ctx)
        assert len(ctx.exits) == 100, "0 = uncapped"
        assert "model_sell_throttled" not in ctx.counters

    def test_no_exits_does_not_crash(self):
        ctx = _ctx(exits=[], holdings={}, max_sells_per_bar=2)
        LimitSellsPerBarTask().run(ctx)
        assert ctx.exits == []

    def test_missing_risk_block_does_not_crash(self):
        """No `risk` key at all in config."""
        ctx = InferenceContext(
            config={},
            today=datetime.date(2026, 4, 27),
            holdings={},
            exits=[],
            counters={},
        )
        LimitSellsPerBarTask().run(ctx)   # no-op, no crash


# ── Risk exits exempt ─────────────────────────────────────────────────────────

class TestRiskExitsExempt:
    """Path-dependent rules MUST always fire — never throttled."""

    @pytest.mark.parametrize("risk_type", [
        "stop_loss", "trailing_stop", "single_day_loss", "max_hold",
        "panel_conviction", "rotation", "kelly_trim", "joint_sell",
    ])
    def test_risk_exits_pass_unchanged(self, risk_type):
        """5 risk exits + cap=2 → all 5 still pass (exempt)."""
        exits = [(f"T{i}", _exit(risk_type)) for i in range(5)]
        holdings = {f"T{i}": _holding(mu=0.05) for i in range(5)}
        ctx = _ctx(exits=exits, holdings=holdings, max_sells_per_bar=2)

        LimitSellsPerBarTask().run(ctx)
        assert len(ctx.exits) == 5

    def test_mixed_risk_and_model_sells_only_caps_model_sells(self):
        """3 stop_loss + 4 model_sell + cap=2 → 3 stop + 2 model = 5 total."""
        exits = [
            ("STOP1", _exit("stop_loss")),
            ("STOP2", _exit("stop_loss")),
            ("STOP3", _exit("stop_loss")),
            ("MS1",   _exit("model_sell")),
            ("MS2",   _exit("model_sell")),
            ("MS3",   _exit("model_sell")),
            ("MS4",   _exit("model_sell")),
        ]
        holdings = {
            "STOP1": _holding(mu=0.05), "STOP2": _holding(mu=0.05),
            "STOP3": _holding(mu=0.05),
            "MS1":   _holding(mu=-0.10),  # most bearish
            "MS2":   _holding(mu=-0.05),
            "MS3":   _holding(mu= 0.00),
            "MS4":   _holding(mu= 0.05),  # least bearish
        }
        ctx = _ctx(exits=exits, holdings=holdings, max_sells_per_bar=2)
        LimitSellsPerBarTask().run(ctx)

        kept_tickers = sorted(t for t, _ in ctx.exits)
        # All 3 stop_loss kept + 2 most-bearish model_sells (MS1, MS2)
        assert "STOP1" in kept_tickers
        assert "STOP2" in kept_tickers
        assert "STOP3" in kept_tickers
        assert "MS1"   in kept_tickers
        assert "MS2"   in kept_tickers
        # MS3, MS4 dropped
        assert "MS3" not in kept_tickers
        assert "MS4" not in kept_tickers


# ── Model_sell ordering by μ ──────────────────────────────────────────────────

class TestModelSellOrdering:
    def test_keeps_most_bearish_mu(self):
        """6 model_sells (μ from -0.20 to +0.30), cap=3 → keep 3 most-bearish."""
        exits    = [(f"T{i}", _exit("model_sell")) for i in range(6)]
        # μ values: T0=-0.20 (most bearish), T1=-0.10, T2=0.00, T3=+0.10,
        #          T4=+0.20, T5=+0.30 (least bearish)
        mus = [-0.20, -0.10, 0.00, 0.10, 0.20, 0.30]
        holdings = {f"T{i}": _holding(mu=mus[i]) for i in range(6)}
        ctx = _ctx(exits=exits, holdings=holdings, max_sells_per_bar=3)

        LimitSellsPerBarTask().run(ctx)
        kept = sorted(t for t, _ in ctx.exits)
        assert kept == ["T0", "T1", "T2"], "keep 3 most-bearish μ"

    def test_under_cap_no_throttle(self):
        """3 model_sells, cap=5 → all 3 pass."""
        exits = [("A", _exit("model_sell")),
                 ("B", _exit("model_sell")),
                 ("C", _exit("model_sell"))]
        holdings = {"A": _holding(mu=-0.1), "B": _holding(mu=-0.05),
                    "C": _holding(mu= 0.0)}
        ctx = _ctx(exits=exits, holdings=holdings, max_sells_per_bar=5)

        LimitSellsPerBarTask().run(ctx)
        assert len(ctx.exits) == 3
        assert ctx.counters.get("model_sell_throttled", 0) == 0

    def test_at_cap_no_throttle(self):
        """N == cap → no-op (boundary)."""
        exits = [("A", _exit("model_sell")), ("B", _exit("model_sell"))]
        holdings = {"A": _holding(mu=-0.1), "B": _holding(mu=0.05)}
        ctx = _ctx(exits=exits, holdings=holdings, max_sells_per_bar=2)

        LimitSellsPerBarTask().run(ctx)
        assert len(ctx.exits) == 2


# ── Defensive: missing/NaN μ ──────────────────────────────────────────────────

class TestDefensiveMu:
    def test_missing_mu_treated_as_least_urgent(self):
        """μ=None → +inf in sort key → first to drop."""
        exits = [("HAS_MU",     _exit("model_sell")),
                 ("MISSING_MU", _exit("model_sell"))]
        holdings = {"HAS_MU":     _holding(mu=-0.05),
                    "MISSING_MU": _holding(mu=None)}
        ctx = _ctx(exits=exits, holdings=holdings, max_sells_per_bar=1)

        LimitSellsPerBarTask().run(ctx)
        kept = sorted(t for t, _ in ctx.exits)
        assert kept == ["HAS_MU"], "missing μ dropped first"

    def test_nan_mu_treated_as_least_urgent(self):
        exits = [("FINITE", _exit("model_sell")),
                 ("NAN",    _exit("model_sell"))]
        holdings = {"FINITE": _holding(mu=0.0),
                    "NAN":    _holding(mu=float("nan"))}
        ctx = _ctx(exits=exits, holdings=holdings, max_sells_per_bar=1)

        LimitSellsPerBarTask().run(ctx)
        kept = sorted(t for t, _ in ctx.exits)
        assert kept == ["FINITE"]

    def test_held_missing_from_holdings_does_not_crash(self):
        """exits has a ticker not in holdings — defensive: drop first."""
        exits = [("REAL", _exit("model_sell")),
                 ("GHOST", _exit("model_sell"))]
        holdings = {"REAL": _holding(mu=-0.05)}   # GHOST missing
        ctx = _ctx(exits=exits, holdings=holdings, max_sells_per_bar=1)

        LimitSellsPerBarTask().run(ctx)
        kept = sorted(t for t, _ in ctx.exits)
        assert "REAL" in kept


# ── Diagnostic surface ────────────────────────────────────────────────────────

class TestDiagnostic:
    def test_throttled_exits_recorded(self):
        exits = [(f"T{i}", _exit("model_sell")) for i in range(4)]
        mus = [-0.10, -0.05, 0.00, 0.10]
        holdings = {f"T{i}": _holding(mu=mus[i]) for i in range(4)}
        ctx = _ctx(exits=exits, holdings=holdings, max_sells_per_bar=2)

        LimitSellsPerBarTask().run(ctx)
        assert hasattr(ctx, "exits_throttled")
        assert len(ctx.exits_throttled) == 2
        # T2 (μ=0.00), T3 (μ=0.10) dropped
        dropped_tickers = sorted(d["ticker"] for d in ctx.exits_throttled)
        assert dropped_tickers == ["T2", "T3"]

    def test_counter_incremented(self):
        exits = [(f"T{i}", _exit("model_sell")) for i in range(5)]
        holdings = {f"T{i}": _holding(mu=0.0) for i in range(5)}
        ctx = _ctx(exits=exits, holdings=holdings, max_sells_per_bar=2)

        LimitSellsPerBarTask().run(ctx)
        assert ctx.counters["model_sell_throttled"] == 3

    def test_throttled_records_have_diagnostic_fields(self):
        exits = [("KEEP", _exit("model_sell", reason="streak=3")),
                 ("DROP", _exit("model_sell", reason="streak=4"))]
        holdings = {"KEEP": _holding(mu=-0.10),
                    "DROP": _holding(mu= 0.05)}
        ctx = _ctx(exits=exits, holdings=holdings, max_sells_per_bar=1)

        LimitSellsPerBarTask().run(ctx)
        rec = ctx.exits_throttled[0]
        assert rec["ticker"]    == "DROP"
        assert rec["exit_type"] == "model_sell"
        assert rec["reason"]    == "streak=4"
        assert rec["mu"]        == pytest.approx(0.05)
        assert rec["cap"]       == 1
        assert rec["n_total"]   == 2
