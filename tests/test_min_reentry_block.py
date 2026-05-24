"""Regression test for the 2026-05-18 MCD-rebuy incident.

Background: MCD sold +$14.02 gain on 2026-05-17. Next day (2026-05-18)
calibrator saturation caused tie-broken QP to re-pick MCD at $5/share
HIGHER, eating the $14 gain plus another $5/share = -$15 round-trip
on the rebuy. Wash-sale (§1091) doesn't apply to gains, so the existing
guard correctly let it through. The behavioral gap: model has no memory
of "just sold this — don't immediately re-pick without significant new
information."

Fix: kernel/portfolio_qp/tasks.py::ComputeWashSaleMaskTask now compounds
two blocks:
  1) §1091 wash-sale (loss-only)
  2) min_reentry_days (any sell, regardless of P/L sign) ← NEW

Default min_reentry_days = 5 (≈ 1 trading week). Operator can override
via strategy_config.json::min_reentry_days.
"""
from __future__ import annotations
import datetime
from pathlib import Path
import json
import pytest

REPO = Path(__file__).resolve().parent.parent


class TestMinReentryConfigStamped:
    """Pin the config-level default — without this, a future config wipe
    would silently revert the protection."""

    def test_golden_has_min_reentry_days(self):
        cfg = json.loads((REPO / "backtesting/renquant_104"
                          / "strategy_config.golden.json").read_text())
        assert "min_reentry_days" in cfg
        assert cfg["min_reentry_days"] >= 1, \
            "min_reentry_days must be ≥1 to provide ANY anti-churn"

    def test_live_matches_golden(self):
        # Pre-commit hook also catches this but assert here for clarity
        golden = json.loads((REPO / "backtesting/renquant_104"
                              / "strategy_config.golden.json").read_text())
        live = json.loads((REPO / "backtesting/renquant_104"
                            / "strategy_config.json").read_text())
        assert golden["min_reentry_days"] == live["min_reentry_days"]

    def test_provenance_documented(self):
        cfg = json.loads((REPO / "backtesting/renquant_104"
                          / "strategy_config.golden.json").read_text())
        assert "_min_reentry_days_provenance" in cfg


class TestComputeWashSaleMaskAntiChurn:
    """Unit tests for ComputeWashSaleMaskTask with min_reentry_days."""

    def _build_ctx(self, **overrides):
        # Minimal ctx mimicking real one's relevant fields
        class Ctx:
            pass
        ctx = Ctx()
        ctx.config = overrides.get("config", {})
        ctx.today = overrides.get("today", datetime.date(2026, 5, 18))
        ctx._qp_tickers = overrides.get("tickers", ["MCD", "AAPL", "GOOG"])
        ctx.last_sell_dates = overrides.get("last_sell_dates", {})
        ctx.last_sell_pls = overrides.get("last_sell_pls", {})
        return ctx

    def _run_task(self, ctx):
        # Import inside test so import errors don't break collection
        import sys
        sys.path.insert(0, str(REPO / "backtesting/renquant_104"))
        from kernel.portfolio_qp.tasks import ComputeWashSaleMaskTask
        ComputeWashSaleMaskTask().run(ctx)
        return ctx._qp_wash_mask

    def test_gain_sale_yesterday_blocked_by_anti_churn(self):
        """The MCD-2026-05-18 scenario: sold yesterday at gain, today blocked."""
        ctx = self._build_ctx(
            config={"min_reentry_days": 5, "wash_sale_days": 30},
            today=datetime.date(2026, 5, 18),
            last_sell_dates={"MCD": datetime.date(2026, 5, 17)},
            last_sell_pls={"MCD": +14.02},  # gain (positive P/L)
        )
        mask = self._run_task(ctx)
        assert mask[0] == True, "MCD must be blocked by anti-churn (sold yesterday)"
        assert mask[1] == False  # AAPL untouched
        assert mask[2] == False  # GOOG untouched

    def test_gain_sale_outside_window_not_blocked(self):
        ctx = self._build_ctx(
            config={"min_reentry_days": 5, "wash_sale_days": 30},
            today=datetime.date(2026, 5, 18),
            last_sell_dates={"MCD": datetime.date(2026, 5, 8)},  # 10d ago
            last_sell_pls={"MCD": +14.02},
        )
        mask = self._run_task(ctx)
        assert mask[0] == False, "Sold 10d ago, outside 5d reentry → not blocked"

    def test_loss_sale_still_blocked_by_wash_sale_within_30d(self):
        """Loss sale 10d ago is past 5d reentry but inside 30d wash-sale."""
        ctx = self._build_ctx(
            config={"min_reentry_days": 5, "wash_sale_days": 30},
            today=datetime.date(2026, 5, 18),
            last_sell_dates={"MCD": datetime.date(2026, 5, 8)},  # 10d ago
            last_sell_pls={"MCD": -50.0},  # loss
        )
        mask = self._run_task(ctx)
        assert mask[0] == True, "Loss 10d ago must still be blocked by §1091"

    def test_zero_min_reentry_disables_anti_churn(self):
        """Backwards-compatible: min_reentry_days=0 behaves as before."""
        ctx = self._build_ctx(
            config={"min_reentry_days": 0, "wash_sale_days": 30},
            today=datetime.date(2026, 5, 18),
            last_sell_dates={"MCD": datetime.date(2026, 5, 17)},
            last_sell_pls={"MCD": +14.02},  # gain
        )
        mask = self._run_task(ctx)
        assert mask[0] == False, "min_reentry=0 → gain sale passes (legacy behavior)"

    def test_string_date_gain_sale_blocked_by_anti_churn(self):
        """live_state persists last_sell_dates as ISO strings; anti-churn
        must parse the same shape as the wash-sale helper."""
        ctx = self._build_ctx(
            config={"min_reentry_days": 5, "wash_sale_days": 0},
            today=datetime.date(2026, 5, 18),
            last_sell_dates={"MCD": "2026-05-17"},
            last_sell_pls={"MCD": +14.02},
        )
        mask = self._run_task(ctx)
        assert mask[0] == True, "ISO string sell date must still trigger min_reentry"

    def test_both_guards_compound(self):
        """If both wash-sale AND anti-churn fire (loss sale yesterday),
        block once (no double-counting)."""
        ctx = self._build_ctx(
            config={"min_reentry_days": 5, "wash_sale_days": 30},
            today=datetime.date(2026, 5, 18),
            last_sell_dates={"MCD": datetime.date(2026, 5, 17)},
            last_sell_pls={"MCD": -50.0},  # loss yesterday
        )
        mask = self._run_task(ctx)
        assert mask[0] == True
        # Mask is boolean — single value regardless of how many guards fired


class TestSourceCodeContract:
    """Pin the source-code shape so future refactors don't silently drop
    the second guard."""

    def test_kernel_reads_min_reentry_days_config(self):
        src = (REPO / "backtesting/renquant_104/kernel/portfolio_qp"
               / "tasks.py").read_text()
        assert "min_reentry_days" in src

    def test_kernel_logs_churn_separately(self):
        src = (REPO / "backtesting/renquant_104/kernel/portfolio_qp"
               / "tasks.py").read_text()
        assert "n_churn" in src or "churn" in src.lower()

    def test_2026_05_18_marker(self):
        src = (REPO / "backtesting/renquant_104/kernel/portfolio_qp"
               / "tasks.py").read_text()
        assert "2026-05-18 ANTI-CHURN" in src
