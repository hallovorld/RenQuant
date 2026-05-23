"""Regression guard for CrossSectionalPanelExitTask (BA-style alpha-decay exit).

Bug context (2026-05-11 audit):
  * Old PanelConvictionExitTask used hs.rank_score (calibrator output).
  * Calibrator saturates: any panel ≥ 0.5 → rank = 1.0.
  * Audit DB query: 0/5983 alpaca-live sells were panel_conviction.
  * 12,336 historical bar-instances had rank<0.20 AND mu<0 — STILL 0 fires.
  * Effectively dead code; bearish-panel holdings (BA: panel=+0.198,
    mu=-0.123) never get exited via model signal.

Fix (CrossSectionalPanelExitTask): cross-sectional percentile of TODAY's
raw panel_score, bypassing calibrator. Runs at pipeline level AFTER
PanelScoringJob so cross-section is finalized.
"""
from __future__ import annotations

import datetime
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.exits import ExitSignal, HoldingState  # noqa: E402
from kernel.pipeline.task_panel_conviction_xs import (  # noqa: E402
    CrossSectionalPanelExitTask,
)


def _holding(panel: float | None, mu: float | None, *,
              entry_price=100.0, days_back=20) -> HoldingState:
    return HoldingState(
        entry_price=entry_price,
        entry_date=datetime.date(2025, 6, 15) - datetime.timedelta(days=days_back),
        high_watermark=entry_price * 1.05,
        panel_score=panel,
        mu=mu,
    )


def _cand(ticker: str, panel: float) -> SimpleNamespace:
    return SimpleNamespace(ticker=ticker, panel_score=panel)


def _ctx(*, holdings: dict, candidates: list, cfg_panel_exit: dict | None = None,
         exits=None, regime="BULL_CALM", prices=None, tax=None,
         earnings_calendar=None, max_sells_per_bar: int | None = None):
    cfg = {
        "risk": {},
        "regime": {
            "earnings_sell_buffer_pre_days": 2,
            "earnings_sell_buffer_post_days": 5,
        },
        "lt_hold_gate_days": 330,
        "lt_hold_min_gain": 0.10,
    }
    if cfg_panel_exit is not None:
        cfg["risk"]["panel_exit"] = cfg_panel_exit
    if max_sells_per_bar is not None:
        cfg["risk"]["max_sells_per_bar"] = max_sells_per_bar
    if tax is not None:
        cfg["tax"] = tax
    return SimpleNamespace(
        config=cfg,
        today=datetime.date(2025, 6, 15),
        regime=regime,
        holdings=holdings,
        candidates=candidates,
        exits=exits if exits is not None else [],
        counters={},
        prices=prices or {},
        earnings_calendar=earnings_calendar,
    )


class TestGuards:
    def test_disabled_is_noop(self):
        h = _holding(panel=0.0, mu=-0.5)   # extreme bearish
        ctx = _ctx(
            holdings={"BA": h},
            candidates=[_cand("X1", 0.9), _cand("X2", 0.8)],
            cfg_panel_exit={"enabled": False},
        )
        CrossSectionalPanelExitTask().run(ctx)
        assert ctx.exits == []

    def test_absent_block_is_noop(self):
        h = _holding(panel=0.0, mu=-0.5)
        ctx = _ctx(holdings={"BA": h},
                   candidates=[_cand("X1", 0.9), _cand("X2", 0.8)])
        CrossSectionalPanelExitTask().run(ctx)
        assert ctx.exits == []

    def test_too_few_candidates_noop(self):
        h = _holding(panel=0.0, mu=-0.5)
        ctx = _ctx(holdings={"BA": h}, candidates=[_cand("X1", 0.9)],
                   cfg_panel_exit={"enabled": True})
        CrossSectionalPanelExitTask().run(ctx)
        # Fewer than min_universe (5 default) → skip
        assert ctx.exits == []


class TestFireBA:
    """BA case: panel=+0.198 (bottom of universe), mu=-0.123."""

    def test_ba_fires_via_strong_mu_bypass(self):
        """BA case: panel=+0.198 is ~32%ile (NOT in bottom 20%) but
        mu=-0.123 is strongly negative. The mu_strong_sell_ceiling=-0.05
        OR-bypass fires regardless of percentile. This is the user-
        reported case from the 2026-05-11 audit."""
        # Realistic 41 panel scores from the 2026-05-09 alpaca run
        panels = [-0.037, -0.017, -0.003, 0.019, 0.026, 0.046, 0.047, 0.048,
                  0.048, 0.056, 0.087, 0.103, 0.124, 0.198,  # ← BA at idx 13/41 = 32%ile
                  0.215, 0.300, 0.350, 0.400, 0.450, 0.485,
                  0.495, 0.526, 0.530, 0.546, 0.558, 0.595,
                  0.624, 0.624, 0.647, 0.659, 0.662, 0.670,
                  0.670, 0.674, 0.674, 0.698, 0.702, 0.734,
                  0.735, 0.745, 0.941]
        cands = [_cand(f"X{i:02d}", p) for i, p in enumerate(panels) if p != 0.198]
        ctx = _ctx(
            holdings={"BA": _holding(panel=0.198, mu=-0.123)},
            candidates=cands,
            cfg_panel_exit={
                "enabled": True,
                "xs_panel_percentile_floor": 0.20,
                "mu_sell_ceiling": 0.0,
                "mu_strong_sell_ceiling": -0.05,    # ← strong-mu bypass
            },
        )
        CrossSectionalPanelExitTask().run(ctx)
        assert len(ctx.exits) == 1
        ticker, sig = ctx.exits[0]
        assert ticker == "BA"
        assert sig.should_exit is True
        assert sig.exit_type == "panel_conviction"
        # Reason should mention strong_mu (not xs) since BA isn't in bottom %ile
        assert "strong_mu" in sig.reason
        assert ctx.counters.get("xs_panel_exit") == 1

    def test_xs_bottom_fires_via_and_rule(self):
        """Position in bottom 20% AND mu ≤ 0 fires through AND-rule."""
        # Use 20 candidates 0.10..0.86, holding at panel=0.05 (bottom %ile)
        cands = [_cand(f"X{i}", 0.10 + i * 0.04) for i in range(20)]
        ctx = _ctx(
            holdings={"WEAK": _holding(panel=0.05, mu=-0.02)},  # bottom + slightly negative mu
            candidates=cands,
            cfg_panel_exit={
                "enabled": True,
                "xs_panel_percentile_floor": 0.20,
                "mu_sell_ceiling": 0.0,
                "mu_strong_sell_ceiling": -0.05,
            },
        )
        CrossSectionalPanelExitTask().run(ctx)
        assert len(ctx.exits) == 1
        _, sig = ctx.exits[0]
        # mu=-0.02 > -0.05 so NOT strong; should fire via xs+mu AND-rule
        assert "xs" in sig.reason or "panel_conviction[xs]" in sig.reason


class TestFireGates:
    def test_positive_mu_blocks_fire(self):
        # Same low panel, but mu positive — shouldn't fire
        cands = [_cand(f"X{i}", 0.9 - i * 0.01) for i in range(20)]
        ctx = _ctx(
            holdings={"BA": _holding(panel=-0.1, mu=+0.05)},
            candidates=cands,
            cfg_panel_exit={"enabled": True, "xs_panel_percentile_floor": 0.20,
                            "mu_sell_ceiling": 0.0},
        )
        CrossSectionalPanelExitTask().run(ctx)
        assert ctx.exits == []

    def test_high_panel_blocks_fire(self):
        # Panel at top of universe — shouldn't fire even with mu negative
        cands = [_cand(f"X{i}", 0.1 + i * 0.04) for i in range(20)]  # 0.1..0.86
        ctx = _ctx(
            holdings={"MCD": _holding(panel=0.9, mu=-0.05)},  # top
            candidates=cands,
            cfg_panel_exit={"enabled": True, "xs_panel_percentile_floor": 0.20,
                            "mu_sell_ceiling": 0.0},
        )
        CrossSectionalPanelExitTask().run(ctx)
        assert ctx.exits == []

    def test_none_panel_skipped(self):
        cands = [_cand(f"X{i}", 0.1 + i * 0.04) for i in range(20)]
        ctx = _ctx(
            holdings={"NEW": _holding(panel=None, mu=-0.5)},
            candidates=cands,
            cfg_panel_exit={"enabled": True, "xs_panel_percentile_floor": 0.20,
                            "mu_sell_ceiling": 0.0},
        )
        CrossSectionalPanelExitTask().run(ctx)
        assert ctx.exits == []

    def test_none_mu_skipped(self):
        cands = [_cand(f"X{i}", 0.1 + i * 0.04) for i in range(20)]
        ctx = _ctx(
            holdings={"NEW": _holding(panel=-0.1, mu=None)},
            candidates=cands,
            cfg_panel_exit={"enabled": True, "xs_panel_percentile_floor": 0.20,
                            "mu_sell_ceiling": 0.0},
        )
        CrossSectionalPanelExitTask().run(ctx)
        assert ctx.exits == []

    def test_already_exiting_skipped(self):
        # Higher-priority rule already added exit; xs-task must not duplicate
        cands = [_cand(f"X{i}", 0.1 + i * 0.04) for i in range(20)]
        prior = ExitSignal(should_exit=True, reason="sl", exit_type="stop_loss")
        ctx = _ctx(
            holdings={"BA": _holding(panel=-0.2, mu=-0.5)},
            candidates=cands,
            exits=[("BA", prior)],
            cfg_panel_exit={"enabled": True, "xs_panel_percentile_floor": 0.20,
                            "mu_sell_ceiling": 0.0},
        )
        CrossSectionalPanelExitTask().run(ctx)
        # Only the original stop_loss should be in exits
        assert len(ctx.exits) == 1
        assert ctx.exits[0][1].exit_type == "stop_loss"

    def test_nan_panel_skipped(self):
        cands = [_cand(f"X{i}", 0.1 + i * 0.04) for i in range(20)]
        ctx = _ctx(
            holdings={"BAD": _holding(panel=math.nan, mu=-0.5)},
            candidates=cands,
            cfg_panel_exit={"enabled": True, "xs_panel_percentile_floor": 0.20,
                            "mu_sell_ceiling": 0.0},
        )
        CrossSectionalPanelExitTask().run(ctx)
        assert ctx.exits == []


class TestHorizonAndTaxGates:
    def _bearish_cfg(self, **extras):
        cfg = {
            "enabled": True,
            "xs_panel_percentile_floor": 0.20,
            "mu_sell_ceiling": 0.0,
            "mu_strong_sell_ceiling": -0.05,
            "min_universe": 5,
        }
        cfg.update(extras)
        return cfg

    def test_bull_calm_soft_exit_waits_for_configured_thesis_days(self):
        cands = [_cand(f"X{i}", 0.10 + i * 0.04) for i in range(20)]
        ctx = _ctx(
            holdings={"EARLY": _holding(panel=0.05, mu=-0.20, days_back=3)},
            candidates=cands,
            cfg_panel_exit=self._bearish_cfg(
                min_holding_days_by_regime={"BULL_CALM": 10},
            ),
            prices={"EARLY": 95.0},
            regime="BULL_CALM",
        )
        CrossSectionalPanelExitTask().run(ctx)
        assert ctx.exits == []
        assert ctx.counters.get("xs_panel_exit_horizon_suppressed") == 1

    def test_deteriorated_regime_can_exit_before_bull_calm_thesis_days(self):
        cands = [_cand(f"X{i}", 0.10 + i * 0.04) for i in range(20)]
        ctx = _ctx(
            holdings={"EARLY": _holding(panel=0.05, mu=-0.20, days_back=3)},
            candidates=cands,
            cfg_panel_exit=self._bearish_cfg(
                min_holding_days_by_regime={"BULL_CALM": 10},
            ),
            prices={"EARLY": 95.0},
            regime="CHOPPY",
        )
        CrossSectionalPanelExitTask().run(ctx)
        assert len(ctx.exits) == 1
        assert ctx.exits[0][0] == "EARLY"

    def test_root_level_lt_gate_330_does_not_act_like_30_day_default(self):
        cands = [_cand(f"X{i}", 0.10 + i * 0.04) for i in range(20)]
        ctx = _ctx(
            holdings={"GAIN": _holding(panel=0.05, mu=-0.20, days_back=60)},
            candidates=cands,
            cfg_panel_exit=self._bearish_cfg(),
            prices={"GAIN": 120.0},
        )
        CrossSectionalPanelExitTask().run(ctx)
        assert len(ctx.exits) == 1
        assert ctx.exits[0][0] == "GAIN"

    def test_tax_drag_blocks_marginal_short_term_gain_exit(self):
        cands = [_cand(f"X{i}", 0.10 + i * 0.04) for i in range(20)]
        ctx = _ctx(
            holdings={"TAX": _holding(panel=0.05, mu=-0.02, days_back=30)},
            candidates=cands,
            cfg_panel_exit=self._bearish_cfg(
                tax_adjusted_soft_exit={"enabled": True},
            ),
            prices={"TAX": 120.0},
            tax={"short_term_rate": 0.50, "long_term_rate": 0.32,
                 "long_term_threshold_days": 365},
        )
        CrossSectionalPanelExitTask().run(ctx)
        assert ctx.exits == []
        assert ctx.counters.get("xs_panel_exit_tax_suppressed") == 1

    def test_large_negative_mu_can_pay_short_term_tax_drag(self):
        cands = [_cand(f"X{i}", 0.10 + i * 0.04) for i in range(20)]
        ctx = _ctx(
            holdings={"TAX": _holding(panel=0.05, mu=-0.12, days_back=30)},
            candidates=cands,
            cfg_panel_exit=self._bearish_cfg(
                tax_adjusted_soft_exit={"enabled": True},
            ),
            prices={"TAX": 120.0},
            tax={"short_term_rate": 0.50, "long_term_rate": 0.32,
                 "long_term_threshold_days": 365},
        )
        CrossSectionalPanelExitTask().run(ctx)
        assert len(ctx.exits) == 1
        assert ctx.exits[0][0] == "TAX"

    def test_unrealized_loss_has_no_tax_drag_suppression(self):
        cands = [_cand(f"X{i}", 0.10 + i * 0.04) for i in range(20)]
        ctx = _ctx(
            holdings={"LOSS": _holding(panel=0.05, mu=-0.02, days_back=30)},
            candidates=cands,
            cfg_panel_exit=self._bearish_cfg(
                tax_adjusted_soft_exit={"enabled": True},
            ),
            prices={"LOSS": 90.0},
            tax={"short_term_rate": 0.50, "long_term_rate": 0.32,
                 "long_term_threshold_days": 365},
        )
        CrossSectionalPanelExitTask().run(ctx)
        assert len(ctx.exits) == 1
        assert ctx.exits[0][0] == "LOSS"


class TestPortfolioLevelGuards:
    def _bearish_cfg(self):
        return {
            "enabled": True,
            "xs_panel_percentile_floor": 0.20,
            "mu_sell_ceiling": 0.0,
            "mu_strong_sell_ceiling": -0.05,
            "min_universe": 5,
        }

    def test_xs_panel_exit_respects_earnings_blackout(self):
        cands = [_cand(f"X{i}", 0.10 + i * 0.04) for i in range(20)]
        ctx = _ctx(
            holdings={"CAT": _holding(panel=0.05, mu=-0.20, days_back=60)},
            candidates=cands,
            cfg_panel_exit=self._bearish_cfg(),
            earnings_calendar={"CAT": ["2025-06-14"]},
        )

        CrossSectionalPanelExitTask().run(ctx)

        assert ctx.exits == [], (
            "pipeline-level panel_conviction is model-driven and must obey "
            "the same post-earnings blackout as TickerSellJob exits"
        )

    def test_xs_panel_exit_reapplies_max_sells_per_bar_cap(self):
        cands = [_cand(f"X{i}", 0.10 + i * 0.04) for i in range(20)]
        ctx = _ctx(
            holdings={
                "A": _holding(panel=0.05, mu=-0.20, days_back=60),
                "B": _holding(panel=0.06, mu=-0.10, days_back=60),
            },
            candidates=cands,
            cfg_panel_exit=self._bearish_cfg(),
            max_sells_per_bar=1,
            prices={"A": 95.0, "B": 95.0},
        )

        CrossSectionalPanelExitTask().run(ctx)

        assert [ticker for ticker, _ in ctx.exits] == ["A"], (
            "xs panel exits are soft model exits; after adding them in Phase 3 "
            "the portfolio-level same-bar sell cap must be applied again"
        )
        assert ctx.counters.get("model_sell_throttled") == 1
