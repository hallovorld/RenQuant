"""Tests for the QP min_share_floor guard.

Pre-fix: any candidate whose share price exceeded the QP's dollar budget
(target_w × NAV) had `_shares_from_dw` return 0 → silently dropped at the
`if shares <= 0: continue` gate. For a $10k account this blocks EQIX
($1059/share), BKNG ($5k), NVR ($8k), etc. entirely — biasing the
strategy toward low-price names.

2026-05-24 safety fix: the old default rounded sub-1-share QP targets up
to 1 share when one-share weight was within [floor, ceiling]. That violates
the optimizer target/cap contract. Default is now disabled; explicit
experimental enablement still retains the BUY-only and ceiling guards.

Invariant: this only fires on BUY intent (dw > 0) and never on SELL.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

TASKS_PATH = REPO_ROOT / "backtesting/renquant_104/kernel/portfolio_qp/tasks.py"
TASKS_SRC = TASKS_PATH.read_text()


class TestQpMinShareFloor:
    """Integer execution must not overrule QP targets by default."""

    def test_fix_tag_present(self):
        assert "QP_MIN_SHARE_FLOOR" in TASKS_SRC
        assert "min_share_floor for high-price stocks (EQIX/META class)" in TASKS_SRC

    def test_env_carries_floor_and_ceiling(self):
        assert "min_share_floor_pct=" in TASKS_SRC
        assert "min_share_ceiling_pct=" in TASKS_SRC
        assert 'cfg.get("qp_min_share_floor_pct"' in TASKS_SRC
        assert 'cfg.get("qp_min_share_ceiling_pct"' in TASKS_SRC

    def test_only_fires_on_buy_intent(self):
        """dw > 0 guard prevents this from firing on accidental sell paths."""
        assert "shares <= 0 and dw > 0" in TASKS_SRC, \
            "min_share_floor must only fire on positive dw (BUY intent)"

    def test_ceiling_caps_overallocation(self):
        snippet_start = TASKS_SRC.index("QP_MIN_SHARE_FLOOR")
        nearby = TASKS_SRC[snippet_start - 1000: snippet_start + 600]
        # Ceiling check appears
        assert "one_share_pct <= ceiling" in nearby, \
            "ceiling must prevent buying 1 share when it'd exceed max_position cap"
        assert "floor <= one_share_pct" in nearby

    def test_default_is_disabled(self):
        """Integer execution must not round a sub-1-share QP target upward."""
        assert '0.0' in TASKS_SRC.split("qp_min_share_floor_pct")[1][:200], \
            "default floor should be disabled"
        assert '0.15' in TASKS_SRC.split("qp_min_share_ceiling_pct")[1][:200], \
            "legacy experiment ceiling should remain explicit"

    def test_disable_via_floor_zero(self):
        """Setting floor=0 must disable the feature (regression guard)."""
        snippet_start = TASKS_SRC.index("QP_MIN_SHARE_FLOOR")
        nearby = TASKS_SRC[snippet_start - 600: snippet_start + 200]
        assert "if floor > 0" in nearby, \
            "floor=0 must skip the min-share path (disable knob)"

    def test_disabled_default_rounds_down_sub_share_qp_target(self):
        """Do not manufacture a 1-share buy beyond the solver delta."""
        from kernel.portfolio_qp.tasks import EmitOrdersFromQPSolutionTask

        cand = SimpleNamespace(
            ticker="EQIX",
            rank_score=0.80,
            panel_score=0.20,
            mu=0.04,
            sigma=0.20,
        )
        ctx = SimpleNamespace(
            config={},
            orders=[],
            exits=[],
            holdings={},
            candidates=[cand],
            counters={},
            _blocked_by_ticker={},
        )
        env = {
            "cfg": {},
            "sol": SimpleNamespace(
                status="optimal",
                delta_w=np.array([0.04]),
                target_w=np.array([0.04]),
            ),
            "tickers": ["EQIX"],
            "prices": {"EQIX": 1059.0},
            "nav": 10_000.0,
            "cash": 10_000.0,
            "cash_actual": 10_000.0,
            "cash_reserve": 0.0,
            "buy_cost_multiplier": 1.0,
            "min_dw": 0.02,
            "no_trade_factor": 0.0,
            "band_cap": 0.05,
            "sigma_vec": np.array([0.20]),
            "cands": {"EQIX": cand},
            "score_sources": {"EQIX": cand},
            "buy_blocked": False,
            "buys_gated": False,
            "earnings_cal": {},
            "earn_buf": 0,
            "today": None,
            "holdings_set": set(),
            "holdings": {},
            "max_positions": 8,
            "min_share_floor_pct": 0.0,
            "min_share_ceiling_pct": 0.15,
            "defensive_set": set(),
            "bear_only": False,
            "preexisting_exit_tickers": set(),
            "emitted_new_tickers": set(),
        }

        nb, ns, counters = EmitOrdersFromQPSolutionTask._emit_orders_loop(ctx, env)

        assert (nb, ns) == (0, 0)
        assert counters["zero_shares"] == 1
        assert ctx.orders == []
        assert ctx._blocked_by_ticker["EQIX"] == "qp_zero_shares"


class TestDeepDrawdownVetoDisabled:
    """2026-05-17: DDV disabled globally per HXZ 2020 (RFS) Replicating
    Anomalies finding that distress/loser anomaly fails to replicate
    in modern data. Config retained for regime-conditional re-enable."""

    def test_disabled_in_golden(self):
        import json
        cfg_path = REPO_ROOT / "backtesting/renquant_104/strategy_config.golden.json"
        c = json.loads(cfg_path.read_text())
        ddv = c["ranking"]["buy_quality_gates"]["deep_drawdown_veto"]
        assert ddv["enabled"] is False, "DDV should be disabled in golden"
        assert "_disable_reason_2026-05-17" in ddv, \
            "disable reason must be documented inline for future audit"
        assert "Hou-Xue-Zhang 2020" in ddv["_disable_reason_2026-05-17"]

    def test_disabled_in_live_config(self):
        import json
        cfg_path = REPO_ROOT / "backtesting/renquant_104/strategy_config.json"
        c = json.loads(cfg_path.read_text())
        ddv = c["ranking"]["buy_quality_gates"]["deep_drawdown_veto"]
        assert ddv["enabled"] is False
