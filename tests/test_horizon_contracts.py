from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"


def _load(name: str) -> dict:
    return json.loads((STRATEGY_DIR / name).read_text())


def test_prod_rotation_qp_horizons_match_panel_ltr_contract() -> None:
    """AUDIT REGRESSION GUARD: 104's alpha, rotation, and QP μ horizons
    must be one explicit contract, not three quiet knobs.
    """
    for name in ("strategy_config.json", "strategy_config.golden.json"):
        cfg = _load(name)
        panel_horizon = int(cfg["panel_ltr"]["lookahead_days"])
        rotation_horizon = int(cfg["rotation"]["target_horizon_days"])
        qp_horizon = int(cfg["rotation"]["joint_actions"]["qp_mu_horizon_days"])

        assert rotation_horizon == panel_horizon, name
        assert qp_horizon == panel_horizon, name


def test_bull_calm_qp_soft_sell_waits_for_panel_thesis_horizon() -> None:
    """QP soft sells are not hard risk exits; in BULL_CALM they should not
    cut a 60d panel thesis before the thesis has had time to play out.
    """
    for name in ("strategy_config.json", "strategy_config.golden.json"):
        cfg = _load(name)
        panel_horizon = int(cfg["panel_ltr"]["lookahead_days"])
        qp_guard = cfg["rotation"]["joint_actions"]["qp_soft_sell_guard"]
        bull_calm_days = int(
            qp_guard["min_holding_days_by_regime"]["BULL_CALM"]
        )

        assert bull_calm_days >= panel_horizon, name


def test_bull_calm_panel_soft_exits_wait_for_panel_thesis_horizon() -> None:
    """Panel/model soft exits share the same BULL_CALM thesis horizon as QP."""
    for name in ("strategy_config.json", "strategy_config.golden.json"):
        cfg = _load(name)
        panel_horizon = int(cfg["panel_ltr"]["lookahead_days"])
        panel_exit = cfg["risk"]["panel_exit"]
        bull_calm_days = int(
            panel_exit["min_holding_days_by_regime"]["BULL_CALM"]
        )

        assert bull_calm_days >= panel_horizon, name


def test_bull_calm_stop_loss_uses_entry_thesis_floor_after_regime_change() -> None:
    """Prod config should not tighten BULL_CALM entry stops on relabel alone."""
    for name in ("strategy_config.json", "strategy_config.golden.json"):
        cfg = _load(name)
        policy = cfg["risk"]["stop_loss_anchor_policy"]

        assert policy["mode"] == "max_entry_current", name
        assert "BULL_CALM" in policy["entry_regimes"], name
        assert {"BULL_VOLATILE", "CHOPPY", "BEAR"}.issubset(
            set(policy["current_regimes"])
        ), name
