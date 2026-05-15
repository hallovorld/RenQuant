"""Production invariant: NO leverage. Σ|w| ≤ 1.0 at all times.

User explicitly did NOT authorize leverage (`max_gross_exposure > 1.0`).
Per `feedback_no_leverage_invariant` memory: the QP must always solve
within Σ|w| ≤ 1.0 regardless of what `long_short.max_gross_exposure`
is set to in any config.

This test pins three invariants:

1. With `long_short.enabled=true`, `_qp_gross_max` is silently clamped
   to 1.0 even when config says 1.30 / 1.50 / 999.
2. With `long_short.enabled=false`, `_qp_gross_max = None` (long-only
   path with implicit Σw ≤ 1.0).
3. Both prod configs (strategy_config.json + .golden.json) declare
   `max_gross_exposure ≤ 1.0`.

To re-enable leverage requires:
  (a) explicit user authorization
  (b) editing `_LEVERAGE_HARDCAP` in `kernel/portfolio_qp/tasks.py`
  (c) updating this test
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _run_init_task(config: dict, regime: str = "BULL_CALM"):
    """Run the QP init step in isolation; return ctx with _qp_gross_max set."""
    from kernel.portfolio_qp.tasks import ComputeQPConstraintsTask  # noqa: PLC0415
    task = ComputeQPConstraintsTask()
    ctx = SimpleNamespace(
        config=config, regime=regime,
        candidates=[SimpleNamespace(ticker=f"T{i}") for i in range(5)],
        holdings={}, confidence=1.0, regime_state=None,
    )
    task.run(ctx)
    return ctx


class TestNoLeverageInvariant:
    """Hardcap pins production to Σ|w| ≤ 1.0 regardless of config."""

    def test_enabled_with_gross_max_1_30_clamps_to_1_0(self):
        """Even if a side config says 1.30, the QP only sees 1.0."""
        cfg = {
            "max_position_pct": 0.20,
            "long_short": {"enabled": True, "max_short_pct": 0.05,
                            "max_gross_exposure": 1.30},
        }
        ctx = _run_init_task(cfg)
        assert ctx._qp_gross_max == 1.0, (
            f"Hardcap broken: cfg said 1.30 but _qp_gross_max={ctx._qp_gross_max}. "
            f"This would silently re-enable leverage."
        )

    def test_enabled_with_gross_max_1_50_clamps_to_1_0(self):
        """Reg-T max (150% gross) is still capped."""
        cfg = {
            "max_position_pct": 0.20,
            "long_short": {"enabled": True, "max_short_pct": 0.05,
                            "max_gross_exposure": 1.50},
        }
        ctx = _run_init_task(cfg)
        assert ctx._qp_gross_max == 1.0

    def test_enabled_with_gross_max_999_clamps_to_1_0(self):
        """Pathological config value still capped — defense in depth."""
        cfg = {
            "max_position_pct": 0.20,
            "long_short": {"enabled": True, "max_short_pct": 0.05,
                            "max_gross_exposure": 999.0},
        }
        ctx = _run_init_task(cfg)
        assert ctx._qp_gross_max == 1.0

    def test_enabled_with_gross_max_0_8_preserved(self):
        """Values BELOW the cap pass through unchanged."""
        cfg = {
            "max_position_pct": 0.20,
            "long_short": {"enabled": True, "max_short_pct": 0.05,
                            "max_gross_exposure": 0.8},
        }
        ctx = _run_init_task(cfg)
        assert ctx._qp_gross_max == 0.8, "below-cap values should NOT be modified"

    def test_disabled_path_gross_max_is_none(self):
        """When shorts off, no gross-cap constraint is added to QP."""
        cfg = {
            "max_position_pct": 0.20,
            "long_short": {"enabled": False, "max_gross_exposure": 1.30},
        }
        ctx = _run_init_task(cfg)
        assert ctx._qp_gross_max is None, (
            "Long-only path: cp.sum(wp) ≤ 1 should bind, no extra gross constraint"
        )

    def test_bear_regime_disables_gross_cap_even_when_enabled(self):
        """BEAR regime: shorts blocked, no leverage."""
        cfg = {
            "max_position_pct": 0.20,
            "long_short": {"enabled": True, "max_short_pct": 0.05,
                            "max_gross_exposure": 1.0},
        }
        ctx = _run_init_task(cfg, regime="BEAR")
        assert ctx._qp_w_lower == 0.0, "BEAR must zero w_lower"
        assert ctx._qp_gross_max is None, "BEAR must drop gross cap entirely"


class TestProdConfigsDeclareNoLeverage:
    """Both shipped configs must declare `max_gross_exposure ≤ 1.0`."""

    @pytest.mark.parametrize("path", [
        REPO_ROOT / "backtesting" / "renquant_104" / "strategy_config.json",
        REPO_ROOT / "backtesting" / "renquant_104" / "strategy_config.golden.json",
    ])
    def test_config_gross_max_at_most_1_0(self, path):
        cfg = json.loads(path.read_text())
        ls = cfg.get("long_short", {})
        assert "max_gross_exposure" in ls, f"{path.name} missing long_short.max_gross_exposure"
        assert ls["max_gross_exposure"] <= 1.0, (
            f"{path.name}: max_gross_exposure={ls['max_gross_exposure']} > 1.0. "
            f"This would re-enable leverage. User has NOT authorized leverage."
        )
