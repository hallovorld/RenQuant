"""TDD: optional regime-conditional alpha gate for buys in regimes where
PROD's truly-OOS top-10 alpha is non-positive.

Source data: artifacts/prod/truly_oos_eval/eval_truly_oos.json
  BEAR:          IC +0.345  top10_α +0.696  → KEEP buys
  CHOPPY:        IC +0.103  top10_α +0.259  → KEEP buys
  BULL_VOLATILE: IC +0.105  top10_α +0.129  → KEEP buys
  BULL_STRONG:   IC +0.060  top10_α +0.245  → KEEP buys
  BULL_CALM:     IC +0.005  top10_α -0.045  → optional risk gate

Production operator override on 2026-05-21 keeps BULL_CALM buys enabled.
The gate stays per-regime and testable for future graduated-risk work.

Implementation surface: a new gate task in
backtesting/renquant_104/kernel/pipeline/task_gates.py that consults
`ctx.config["regime_params"][ctx.regime].get("disable_new_buys", False)`.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))


def _make_ctx(regime: str, disable_new_buys: bool):
    """Minimal context for gate testing."""
    from kernel.pipeline.context import InferenceContext  # noqa: PLC0415
    cfg = {
        "regime_params": {
            regime: {"disable_new_buys": disable_new_buys},
        },
        "regime": {},
    }
    import datetime as _dt  # noqa: PLC0415
    ctx = InferenceContext(config=cfg, today=_dt.date(2026, 5, 20))
    ctx.regime = regime
    ctx.confidence = 0.58
    ctx.skip_buys = False
    ctx.buy_blocked = False
    ctx.bear_only = False
    ctx.counters = {}
    return ctx


class TestRegimeAlphaGate:
    """Per-regime alpha gate — block new buys in regimes where the prod
    model has no top-10 OOS alpha."""

    def test_bull_calm_with_flag_blocks_new_buys(self):
        from kernel.pipeline.task_gates import RegimeAlphaGateTask  # noqa: PLC0415
        ctx = _make_ctx(regime="BULL_CALM", disable_new_buys=True)
        result = RegimeAlphaGateTask().run(ctx)
        assert result is False, (
            "Gate must return False (short-circuit) when blocking buys"
        )
        assert ctx.buy_blocked is True, (
            "Gate must set ctx.buy_blocked=True for downstream consumers "
            "(QP / TopUp / JointActions) — they check this flag"
        )
        assert ctx.counters.get("regime_alpha_blocks", 0) >= 1, (
            "Gate must increment counters['regime_alpha_blocks'] for ntfy + audit"
        )

    def test_bull_calm_without_flag_does_not_block(self):
        from kernel.pipeline.task_gates import RegimeAlphaGateTask  # noqa: PLC0415
        ctx = _make_ctx(regime="BULL_CALM", disable_new_buys=False)
        result = RegimeAlphaGateTask().run(ctx)
        assert result is not False, "Default (flag off) must NOT block"
        assert ctx.buy_blocked is False
        assert ctx.counters.get("regime_alpha_blocks", 0) == 0

    def test_bear_regime_never_blocked_by_this_gate(self):
        """BEAR has strong OOS skill (+0.345 IC, +0.696 alpha) — even if
        misconfigured with disable_new_buys=True, BEAR is a known-good
        regime. Sanity test: gate should ONLY fire for the configured
        regime, not all regimes blanket."""
        from kernel.pipeline.task_gates import RegimeAlphaGateTask  # noqa: PLC0415
        ctx = _make_ctx(regime="BEAR", disable_new_buys=False)  # BEAR not flagged
        # Pollute another regime's config to verify the gate only reads
        # ctx.regime's own params, not someone else's.
        ctx.config["regime_params"]["BULL_CALM"] = {"disable_new_buys": True}
        result = RegimeAlphaGateTask().run(ctx)
        assert result is not False
        assert ctx.buy_blocked is False

    def test_missing_regime_params_does_not_block(self):
        """If regime_params for ctx.regime is absent (e.g. legacy config),
        gate is a no-op — never block silently."""
        from kernel.pipeline.task_gates import RegimeAlphaGateTask  # noqa: PLC0415
        ctx = _make_ctx(regime="UNKNOWN_NEW_REGIME", disable_new_buys=False)
        ctx.config.pop("regime_params")
        result = RegimeAlphaGateTask().run(ctx)
        assert result is not False
        assert ctx.buy_blocked is False

    def test_gate_registered_in_buy_gates_job(self):
        """Pin: gate is included in the BuyGatesJob task chain in canonical
        order. Without this, the gate is dead code — task exists but pipeline
        doesn't run it (§5.13.2)."""
        from kernel.pipeline.job_gates import BuyGatesJob  # noqa: PLC0415
        from kernel.pipeline.task_gates import RegimeAlphaGateTask  # noqa: PLC0415
        tasks = BuyGatesJob().tasks
        task_classes = [type(t).__name__ for t in tasks]
        assert "RegimeAlphaGateTask" in task_classes, (
            f"RegimeAlphaGateTask must be in BuyGatesJob.tasks; got: {task_classes}"
        )


class TestRegimeAlphaGateConfigWired:
    """The golden config follows the operator override for BULL_CALM.

    The truly-OOS warning is still documented, but the normal-risk
    accumulation regime is no longer hard-disabled.
    """

    def test_golden_bull_calm_disable_new_buys_false_operator_override(self):
        import json  # noqa: PLC0415
        g = json.loads((REPO / "backtesting/renquant_104/strategy_config.golden.json").read_text())
        bc = g.get("regime_params", {}).get("BULL_CALM", {})
        assert bc.get("disable_new_buys") is False, (
            f"BULL_CALM.disable_new_buys must be False after the "
            f"2026-05-21 operator override (currently "
            f"{bc.get('disable_new_buys')}). Keep downstream buy "
            f"constraints active instead of hard-disabling this regime."
        )

    def test_other_regimes_disable_new_buys_not_set(self):
        """BEAR / CHOPPY / BULL_VOL / BULL_STRONG all have positive OOS
        top10 alpha — they should NOT have disable_new_buys=True. Guard
        against a copy-paste blast-radius bug."""
        import json  # noqa: PLC0415
        g = json.loads((REPO / "backtesting/renquant_104/strategy_config.golden.json").read_text())
        for r in ("BEAR", "CHOPPY", "BULL_VOLATILE", "BULL_STRONG"):
            v = g.get("regime_params", {}).get(r, {}).get("disable_new_buys")
            assert v is not True, (
                f"{r} has disable_new_buys=True but its truly-OOS top10 "
                f"alpha is POSITIVE; do not block buys in this regime."
            )
