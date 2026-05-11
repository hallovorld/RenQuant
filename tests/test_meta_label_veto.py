"""TDD — MetaLabelVetoTask: veto false-positive path-rule exits.

Pipeline placement: pp_inference.py post-Phase-2a, AFTER the parallel
TickerSellJob has populated ctx.exits, BEFORE the buy phase. Same
position as DrawdownFlattenTask (S-2 mechanism).

Contract:
  * Reads ctx.config.ranking.meta_label.{enabled, threshold}.
  * Reads ctx._meta_label_predictor: Callable[[features_dict], float]
    where the return is P(exit_is_profitable) ∈ [0, 1]. The predictor
    is loaded by the adapter from a trained artifact (P4.3) at init.
  * For each (ticker, sig) in ctx.exits where sig.exit_type ∈
    {stop_loss, trailing_stop, single_day_loss, max_hold}:
      - Build features from ctx + ctx.holdings[ticker]
      - p = predictor(features)
      - If p < threshold: remove (ticker, sig) from ctx.exits
        (the position is held — primary signal was a false positive)
  * model_sell, qp_*, panel_conviction NEVER vetoed (they are model
    signals, not path rules — meta-label is an exit-on-noise filter).

§5.13.10 fallback: predictor missing / None → no-op (don't crash prod).
§5.13.1: integration test through pipeline wiring (separate file).
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.exits import ExitSignal, HoldingState  # noqa: E402
from kernel.meta_label.task_meta_label_veto import MetaLabelVetoTask  # noqa: E402


def _holding(*, entry_price=100.0, hwm=110.0, entry_panel_score=0.5,
              panel_score=0.4):
    return HoldingState(
        entry_price=entry_price,
        entry_date=datetime.date(2025, 1, 1),
        high_watermark=hwm,
        entry_panel_score=entry_panel_score,
        panel_score=panel_score,
    )


def _exit(*, exit_type: str, reason: str = "") -> ExitSignal:
    return ExitSignal(
        should_exit=True, reason=reason or exit_type, exit_type=exit_type,
    )


def _ctx(*, exits, predictor=None, threshold=0.5, enabled=True,
         holdings=None, prices=None):
    cfg = {"ranking": {"meta_label": {
        "enabled": enabled, "threshold": threshold,
    }}}
    holdings = holdings or {t: _holding() for (t, _) in exits}
    prices   = prices   or {t: 95.0      for (t, _) in exits}
    return SimpleNamespace(
        config=cfg,
        today=datetime.date(2025, 1, 15),
        exits=list(exits),
        holdings=holdings,
        prices=prices,
        spy_returns=[0.001] * 80,
        regime="BULL_CALM",
        confidence=0.8,
        portfolio_value=95000.0,
        hwm=100000.0,
        candidates=[],
        _meta_label_predictor=predictor,
        counters={},
    )


class TestMetaLabelVetoGuards:

    def test_disabled_block_is_noop(self):
        ctx = _ctx(
            exits=[("AAPL", _exit(exit_type="stop_loss"))],
            predictor=lambda feats: 0.0,   # would veto if enabled
            enabled=False,
        )
        MetaLabelVetoTask().run(ctx)
        assert len(ctx.exits) == 1   # exit preserved (task is off)

    def test_no_predictor_is_noop(self):
        ctx = _ctx(
            exits=[("AAPL", _exit(exit_type="stop_loss"))],
            predictor=None,
            enabled=True,
        )
        MetaLabelVetoTask().run(ctx)
        assert len(ctx.exits) == 1   # § 5.13.10 fallback: don't crash

    def test_missing_predictor_attr_is_noop(self):
        ctx = SimpleNamespace(
            config={"ranking": {"meta_label": {"enabled": True, "threshold": 0.5}}},
            today=datetime.date(2025, 1, 15),
            exits=[("AAPL", _exit(exit_type="stop_loss"))],
            holdings={"AAPL": _holding()},
            prices={"AAPL": 95.0},
            spy_returns=[0.001] * 80,
            regime="BULL_CALM",
            confidence=0.8,
            portfolio_value=95000.0,
            hwm=100000.0,
            candidates=[],
            counters={},
            # no _meta_label_predictor attribute at all
        )
        MetaLabelVetoTask().run(ctx)
        assert len(ctx.exits) == 1


class TestMetaLabelVetoBehavior:

    def test_low_probability_vetoes_path_rule_exit(self):
        # Predictor says exit is wrong (low P=profitable) → veto
        ctx = _ctx(
            exits=[("AAPL", _exit(exit_type="stop_loss"))],
            predictor=lambda feats: 0.20,   # low → veto
            threshold=0.50,
        )
        MetaLabelVetoTask().run(ctx)
        assert len(ctx.exits) == 0
        assert ctx.counters.get("meta_veto") == 1

    def test_high_probability_preserves_path_rule_exit(self):
        ctx = _ctx(
            exits=[("AAPL", _exit(exit_type="stop_loss"))],
            predictor=lambda feats: 0.80,   # high → keep
            threshold=0.50,
        )
        MetaLabelVetoTask().run(ctx)
        assert len(ctx.exits) == 1
        assert ctx.exits[0][1].exit_type == "stop_loss"

    def test_threshold_edge_above_keeps_exit(self):
        ctx = _ctx(
            exits=[("AAPL", _exit(exit_type="stop_loss"))],
            predictor=lambda feats: 0.50,   # exactly at threshold
            threshold=0.50,
        )
        MetaLabelVetoTask().run(ctx)
        # >= threshold → keep (not vetoed)
        assert len(ctx.exits) == 1

    def test_model_sell_never_vetoed(self):
        # model_sell is NOT a path rule; veto must skip it
        ctx = _ctx(
            exits=[("AAPL", _exit(exit_type="model_sell"))],
            predictor=lambda feats: 0.0,   # would veto if applicable
        )
        MetaLabelVetoTask().run(ctx)
        assert len(ctx.exits) == 1
        assert ctx.exits[0][1].exit_type == "model_sell"

    def test_qp_close_never_vetoed(self):
        ctx = _ctx(
            exits=[("AAPL", _exit(exit_type="qp_close"))],
            predictor=lambda feats: 0.0,
        )
        MetaLabelVetoTask().run(ctx)
        assert len(ctx.exits) == 1

    def test_trailing_stop_eligible_for_veto(self):
        ctx = _ctx(
            exits=[("AAPL", _exit(exit_type="trailing_stop"))],
            predictor=lambda feats: 0.10,
            threshold=0.50,
        )
        MetaLabelVetoTask().run(ctx)
        assert len(ctx.exits) == 0

    def test_sdl_eligible_for_veto(self):
        ctx = _ctx(
            exits=[("AAPL", _exit(exit_type="single_day_loss"))],
            predictor=lambda feats: 0.10,
            threshold=0.50,
        )
        MetaLabelVetoTask().run(ctx)
        assert len(ctx.exits) == 0

    def test_max_hold_eligible_for_veto(self):
        ctx = _ctx(
            exits=[("AAPL", _exit(exit_type="max_hold"))],
            predictor=lambda feats: 0.10,
            threshold=0.50,
        )
        MetaLabelVetoTask().run(ctx)
        assert len(ctx.exits) == 0

    def test_mixed_exit_types_only_vetoes_path_rules(self):
        ctx = _ctx(
            exits=[
                ("AAPL", _exit(exit_type="stop_loss")),       # vetoable
                ("MSFT", _exit(exit_type="model_sell")),      # immune
                ("GOOG", _exit(exit_type="trailing_stop")),   # vetoable
                ("AMZN", _exit(exit_type="qp_close")),        # immune
            ],
            predictor=lambda feats: 0.20,
            threshold=0.50,
        )
        MetaLabelVetoTask().run(ctx)
        remaining = sorted(t for (t, _) in ctx.exits)
        assert remaining == ["AMZN", "MSFT"]
        assert ctx.counters.get("meta_veto") == 2

    def test_predictor_receives_features_dict(self):
        seen = []
        def spy_predictor(feats):
            seen.append(feats)
            return 0.99
        ctx = _ctx(
            exits=[("AAPL", _exit(exit_type="stop_loss"))],
            predictor=spy_predictor,
        )
        MetaLabelVetoTask().run(ctx)
        assert len(seen) == 1
        # Features dict should contain core fields
        feats = seen[0]
        assert isinstance(feats, dict)
        for k in ("cum_pnl_pct", "peak_gain_pct", "drawdown_from_peak_pct",
                 "days_held", "trigger_stop_loss", "any_trigger"):
            assert k in feats, f"feature missing: {k}"


class TestMetaLabelVetoNaNGuards:

    def test_predictor_returns_nan_keeps_exit(self):
        # Predictor failed (returned NaN) → fail-safe: keep the exit
        # (don't accidentally veto on garbage prediction)
        import math
        ctx = _ctx(
            exits=[("AAPL", _exit(exit_type="stop_loss"))],
            predictor=lambda feats: math.nan,
            threshold=0.50,
        )
        MetaLabelVetoTask().run(ctx)
        assert len(ctx.exits) == 1   # fail-safe: keep

    def test_predictor_exception_keeps_exit(self):
        # If predictor raises, treat as if it returned NaN: keep the exit
        def boom(feats):
            raise RuntimeError("model crashed")
        ctx = _ctx(
            exits=[("AAPL", _exit(exit_type="stop_loss"))],
            predictor=boom,
        )
        MetaLabelVetoTask().run(ctx)   # must not raise
        assert len(ctx.exits) == 1
