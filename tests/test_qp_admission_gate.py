from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parent.parent
KERNEL = REPO / "backtesting/renquant_104"
if str(KERNEL) not in sys.path:
    sys.path.insert(0, str(KERNEL))

from kernel.portfolio_qp.job_qp import _BuildSourceMapTask  # noqa: E402
from kernel.portfolio_qp.tasks import (  # noqa: E402
    ApplyExitOnlyTopupGuardTask,
    _qp_buy_admission_block_reason,
)


def _env(
    *,
    holdings: dict | None = None,
    source=None,
    max_positions: int = 8,
) -> dict:
    holdings = holdings or {}
    return {
        "cfg": {
            "qp_admission_gate": {
                "enabled": True,
                "min_rank_score": 0.55,
                "min_panel_score": 0.0,
                "topup_min_rank_score": 0.55,
                "topup_min_panel_score": 0.0,
                "respect_open_slots": True,
            }
        },
        "holdings_set": set(holdings),
        "holdings": holdings,
        "preexisting_exit_tickers": set(),
        "max_positions": max_positions,
        "cands": {"AAA": source} if source is not None else {},
        "score_sources": {"AAA": source} if source is not None else {},
    }


def test_qp_blocks_new_candidate_with_negative_raw_panel_score() -> None:
    source = SimpleNamespace(ticker="AAA", rank_score=0.61, panel_score=-0.01)

    reason = _qp_buy_admission_block_reason(SimpleNamespace(config={}), _env(source=source), "AAA")

    assert reason == "qp_admission_panel"


def test_qp_blocks_high_sigma_candidate_when_configured() -> None:
    source = SimpleNamespace(
        ticker="AAA",
        rank_score=0.61,
        panel_score=0.10,
        sigma=0.55,
    )
    env = _env(source=source)
    env["cfg"]["qp_admission_gate"]["max_sigma"] = 0.40

    reason = _qp_buy_admission_block_reason(
        SimpleNamespace(config={}, regime="BULL_CALM"),
        env,
        "AAA",
    )

    assert reason == "qp_admission_sigma"


def test_qp_sigma_cap_uses_regime_override() -> None:
    source = SimpleNamespace(
        ticker="AAA",
        rank_score=0.61,
        panel_score=0.10,
        sigma=0.55,
    )
    env = _env(source=source)
    env["cfg"]["qp_admission_gate"]["max_sigma"] = 0.40
    env["cfg"]["qp_admission_gate"]["max_sigma_by_regime"] = {"CHOPPY": 0.60}

    reason = _qp_buy_admission_block_reason(
        SimpleNamespace(config={}, regime="CHOPPY"),
        env,
        "AAA",
    )

    assert reason is None


def test_qp_blocks_new_candidate_when_slots_full() -> None:
    source = SimpleNamespace(ticker="AAA", rank_score=0.80, panel_score=0.20)
    holdings = {f"H{i}": object() for i in range(8)}

    reason = _qp_buy_admission_block_reason(
        SimpleNamespace(config={}),
        _env(holdings=holdings, source=source, max_positions=8),
        "AAA",
    )

    assert reason == "qp_admission_no_slot"


def test_qp_blocks_held_topup_below_rank_floor() -> None:
    holding = SimpleNamespace(rank_score=0.54, panel_score=0.10)
    env = _env(holdings={"AAA": holding}, source=None)

    reason = _qp_buy_admission_block_reason(SimpleNamespace(config={}), env, "AAA")

    assert reason == "qp_admission_rank"


def test_qp_blocks_held_topup_when_marked_exit_only() -> None:
    holding = SimpleNamespace(rank_score=0.80, panel_score=0.20)
    env = _env(holdings={"AAA": holding}, source=None)
    env["exit_only_tickers"] = {"AAA"}

    reason = _qp_buy_admission_block_reason(SimpleNamespace(config={}), env, "AAA")

    assert reason == "qp_universe_exit_only"


def test_qp_exit_only_topup_reports_specific_reason() -> None:
    holding = SimpleNamespace(rank_score=0.80, panel_score=0.20)
    env = _env(holdings={"AAA": holding}, source=None)
    env["exit_only_tickers"] = {"AAA"}
    env["exit_only_reasons"] = {"AAA": "regime_admission:failed:BULL_CALM"}

    reason = _qp_buy_admission_block_reason(SimpleNamespace(config={}), env, "AAA")

    assert reason == "regime_admission:failed:BULL_CALM"


def test_qp_allows_prequalified_candidate_with_available_slot() -> None:
    source = SimpleNamespace(ticker="AAA", rank_score=0.56, panel_score=0.01)

    reason = _qp_buy_admission_block_reason(SimpleNamespace(config={}), _env(source=source), "AAA")

    assert reason is None


def test_qp_blocks_new_candidate_below_expected_return_floor() -> None:
    source = SimpleNamespace(
        ticker="AAA",
        rank_score=0.61,
        panel_score=0.10,
        expected_return=0.039,
    )
    env = _env(source=source)
    env["cfg"]["qp_admission_gate"]["min_expected_return"] = 0.04

    reason = _qp_buy_admission_block_reason(SimpleNamespace(config={}), env, "AAA")

    assert reason == "qp_admission_expected_return"


def test_qp_blocks_missing_expected_return_when_floor_configured() -> None:
    source = SimpleNamespace(ticker="AAA", rank_score=0.61, panel_score=0.10)
    env = _env(source=source)
    env["cfg"]["qp_admission_gate"]["min_expected_return"] = 0.04

    reason = _qp_buy_admission_block_reason(SimpleNamespace(config={}), env, "AAA")

    assert reason == "qp_admission_expected_return"


def test_qp_expected_return_floor_uses_regime_override() -> None:
    source = SimpleNamespace(
        ticker="AAA",
        rank_score=0.61,
        panel_score=0.10,
        expected_return=0.025,
    )
    env = _env(source=source)
    env["cfg"]["qp_admission_gate"]["min_expected_return"] = 0.04
    env["cfg"]["qp_admission_gate"]["min_expected_return_by_regime"] = {
        "CHOPPY": 0.02,
    }

    reason = _qp_buy_admission_block_reason(
        SimpleNamespace(config={}, regime="CHOPPY"),
        env,
        "AAA",
    )

    assert reason is None


def test_qp_held_topup_uses_topup_expected_return_floor() -> None:
    holding = SimpleNamespace(
        ticker="AAA",
        rank_score=0.61,
        panel_score=0.10,
        expected_return=0.025,
    )
    env = _env(holdings={"AAA": holding}, source=None)
    env["cfg"]["qp_admission_gate"]["min_expected_return"] = 0.04
    env["cfg"]["qp_admission_gate"]["topup_min_expected_return"] = 0.03

    reason = _qp_buy_admission_block_reason(SimpleNamespace(config={}), env, "AAA")

    assert reason == "qp_admission_expected_return"


def test_qp_blocks_new_candidate_below_expected_return_over_sigma_floor() -> None:
    source = SimpleNamespace(
        ticker="AAA",
        rank_score=0.61,
        panel_score=0.10,
        expected_return=0.04,
        sigma=0.50,
    )
    env = _env(source=source)
    env["cfg"]["qp_admission_gate"]["min_expected_return_over_sigma"] = 0.10

    reason = _qp_buy_admission_block_reason(SimpleNamespace(config={}), env, "AAA")

    assert reason == "qp_admission_expected_return_over_sigma"


def test_qp_expected_return_over_sigma_floor_uses_regime_override() -> None:
    source = SimpleNamespace(
        ticker="AAA",
        rank_score=0.61,
        panel_score=0.10,
        expected_return=0.04,
        sigma=0.50,
    )
    env = _env(source=source)
    env["cfg"]["qp_admission_gate"]["min_expected_return_over_sigma"] = 0.10
    env["cfg"]["qp_admission_gate"]["min_expected_return_over_sigma_by_regime"] = {
        "BULL_CALM": 0.07,
    }

    reason = _qp_buy_admission_block_reason(
        SimpleNamespace(config={}, regime="BULL_CALM"),
        env,
        "AAA",
    )

    assert reason is None


def test_qp_expected_return_over_sigma_falls_back_to_mu() -> None:
    source = SimpleNamespace(
        ticker="AAA",
        rank_score=0.61,
        panel_score=0.10,
        mu=0.06,
        sigma=0.50,
    )
    env = _env(source=source)
    env["cfg"]["qp_admission_gate"]["min_expected_return_over_sigma"] = 0.10

    reason = _qp_buy_admission_block_reason(SimpleNamespace(config={}), env, "AAA")

    assert reason is None


def _ctx_for_source_map(**overrides):
    cfg = {
        "rotation": {
            "joint_actions": {
                "qp_admission_gate": {
                    "enabled": True,
                    "min_rank_score": 0.55,
                    "min_panel_score": 0.0,
                    "topup_min_rank_score": 0.55,
                    "topup_min_panel_score": 0.0,
                    "respect_open_slots": True,
                    "slot_priority": "rank_score",
                }
            }
        },
        "regime_params": {"BULL_CALM": {"max_concurrent_positions": 8}},
    }
    base = dict(
        config=cfg,
        holdings={},
        candidates=[],
        short_candidates=[],
        exits=[],
        regime="BULL_CALM",
        buy_blocked=False,
        skip_buys=False,
        _qp_tickers=[],
        _blocked_by_ticker={},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_qp_solver_universe_excludes_unadmitted_new_candidates() -> None:
    bad = SimpleNamespace(ticker="BAD", rank_score=0.40, panel_score=0.10)
    good = SimpleNamespace(ticker="GOOD", rank_score=0.70, panel_score=0.10)
    ctx = _ctx_for_source_map(candidates=[bad, good], _qp_tickers=["BAD", "GOOD"])

    _BuildSourceMapTask().run(ctx)

    assert ctx._qp_tickers == ["GOOD"]
    assert set(ctx._qp_mu_source_map) == {"GOOD"}
    assert ctx._blocked_by_ticker["BAD"] == "qp_admission_rank"


def test_qp_solver_universe_keeps_held_names_for_trim_even_below_topup_floor() -> None:
    held = SimpleNamespace(ticker="HELD", rank_score=0.40, panel_score=0.10)
    ctx = _ctx_for_source_map(
        holdings={"HELD": held},
        candidates=[held],
        _qp_tickers=["HELD"],
    )

    _BuildSourceMapTask().run(ctx)

    assert ctx._qp_tickers == ["HELD"]
    assert ctx._qp_mu_source_map["HELD"] is held
    assert "HELD" not in ctx._blocked_by_ticker


def test_qp_solver_marks_held_without_current_candidate_exit_only() -> None:
    held = SimpleNamespace(ticker="HELD", rank_score=0.80, panel_score=0.20)
    ctx = _ctx_for_source_map(
        holdings={"HELD": held},
        candidates=[],
        _qp_tickers=["HELD"],
    )

    _BuildSourceMapTask().run(ctx)

    assert ctx._qp_tickers == ["HELD"]
    assert ctx._qp_mu_source_map["HELD"] is held
    assert ctx._qp_exit_only_tickers == {"HELD"}


def test_qp_exit_only_guard_caps_upper_at_current_weight() -> None:
    ctx = SimpleNamespace(
        _qp_exit_only_tickers={"HELD"},
        _qp_tickers=["HELD", "OPEN"],
        _qp_w_upper=np.array([0.20, 0.20]),
        _qp_w_current=np.array([0.07, 0.00]),
        _blocked_by_ticker={},
        counters={},
    )

    ApplyExitOnlyTopupGuardTask().run(ctx)

    assert ctx._qp_w_upper.tolist() == [0.07, 0.20]
    assert ctx._blocked_by_ticker["HELD"] == "qp_universe_exit_only"
    assert ctx.counters["qp_exit_only_topup_guard"] == 1


def test_qp_exit_only_guard_stamps_specific_reason() -> None:
    ctx = SimpleNamespace(
        _qp_exit_only_tickers={"HELD"},
        _qp_exit_only_reasons={"HELD": "regime_admission:failed:BULL_CALM"},
        _qp_tickers=["HELD"],
        _qp_w_upper=np.array([0.20]),
        _qp_w_current=np.array([0.07]),
        _blocked_by_ticker={},
        counters={},
    )

    ApplyExitOnlyTopupGuardTask().run(ctx)

    assert ctx._qp_w_upper.tolist() == [0.07]
    assert ctx._blocked_by_ticker["HELD"] == "regime_admission:failed:BULL_CALM"


def test_qp_solver_universe_excludes_new_candidates_when_slots_are_full() -> None:
    good = SimpleNamespace(ticker="GOOD", rank_score=0.70, panel_score=0.10)
    holdings = {f"H{i}": SimpleNamespace(ticker=f"H{i}") for i in range(8)}
    ctx = _ctx_for_source_map(
        holdings=holdings,
        candidates=[good],
        _qp_tickers=[*holdings, "GOOD"],
    )

    _BuildSourceMapTask().run(ctx)

    assert ctx._qp_tickers == list(holdings)
    assert set(ctx._qp_mu_source_map) == set(holdings)
    assert ctx._blocked_by_ticker["GOOD"] == "qp_admission_no_slot"


def test_qp_solver_universe_budgets_multiple_new_candidates_against_open_slots() -> None:
    first = SimpleNamespace(ticker="FIRST", rank_score=0.70, panel_score=0.10)
    second = SimpleNamespace(ticker="SECOND", rank_score=0.69, panel_score=0.09)
    holdings = {f"H{i}": SimpleNamespace(ticker=f"H{i}") for i in range(7)}
    ctx = _ctx_for_source_map(
        holdings=holdings,
        candidates=[first, second],
        _qp_tickers=[*holdings, "FIRST", "SECOND"],
    )

    _BuildSourceMapTask().run(ctx)

    assert ctx._qp_tickers == [*list(holdings), "FIRST"]
    assert set(ctx._qp_mu_source_map) == {*holdings, "FIRST"}
    assert ctx._blocked_by_ticker["SECOND"] == "qp_admission_no_slot"


def test_qp_solver_universe_allocates_slots_by_kelly_priority() -> None:
    high_rank_high_risk = SimpleNamespace(
        ticker="HIGH_RANK_HIGH_SIGMA",
        rank_score=0.70,
        panel_score=0.20,
        mu=0.046,
        sigma=0.58,
        kelly_target_pct=0.06,
    )
    lower_rank_better_edge = SimpleNamespace(
        ticker="LOWER_RANK_BETTER_KELLY",
        rank_score=0.58,
        panel_score=0.08,
        mu=0.033,
        sigma=0.28,
        kelly_target_pct=0.12,
    )
    holdings = {f"H{i}": SimpleNamespace(ticker=f"H{i}") for i in range(7)}
    ctx = _ctx_for_source_map(
        holdings=holdings,
        candidates=[high_rank_high_risk, lower_rank_better_edge],
        _qp_tickers=[
            *holdings,
            "HIGH_RANK_HIGH_SIGMA",
            "LOWER_RANK_BETTER_KELLY",
        ],
    )
    ctx.config["rotation"]["joint_actions"]["qp_admission_gate"][
        "slot_priority"
    ] = "kelly_target_pct"

    _BuildSourceMapTask().run(ctx)

    assert ctx._qp_tickers == [*list(holdings), "LOWER_RANK_BETTER_KELLY"]
    assert set(ctx._qp_mu_source_map) == {*holdings, "LOWER_RANK_BETTER_KELLY"}
    assert (
        ctx._blocked_by_ticker["HIGH_RANK_HIGH_SIGMA"]
        == "qp_admission_no_slot"
    )


def test_qp_solver_universe_excludes_new_candidates_when_buys_are_gated() -> None:
    good = SimpleNamespace(ticker="GOOD", rank_score=0.70, panel_score=0.10)
    ctx = _ctx_for_source_map(
        candidates=[good],
        _qp_tickers=["GOOD"],
        buy_blocked=True,
    )

    _BuildSourceMapTask().run(ctx)

    assert ctx._qp_tickers == []
    assert ctx._qp_mu_source_map == {}
    assert ctx._blocked_by_ticker["GOOD"] == "buy_blocked"


def test_qp_solver_universe_keeps_short_candidates_outside_buy_admission() -> None:
    long_bad = SimpleNamespace(ticker="BAD", rank_score=0.40, panel_score=-0.10, side="long")
    short_bad = SimpleNamespace(ticker="BAD", rank_score=0.40, panel_score=-0.10, side="short")
    ctx = _ctx_for_source_map(
        candidates=[long_bad],
        short_candidates=[short_bad],
        _qp_tickers=["BAD"],
    )

    _BuildSourceMapTask().run(ctx)

    assert ctx._qp_tickers == ["BAD"]
    assert ctx._qp_mu_source_map["BAD"] is short_bad
    assert ctx._blocked_by_ticker == {}


def test_prod_configs_enable_qp_admission_gate() -> None:
    for name in ("strategy_config.json", "strategy_config.golden.json"):
        cfg = json.loads((KERNEL / name).read_text())
        joint = cfg["rotation"]["joint_actions"]
        gate = joint["qp_admission_gate"]
        assert gate["enabled"] is True, name
        assert gate["min_rank_score"] >= 0.55, name
        assert gate["min_panel_score"] >= 0.0, name
        assert joint["qp_min_invested_pct"] == 0.0, name
        assert joint["qp_cash_drag_lambda"] == 0.0, name
        assert joint["qp_c2_infeasible_policy"] == "strict", name
        assert cfg["ranking"]["kelly_sizing"]["topup_conviction_floor"] >= 0.55, name
