from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
KERNEL = REPO / "backtesting/renquant_104"
if str(KERNEL) not in sys.path:
    sys.path.insert(0, str(KERNEL))

from kernel.portfolio_qp.job_qp import _BuildSourceMapTask  # noqa: E402
from kernel.portfolio_qp.tasks import _qp_buy_admission_block_reason  # noqa: E402


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


def test_qp_allows_prequalified_candidate_with_available_slot() -> None:
    source = SimpleNamespace(ticker="AAA", rank_score=0.56, panel_score=0.01)

    reason = _qp_buy_admission_block_reason(SimpleNamespace(config={}), _env(source=source), "AAA")

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
        assert cfg["ranking"]["kelly_sizing"]["topup_conviction_floor"] >= 0.55, name
