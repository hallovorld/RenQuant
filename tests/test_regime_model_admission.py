import sys
from types import SimpleNamespace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from kernel.panel_pipeline.job_panel_scoring import RegimeModelAdmissionTask
from kernel.selection import CandidateResult


def _cand(ticker: str) -> CandidateResult:
    return CandidateResult(ticker=ticker, raw_score=0.1, rank_score=0.6, rs_score=0.0)


def _metadata(regime: str = "BULL_CALM", *, eligible=True, passed=True) -> dict:
    return {
        "wf_gate_metadata": {
            "trade_monotonicity": {
                "regimes": [{
                    "regime": regime,
                    "eligible": eligible,
                    "passed": passed,
                    "spearman": 0.10,
                }]
            }
        }
    }


def _ctx(metadata: dict, *, regime: str = "BULL_CALM", cfg: dict | None = None):
    return SimpleNamespace(
        candidates=[_cand("AAPL"), _cand("MSFT")],
        config={"ranking": {"panel_scoring": {"regime_admission": cfg or {}}}},
        regime=regime,
        counters={},
        _panel_scorer=SimpleNamespace(metadata=metadata),
    )


def test_regime_admission_allows_current_regime_with_passed_evidence() -> None:
    ctx = _ctx(_metadata("BULL_CALM"))

    RegimeModelAdmissionTask().run(ctx)

    assert [c.ticker for c in ctx.candidates] == ["AAPL", "MSFT"]
    assert ctx._regime_model_admission["ok"] is True


def test_regime_admission_blocks_missing_current_regime_stats() -> None:
    ctx = _ctx(_metadata("BEAR"), regime="BULL_CALM")

    RegimeModelAdmissionTask().run(ctx)

    assert ctx.candidates == []
    assert ctx.counters["regime_admission_blocked"] == 2
    assert ctx._blocked_by_ticker == {
        "AAPL": "regime_admission:no_trade_stats:BULL_CALM",
        "MSFT": "regime_admission:no_trade_stats:BULL_CALM",
    }
    assert len(ctx._full_candidate_snapshot) == 2


def test_regime_admission_blocks_ineligible_regime() -> None:
    ctx = _ctx(_metadata("BULL_VOLATILE", eligible=False), regime="BULL_VOLATILE")

    RegimeModelAdmissionTask().run(ctx)

    assert ctx.candidates == []
    assert ctx._blocked_by_ticker["AAPL"] == "regime_admission:ineligible:BULL_VOLATILE"


def test_regime_admission_can_require_sanity_regime_ic() -> None:
    meta = _metadata("BULL_CALM")
    meta["wf_gate_metadata"]["sanity_regime_ic"] = {
        "regimes": {"BULL_CALM": {"mean_ic": 0.003, "passed": True}}
    }
    ctx = _ctx(
        meta,
        cfg={"require_sanity_regime_ic": True, "min_sanity_regime_ic": 0.02},
    )

    RegimeModelAdmissionTask().run(ctx)

    assert ctx.candidates == []
    assert ctx._blocked_by_ticker["MSFT"] == "regime_admission:weak_sanity_ic:BULL_CALM"


def test_regime_admission_can_be_disabled_for_experiments() -> None:
    ctx = _ctx({}, cfg={"enabled": False})

    RegimeModelAdmissionTask().run(ctx)

    assert [c.ticker for c in ctx.candidates] == ["AAPL", "MSFT"]
