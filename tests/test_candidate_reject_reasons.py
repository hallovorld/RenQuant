from __future__ import annotations

import datetime
import sys
from pathlib import Path
from unittest.mock import patch

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.pipeline.context import InferenceContext, TickerInferenceContext
from kernel.pipeline.task_candidates import ScoreThresholdTask


def _make_tctx() -> TickerInferenceContext:
    return TickerInferenceContext(
        ticker="AAPL",
        ohlcv={},
        model={},
        config={},
        today=datetime.date(2026, 1, 2),
        regime="BULL_CALM",
        regime_params={"min_model_score": 0.10},
        exit_params={},
    )


def test_score_threshold_sets_reject_reason():
    tc = _make_tctx()
    tc._rank_score = 0.05  # noqa: SLF001

    stopped = ScoreThresholdTask().run(tc)

    assert stopped is False
    assert tc.candidate_reject_reason == "below_min_model_score"


def test_inference_pipeline_counts_candidate_reject_reasons():
    import kernel.pipeline.pp_inference as pp
    import kernel.panel_pipeline.job_panel_scoring as panel_job

    class _NoOpJob:
        def run(self, _ctx):
            return None

    class _NoOpPanelJob:
        def run(self, _ctx):
            return None

    cfg = {
        "watchlist": ["AAPL", "MSFT"],
        "regime_params": {"BULL_CALM": {}},
        "ranking": {"panel_scoring": {"enabled": False}},
    }
    ctx = InferenceContext(config=cfg, today=datetime.date(2026, 1, 2))
    ctx.models = {"AAPL": {"dummy": True}, "MSFT": {"dummy": True}}
    ctx.ohlcv = {"AAPL": object(), "MSFT": object()}
    ctx.regime = "BULL_CALM"
    ctx.buy_blocked = False
    ctx.bear_only = False

    def _fake_run_parallel(ticker_ctxs, job, max_workers=None, timeout_seconds=None):  # noqa: ARG001
        if isinstance(job, pp.TickerCandidateJob):
            for tc in ticker_ctxs:
                tc.candidate_reject_reason = "model_not_buy"

    with patch.object(pp, "RegimeJob", lambda: _NoOpJob()), \
         patch.object(pp, "DrawdownJob", lambda: _NoOpJob()), \
         patch.object(pp, "BuyGatesJob", lambda: _NoOpJob()), \
         patch.object(pp, "RankingJob", lambda: _NoOpJob()), \
         patch.object(pp, "RotationJob", lambda: _NoOpJob()), \
         patch.object(pp, "SelectionJob", lambda: _NoOpJob()), \
         patch.object(pp, "run_parallel", _fake_run_parallel), \
         patch.object(panel_job, "PanelScoringJob", lambda: _NoOpPanelJob()):
        pp.InferencePipeline().run(ctx)

    assert ctx.counters["candidate_reject_model_not_buy"] == 2
