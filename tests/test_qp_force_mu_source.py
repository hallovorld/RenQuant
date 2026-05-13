"""AUDIT REGRESSION GUARD — Option A: force μ_QP source (2026-05-12).

Pinned invariants:

1. Default (qp_mu_source="mu" or missing) → no-op. _qp_mu preserved.
2. qp_mu_source="panel_score" → _qp_mu rewritten from candidate.panel_score
   ignoring any NGBoost mu that may have been there.
3. Fallback when panel_score missing → try rank_score.

Reference: doc/AUDIT_2026-05-12_dead_paths.md §NGBoost SUSPECT.
This task validates whether NGBoost σ (kept on candidate objects, used
downstream by Kelly + ApplyKellySizingTask + the QP risk term) provides
real value *independent* of the destructive μ-scale mismatch.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _cand(ticker, mu=None, panel_score=None, rank_score=None):
    c = SimpleNamespace(ticker=ticker)
    if mu is not None:          c.mu = mu
    if panel_score is not None: c.panel_score = panel_score
    if rank_score  is not None: c.rank_score  = rank_score
    return c


def _make_ctx(tickers, src_map, qp_mu=None, **cfg):
    ctx = SimpleNamespace()
    ctx._qp_tickers = tickers
    ctx._qp_mu_source_map = src_map
    ctx._qp_mu = np.asarray(qp_mu, dtype=float) if qp_mu is not None else np.zeros(len(tickers))
    ctx.config = cfg
    return ctx


class TestForceMuSource:

    def test_default_mu_source_no_op(self):
        """qp_mu_source not set → identity."""
        from kernel.portfolio_qp.tasks import ForceMuSourceTask
        ctx = _make_ctx(
            tickers=["AAA", "BBB"],
            src_map={"AAA": _cand("AAA", mu=0.005, panel_score=2.0),
                     "BBB": _cand("BBB", mu=-0.003, panel_score=-1.5)},
            qp_mu=[0.005, -0.003],
        )
        before = ctx._qp_mu.copy()
        ForceMuSourceTask().run(ctx)
        np.testing.assert_array_equal(ctx._qp_mu, before)

    def test_force_panel_score_overrides_mu(self):
        """qp_mu_source='panel_score' → ignore NGBoost mu, use panel_score."""
        from kernel.portfolio_qp.tasks import ForceMuSourceTask
        ctx = _make_ctx(
            tickers=["AAA", "BBB"],
            src_map={"AAA": _cand("AAA", mu=0.005, panel_score=+2.0),
                     "BBB": _cand("BBB", mu=-0.003, panel_score=-1.5)},
            qp_mu=[0.005, -0.003],
            ranking={"qp_mu_source": "panel_score"},
        )
        ForceMuSourceTask().run(ctx)
        np.testing.assert_array_equal(ctx._qp_mu, [+2.0, -1.5])

    def test_force_rank_score_alias(self):
        """qp_mu_source='rank_score' also works (alias)."""
        from kernel.portfolio_qp.tasks import ForceMuSourceTask
        ctx = _make_ctx(
            tickers=["X"],
            src_map={"X": _cand("X", mu=0.001, rank_score=+1.3, panel_score=+1.3)},
            ranking={"qp_mu_source": "rank_score"},
        )
        ForceMuSourceTask().run(ctx)
        assert ctx._qp_mu[0] == pytest.approx(1.3)

    def test_fallback_when_panel_score_missing(self):
        """panel_score missing → try rank_score."""
        from kernel.portfolio_qp.tasks import ForceMuSourceTask
        ctx = _make_ctx(
            tickers=["A"],
            src_map={"A": _cand("A", mu=0.01, rank_score=+0.7)},  # no panel_score
            ranking={"qp_mu_source": "panel_score"},
        )
        ForceMuSourceTask().run(ctx)
        # Should fall back to rank_score
        assert ctx._qp_mu[0] == pytest.approx(0.7)

    def test_unknown_source_no_op(self):
        """Unknown source string → no-op + warning, _qp_mu preserved."""
        from kernel.portfolio_qp.tasks import ForceMuSourceTask
        ctx = _make_ctx(
            tickers=["A"], src_map={"A": _cand("A", mu=0.01)},
            qp_mu=[0.01], ranking={"qp_mu_source": "bogus_field"},
        )
        ForceMuSourceTask().run(ctx)
        np.testing.assert_array_equal(ctx._qp_mu, [0.01])

    def test_missing_candidate_in_map_zero_fills(self):
        """Ticker not in source map → 0.0 entry (rare; defensive)."""
        from kernel.portfolio_qp.tasks import ForceMuSourceTask
        ctx = _make_ctx(
            tickers=["A", "MISSING"],
            src_map={"A": _cand("A", panel_score=1.5)},
            ranking={"qp_mu_source": "panel_score"},
        )
        ForceMuSourceTask().run(ctx)
        assert ctx._qp_mu[0] == pytest.approx(1.5)
        assert ctx._qp_mu[1] == 0.0


class TestWiredInQPJob:

    def test_task_runs_after_mu_build_before_grinold_kahn(self):
        """Pin order: BuildMu → ForceMuSource → GK → BuildWeight."""
        from kernel.portfolio_qp.job_qp import JointPortfolioQPJob
        job = JointPortfolioQPJob()
        names = [type(t).__name__ for t in job.tasks]
        idx_mu = names.index("_BuildMuVectorTask")
        idx_fs = names.index("ForceMuSourceTask")
        idx_gk = names.index("ApplyGrinoldKahnTransformTask")
        assert idx_mu < idx_fs < idx_gk, (
            f"Expected _BuildMuVectorTask < ForceMuSourceTask < "
            f"ApplyGrinoldKahnTransformTask but got mu={idx_mu} fs={idx_fs} gk={idx_gk}"
        )
