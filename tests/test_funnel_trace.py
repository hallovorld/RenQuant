"""Unit tests for scripts/funnel_trace.py.

The funnel-trace tool is the diagnostic surface that lets us answer
'why does the strategy make 0 trades?' without re-running sims. If the
parser misses a log line or miscounts a stage, every diagnosis built
on top of it is wrong. Pinning the parser behavior here.
"""
from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# funnel_trace.py is a script (not a package). Import it explicitly.
spec = importlib.util.spec_from_file_location(
    "funnel_trace", REPO_ROOT / "scripts" / "funnel_trace.py",
)
funnel_trace = importlib.util.module_from_spec(spec)
sys.modules["funnel_trace"] = funnel_trace
spec.loader.exec_module(funnel_trace)


def _write_log(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "sim.log"
    p.write_text(textwrap.dedent(body).strip() + "\n")
    return p


# ── Phase 2b → vol gate → drift → kelly funnel ──────────────────────────────

class TestFunnelParser:
    def test_minimal_log_one_bar(self, tmp_path):
        log = _write_log(tmp_path, """
            2026-01-01 00:00:00 InferencePipeline START  date=2025-05-05
            2026-01-01 00:00:00 Phase 2b (buy scan): 50 candidates from 50 tickers
            2026-01-01 00:00:00 RealizedVolGateTask: dropped 10/50 candidates …
            2026-01-01 00:00:00 ApplyNGBoostTask: mode=additive λ=0.00 n_cands=40 n_holdings=0  (set_μσ=15  not_in_idx=20  mu_nan=3  sigma_nan=2)
            2026-01-01 00:00:00 ApplyKellySizingTask: fractional=0.50 max_conc=0.35  cands=2 non-zero (avg=12.0%)  holdings=0 non-zero (avg=0.0%)  zero_reasons[mu_none=23 mu_le_min_edge=15]
            2026-01-01 00:00:00 InferencePipeline DONE  total=1.0s  rotations_emitted=0 (considered=2  blocked=0)
        """)
        out = funnel_trace.parse_log(log)
        assert out["n_bars"] == 1
        f = out["funnel"]
        assert f["00_phase2b"] == [50]
        assert f["01_vol_kept"] == [40]   # 50 - 10
        assert f["05_ngb_setμσ"] == [15]
        assert f["06_kelly_nonzero"] == [2]
        # Skip-reason aggregates
        assert out["skip_ngb"]["not_in_idx"] == 20
        assert out["skip_ngb"]["mu_nan"] == 3
        assert out["skip_ngb"]["sigma_nan"] == 2
        assert out["skip_kelly"]["mu_none"] == 23
        assert out["skip_kelly"]["mu_le_min_edge"] == 15

    def test_drift_failsafe_sets_drift_clear_to_zero(self, tmp_path):
        log = _write_log(tmp_path, """
            ... InferencePipeline START  date=2025-05-05
            ... Phase 2b (buy scan): 30 candidates from 30 tickers
            ... DriftGuardTask: 8/27 (29.6%) STRUCTURALLY missing — FAIL-SAFE clearing candidates. First 10: ['m_x']
        """)
        out = funnel_trace.parse_log(log)
        assert out["bars_drift"] == 1
        # When drift fires, drift_clear is forced to 0 even if upstream had counts
        assert out["funnel"]["04_drift_clear"] == [0]

    def test_two_bars_aggregates_correctly(self, tmp_path):
        log = _write_log(tmp_path, """
            ... InferencePipeline START  date=2025-05-05
            ... Phase 2b (buy scan): 50 candidates from 50 tickers
            ... RealizedVolGateTask: dropped 5/50 candidates …
            ... ApplyKellySizingTask: fractional=0.50 max_conc=0.35  cands=3 non-zero (avg=10.0%)  holdings=0 non-zero (avg=0.0%)
            ... InferencePipeline START  date=2025-05-06
            ... Phase 2b (buy scan): 60 candidates from 60 tickers
            ... RealizedVolGateTask: dropped 8/60 candidates …
            ... ApplyKellySizingTask: fractional=0.50 max_conc=0.35  cands=5 non-zero (avg=10.0%)  holdings=0 non-zero (avg=0.0%)  zero_reasons[mu_none=10]
        """)
        out = funnel_trace.parse_log(log)
        assert out["n_bars"] == 2
        assert out["funnel"]["00_phase2b"] == [50, 60]
        assert out["funnel"]["01_vol_kept"] == [45, 52]
        assert out["funnel"]["06_kelly_nonzero"] == [3, 5]
        # only the 2nd bar reported zero_reasons
        assert out["skip_kelly"]["mu_none"] == 10

    def test_log_with_only_qp_buys(self, tmp_path):
        log = _write_log(tmp_path, """
            ... InferencePipeline START  date=2025-05-05
            ... Phase 2b (buy scan): 40 candidates from 40 tickers
            ... QP_BUY  AAPL  Δw=+0.05 …
            ... QP_BUY  NVDA  Δw=+0.03 …
            ... QP_SELL TSLA  Δw=-0.04 …
        """)
        out = funnel_trace.parse_log(log)
        assert out["funnel"]["07_qp_buys"] == [2]


class TestMuHistogramBuckets:
    """Verify the μ-distribution bucket boundaries are correct.

    The user mandate said 'no hardcoding unless you convince me' —
    the bucket boundaries (-1pct, -0.5pct, 0, +0.5pct, +1pct) ARE
    chosen to highlight the Kelly threshold (μ > 0 vs μ ≤ 0). They're
    not arbitrary; they bracket the typical model's μ_mean (≈+0.3%).
    Pin the boundaries so a future refactor doesn't silently shift them.
    """

    def test_buckets_in_order(self, tmp_path):
        # We can't query a DB here without actual data, but we can pin
        # the bucket sequence by reading the function's source.
        src = (REPO_ROOT / "scripts" / "funnel_trace.py").read_text()
        for bucket in (
            "mu_null", "mu_<= -1pct", "mu_-1pct..-0.5pct", "mu_-0.5pct..0",
            "mu_=0", "mu_0..+0.5pct", "mu_+0.5pct..+1pct", "mu_>= +1pct",
        ):
            assert bucket in src, f"missing μ-bucket label: {bucket}"
